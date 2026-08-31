"""
Coupled MDA group for hybrid VTOL sizing.

The coupling is the mission sizing loop:

    m_tow -> geometry (m2) -> structures (m3) / aero (m4)
          -> power required (m5) -> architecture blending (m5)
          -> battery & fuel sizing -> mass closure -> m_tow

The `m_tow` feedback edge is a genuine algebraic loop; OpenMDAO's
NonlinearBlockGS converges it, playing the same role as the hand-rolled
5-iteration fixed-point loop in hpraptor.core.initial_sizing — except
the derivatives across the converged loop are handled by the framework
rather than being unavailable.
"""

from __future__ import annotations
import numpy as np
import openmdao.api as om

from hpraptor.m5_propulsion.architecture_index import ContinuousArchitectureManager
from hpraptor_mdao.components import (
    GeometryComp, StructuresComp, AeroComp,
    PowerRequiredComp, ArchitectureComp,
    EnergyComp, WeightsComp, ObjectiveComp, TerrainClearanceComp,
    RotorFitComp, AdmissibilityComp, RotorGroupMassComp,
    ASBGeometryComp, ASBAeroComp, aerosandbox_aero_available,
    aerosandbox_aero_reason,
)
from hpraptor_mdao.mission_context import TerrainContext


class HybridVTOLGroup(om.Group):
    """Full coupled sizing model: m2 + m3 + m4 + m5 + energy + weights."""

    def initialize(self):
        self.options.declare("base_manager", types=ContinuousArchitectureManager)
        self.options.declare("payload_kg", default=5.0, types=float)
        self.options.declare("V_cruise", default=30.0, types=float)
        self.options.declare("altitude", default=2900.0, types=float)
        self.options.declare("range_m", default=50000.0, types=float)
        self.options.declare("terrain", default=None, types=TerrainContext,
                             allow_none=True,
                             desc="m1 terrain context; when given it supplies "
                                  "range, pad elevation, VTOL time and the "
                                  "terrain-clearance constraint data")
        self.options.declare("f_empty_nonwing", default=0.25, types=float)
        self.options.declare("penalty_scale", default=200.0, types=float)
        self.options.declare("objective_metric", default="primary_energy",
                             values=("primary_energy", "energy_mass", "shaft_energy"),
                             desc="What the optimizer minimises; see ObjectiveComp")
        self.options.declare("maxiter", default=60, types=int)
        self.options.declare("geometry_source", default="analytical",
                             values=("analytical", "aerosandbox"),
                             desc="'aerosandbox' assembles the real 3D airplane "
                                  "and measures wetted areas instead of "
                                  "estimating them from the planform.")
        self.options.declare("aero_source", default="analytical",
                             values=("analytical", "aerosandbox"),
                             desc="'aerosandbox' runs AeroBuildup for the drag "
                                  "polar. Falls back to the analytical polar, "
                                  "with a printed notice, where AeroSandbox's "
                                  "CasADi backend cannot load.")
        self.options.declare("asb_solver", default="aerobuildup",
                             values=("aerobuildup", "vlm"))
        self.options.declare("coupled", default=False, types=bool,
                             desc="True when this group sits inside the coupled "
                                  "sizing+trajectory problem. The analytical SOC "
                                  "margin and energy objective are then superseded "
                                  "by the integrated ones, so they are renamed "
                                  "rather than promoted into a name collision.")

    def setup(self):
        opt = self.options

        terrain = opt["terrain"]
        range_m = opt["range_m"] if terrain is None else terrain.range_m
        h_origin = 2900.0 if terrain is None else terrain.h_origin_m
        t_vtol = 95.0 if terrain is None else terrain.t_vtol_total_s
        # Start at the lowest terrain-legal altitude; the optimizer moves up
        # only if something else pays for it.
        alt0 = opt["altitude"] if terrain is None else terrain.h_cruise_min_m

        # Flight condition and payload, shared by several disciplines.
        # `altitude` is an IndepVarComp output specifically so it can be
        # declared a design variable: cruise altitude is now traded against
        # climb energy, bounded below by terrain clearance (g5).
        ivc = om.IndepVarComp()
        ivc.add_output("V_cruise", val=opt["V_cruise"], units="m/s")
        ivc.add_output("altitude", val=alt0, units="m")
        self.add_subsystem("flight_condition", ivc,
                           promotes_outputs=["V_cruise", "altitude"])

        geo_outputs = ["S_ref", "span", "chord_mean", "wetted_wing",
                       "fuse_length", "fuse_diameter", "wetted_fuse",
                       "rotor_diameter", "A_rotor", "S_h", "S_v", "tail_arm"]
        if opt["geometry_source"] == "aerosandbox":
            # Measures wetted areas off the assembled surfaces. Costs ~39 ms
            # per call against the analytical component's microseconds, and
            # must be finite-differenced rather than complex-stepped.
            geo_comp = ASBGeometryComp()
            geo_outputs = geo_outputs + ["wetted_total", "V_h"]
        else:
            geo_comp = GeometryComp()

        self.add_subsystem(
            "geometry", geo_comp,
            promotes_inputs=["m_tow", "wing_loading", "AR", "disk_loading"],
            promotes_outputs=geo_outputs,
        )

        self.add_subsystem(
            "structures", StructuresComp(f_empty_nonwing=opt["f_empty_nonwing"]),
            promotes_inputs=["S_ref", "AR", "m_tow", "t_spar_mm"],
            promotes_outputs=["m_wing_structure", "spar_mass", "g4_stress_margin", "m_empty"],
        )

        use_asb_aero = opt["aero_source"] == "aerosandbox"
        if use_asb_aero and not aerosandbox_aero_available():
            print("  [WARN] aero_source='aerosandbox' requested but AeroSandbox's "
                  "aerodynamic solvers cannot run here:")
            print(f"         {aerosandbox_aero_reason()[:100]}")
            print("         Falling back to the analytical drag polar. Geometry "
                  "is unaffected.")
            use_asb_aero = False

        if use_asb_aero:
            self.add_subsystem(
                "aero", ASBAeroComp(solver=opt["asb_solver"]),
                promotes_inputs=["m_tow", "wing_loading", "AR", "disk_loading",
                                 "V_cruise", "altitude"],
                promotes_outputs=["C_D0", "C_L_cruise", "C_D_cruise", "L_D",
                                  "D_cruise", "g1_cl_margin", "alpha_trim"],
            )
        else:
            self.add_subsystem(
                "aero", AeroComp(),
                promotes_inputs=["S_ref", "AR", "fuse_length", "fuse_diameter",
                                 "wetted_wing", "wetted_fuse", "chord_mean",
                                 "m_tow", "V_cruise", "altitude"],
                promotes_outputs=["C_D0", "C_L_cruise", "C_D_cruise", "L_D",
                                  "D_cruise", "g1_cl_margin"],
            )

        self.add_subsystem(
            "power", PowerRequiredComp(),
            promotes_inputs=["m_tow", "A_rotor", "D_cruise", "V_cruise", "altitude"],
            promotes_outputs=["P_hover", "P_cruise", "P_climb", "P_motor_total"],
        )

        self.add_subsystem(
            "architecture", ArchitectureComp(base_manager=opt["base_manager"]),
            promotes_inputs=["z_arch", "P_cruise", "P_motor_total", "k_electric", "altitude"],
            promotes_outputs=["arch_weights", "m_propulsion", "fuel_flow",
                              "P_elec_bus", "fuel_lhv", "fuel_capable",
                              "penalty_discreteness"],
        )

        self.add_subsystem(
            "energy",
            EnergyComp(range_m=range_m, h_origin_m=h_origin, t_vtol_total=t_vtol),
            promotes_inputs=["P_hover", "P_cruise", "P_climb", "L_D", "m_tow",
                             "fuel_lhv", "fuel_capable", "V_cruise", "altitude",
                             "k_electric", "m_battery", "m_fuel"],
            promotes_outputs=["E_battery_wh", "E_fuel_wh", "m_fuel_carried",
                              "E_vtol_wh", "E_climb_wh", "E_cruise_wh", "t_climb_s",
                              "k_effective", "energy_mission_wh",
                              "energy_available_wh", "SOC_final",
                              "P_battery_peak_w", "E_battery_used_wh",
                              "g2_energy_margin", "g6_battery_power"]
                             + ([("g3_soc_margin", "g3_soc_margin_analytic")]
                                if opt["coupled"] else ["g3_soc_margin"]),
        )

        # m1 terrain -> cruise-altitude constraint. Only added when a terrain
        # context exists; without a DEM there is nothing to clear.
        if terrain is not None:
            self.add_subsystem(
                "terrain_clearance",
                TerrainClearanceComp(
                    h_terrain_max_m=terrain.h_terrain_max_m,
                    clearance_cruise_m=terrain.clearance_cruise_m,
                ),
                promotes_inputs=["altitude"],
                promotes_outputs=["agl_cruise", "g5_terrain_clearance"],
            )

        # Geometric admissibility — keeps disk loading off its lower bound
        # for a physical reason rather than an arbitrary one.
        self.add_subsystem(
            "rotor_fit", RotorFitComp(),
            promotes_inputs=["rotor_diameter", "span", "fuse_length"],
            promotes_outputs=["rotor_span_required", "g7_rotor_fit"],
        )

        self.add_subsystem(
            "admissibility", AdmissibilityComp(),
            promotes_inputs=["chord_mean", "V_cruise", "altitude"],
            promotes_outputs=["Re_cruise", "g8_reynolds"],
        )

        self.add_subsystem(
            "rotor_mass", RotorGroupMassComp(),
            promotes_inputs=["A_rotor"],
            promotes_outputs=["m_rotor_group"],
        )

        # Closes the loop: its m_tow output is promoted to the same name the
        # disciplines above consume as an input.
        self.add_subsystem(
            "weights", WeightsComp(payload_kg=opt["payload_kg"]),
            promotes_inputs=["m_empty", "m_propulsion", "m_battery",
                             "m_rotor_group",
                             ("m_fuel", "m_fuel_carried")],
            promotes_outputs=["m_tow"],
        )

        self.add_subsystem(
            "objective", ObjectiveComp(penalty_scale=opt["penalty_scale"],
                                       metric=opt["objective_metric"]),
            promotes_inputs=["energy_mission_wh", "E_battery_used_wh",
                             "m_fuel_carried", "fuel_lhv", "m_battery",
                             "m_propulsion", "penalty_discreteness"],
            promotes_outputs=(
                [("objective", "objective_analytic"),
                 ("energy_primary_wh", "energy_primary_wh_analytic"),
                 "energy_system_mass"]
                if opt["coupled"] else
                ["objective", "energy_primary_wh", "energy_system_mass"]),
        )

        # m_battery and m_fuel are design variables consumed by several
        # components, so their defaults must be declared once at group level
        # rather than left to whichever component happens to declare first.
        self.set_input_defaults("m_battery", val=1.0, units="kg")
        self.set_input_defaults("m_fuel", val=0.2, units="kg")
        # Shared design variables are consumed by several components whose own
        # declared defaults differ. Setting them once here is what stops
        # OpenMDAO calling the promoted value ambiguous.
        self.set_input_defaults("AR", val=12.0)
        self.set_input_defaults("wing_loading", val=300.0, units="N/m**2")
        self.set_input_defaults("disk_loading", val=120.0, units="N/m**2")

        # Fixed-point iteration over the mass-closure loop. Aitken relaxation
        # damps the oscillation that a bare Gauss-Seidel shows when the
        # rotor-group mass responds strongly to MTOW.
        #
        # Inside the coupled problem the driver's line search visits far more
        # extreme iterates than a standalone sizing run ever does, so the
        # loop needs a bigger iteration budget. It does NOT want Newton:
        # tried, and a Newton step drives wing area negative, after which
        # sqrt(S*AR) is NaN and the whole Jacobian is poisoned. Gauss-Seidel
        # stays inside physical territory by construction because every
        # iterate is a real evaluation of the disciplines, which matters far
        # more here than per-iteration cost.
        self.nonlinear_solver = om.NonlinearBlockGS(
            maxiter=200 if opt["coupled"] else opt["maxiter"],
            atol=1e-8, rtol=1e-8, use_aitken=True, iprint=0,
            err_on_non_converge=False,
        )
        self.linear_solver = om.DirectSolver()
