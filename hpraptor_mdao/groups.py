"""
Coupled MDA group for hybrid VTOL sizing.

The coupling is the mission sizing loop:

    m_tow -> geometry (m2) -> structures (m3) / aero (m4)
          -> power required + motor ratings (m5)
          -> architecture: converter ratings, masses (m5)
          -> cruise split at each mission point (m5)
          -> mission energy, segment by segment (energy)
          -> mass closure -> m_tow

The `m_tow` feedback edge is the only algebraic loop; NonlinearBlockGS
converges it.

Two mission points
------------------
* the CORRIDOR mission — the NASADEM route between the two hospitals, flown
  segment by segment over its real terrain profile. Its primary energy is
  the objective, and it carries every constraint.
* the DESIGN-RANGE capability mission — the YAML's ``design_range_km``,
  flown with the same pads and terrain profile and the cruise stretched to
  that ground distance. It only constrains: the storage the optimizer buys
  must also cover it, with the same reserves. It exists when the design
  range exceeds the corridor; it used to be read from the YAML and then
  silently overwritten by the corridor length.

Each point has its own battery/fuel split (``k_electric``,
``k_electric_design``): a hybrid flying 13.6 km and the same hybrid flying
50 km do not use their fuel path the same way.
"""

from __future__ import annotations
import numpy as np
import openmdao.api as om

from hpraptor.m5_propulsion.architecture_index import ContinuousArchitectureManager
from hpraptor.m5_propulsion.technology import DEFAULT_TECHNOLOGY, PowertrainTechnology
from hpraptor_mdao.components import (
    GeometryComp, StructuresComp, AeroComp,
    PowerRequiredComp, ArchitectureComp, CruiseSplitComp,
    EnergyComp, WeightsComp, ObjectiveComp, TerrainClearanceComp,
    RotorFitComp, AdmissibilityComp, RotorGroupMassComp,
    ASBGeometryComp, ASBAeroComp, aerosandbox_aero_available,
    aerosandbox_aero_reason,
)
from hpraptor_mdao.mission_context import TerrainContext

#: EnergyComp outputs that the design-range point exposes, renamed.
DESIGN_POINT_OUTPUTS = ("g3_soc_margin", "g10_fuel_energy", "SOC_final",
                        "E_battery_used_wh", "m_fuel_burned", "t_mission_s",
                        "energy_mission_wh")


def default_context(range_m: float = 50000.0, altitude: float = 2900.0) -> TerrainContext:
    """A flat, still-air mission for runs without a DEM (``--no-terrain``)."""
    return TerrainContext(
        range_m=range_m, h_origin_m=altitude, h_dest_m=altitude,
        h_terrain_max_m=altitude - 100.0, clearance_cruise_m=100.0,
        clearance_min_m=50.0, vtol_climb_m=100.0, vtol_descent_m=100.0,
        t_vtol_total_s=73.3, dem_source="none", h_terrain_mean_m=altitude - 100.0,
    )


class HybridVTOLGroup(om.Group):
    """Full coupled sizing model: m2 + m3 + m4 + m5 + mission energy + weights."""

    def initialize(self):
        self.options.declare("base_manager", types=ContinuousArchitectureManager)
        self.options.declare("technology", default=DEFAULT_TECHNOLOGY,
                             types=PowertrainTechnology)
        self.options.declare("payload_kg", default=5.0, types=float)
        self.options.declare("V_cruise", default=30.0, types=float)
        self.options.declare("altitude", default=2900.0, types=float)
        self.options.declare("range_m", default=50000.0, types=float)
        self.options.declare("terrain", default=None, types=TerrainContext,
                             allow_none=True,
                             desc="m1 context; supplies the corridor, its terrain, "
                                  "the wind, the reserves and the design range")
        self.options.declare("f_empty_nonwing", default=0.25, types=float)
        self.options.declare("penalty_scale", default=1.0, types=float,
                             desc="Relative discreteness penalty mu")
        self.options.declare("objective_metric", default="primary_energy",
                             values=("primary_energy", "energy_mass", "shaft_energy"),
                             desc="What the optimizer minimises; see ObjectiveComp")
        self.options.declare("maxiter", default=60, types=int)
        self.options.declare("geometry_source", default="analytical",
                             values=("analytical", "aerosandbox"))
        self.options.declare("aero_source", default="analytical",
                             values=("analytical", "aerosandbox"))
        self.options.declare("asb_solver", default="aerobuildup",
                             values=("aerobuildup", "vlm"))
        self.options.declare("design_point", default=True, types=bool,
                             desc="Include the design-range capability mission "
                                  "when the context has one")
        self.options.declare("coupled", default=False, types=bool,
                             desc="True inside the coupled sizing+trajectory "
                                  "problem: the analytical SOC margin and "
                                  "objective are then renamed, since the "
                                  "integrated trajectory supersedes them.")

    def setup(self):
        opt = self.options
        tech = opt["technology"]
        mgr = opt["base_manager"]

        terrain = opt["terrain"]
        ctx = terrain if terrain is not None else default_context(opt["range_m"], opt["altitude"])
        alt0 = opt["altitude"] if terrain is None else terrain.h_cruise_min_m
        self._ctx = ctx

        ivc = om.IndepVarComp()
        ivc.add_output("V_cruise", val=opt["V_cruise"], units="m/s")
        ivc.add_output("altitude", val=alt0, units="m")
        self.add_subsystem("flight_condition", ivc,
                           promotes_outputs=["V_cruise", "altitude"])

        # ── m2 geometry ──────────────────────────────────────────────────
        geo_outputs = ["S_ref", "span", "chord_mean", "chord_root", "mac",
                       "e_oswald", "wetted_wing",
                       "fuse_length", "fuse_diameter", "wetted_fuse",
                       "rotor_diameter", "A_rotor", "S_h", "S_v", "tail_arm",
                       "wetted_tail", "chord_tail", "boom_length", "wetted_booms"]
        if opt["geometry_source"] == "aerosandbox":
            geo_comp = ASBGeometryComp()
            geo_outputs = geo_outputs + ["wetted_total", "V_h"]
        else:
            geo_comp = GeometryComp()
        self.add_subsystem(
            "geometry", geo_comp,
            promotes_inputs=["m_tow", "wing_loading", "AR", "disk_loading", "taper_ratio"],
            promotes_outputs=geo_outputs,
        )

        # ── m3 structures ────────────────────────────────────────────────
        self.add_subsystem(
            "structures", StructuresComp(f_empty_nonwing=opt["f_empty_nonwing"]),
            promotes_inputs=["S_ref", "AR", "m_tow", "t_spar_mm", "taper_ratio",
                             "V_cruise", "altitude"],
            promotes_outputs=["m_wing_structure", "spar_mass", "g4_stress_margin",
                              "m_empty", "n_gust", "n_limit"],
        )

        # ── m4 aero ──────────────────────────────────────────────────────
        use_asb_aero = opt["aero_source"] == "aerosandbox"
        if use_asb_aero and not aerosandbox_aero_available():
            print("  [WARN] aero_source='aerosandbox' requested but AeroSandbox's "
                  "aerodynamic solvers cannot run here:")
            print(f"         {aerosandbox_aero_reason()[:100]}")
            print("         Falling back to the analytical drag polar.")
            use_asb_aero = False
        if use_asb_aero:
            self.add_subsystem(
                "aero", ASBAeroComp(solver=opt["asb_solver"]),
                promotes_inputs=["m_tow", "wing_loading", "AR", "disk_loading",
                                 "taper_ratio", "V_cruise", "altitude"],
                promotes_outputs=["C_D0", "C_L_cruise", "C_D_cruise", "L_D",
                                  "D_cruise", "g1_cl_margin", "alpha_trim", "C_D0_vtol"],
            )
        else:
            self.add_subsystem(
                "aero", AeroComp(),
                promotes_inputs=["S_ref", "AR", "fuse_length", "fuse_diameter",
                                 "wetted_wing", "wetted_fuse", "chord_mean",
                                 "e_oswald", "m_tow", "V_cruise", "altitude",
                                 "wetted_tail", "chord_tail", "wetted_booms",
                                 "boom_length", "A_rotor", "rotor_diameter"],
                promotes_outputs=["C_D0", "C_D0_vtol", "C_L_cruise", "C_D_cruise",
                                  "L_D", "D_cruise", "g1_cl_margin"],
            )

        # ── m5 power required and motor ratings ──────────────────────────
        self.add_subsystem(
            "power",
            PowerRequiredComp(technology=tech, h_hover_m=float(ctx.h_hover_m),
                              fw_climb_angle_deg=float(ctx.fw_climb_angle_deg)),
            promotes_inputs=["m_tow", "A_rotor", "D_cruise", "V_cruise", "altitude"],
            promotes_outputs=["P_hover", "P_cruise", "P_climb", "P_lift_rated",
                              "P_cm_rated", "P_motor_total"],
        )

        # ── m5 architecture relaxation ───────────────────────────────────
        self.add_subsystem(
            "architecture", ArchitectureComp(base_manager=mgr, technology=tech),
            promotes_inputs=["z_arch", "P_cruise", "P_lift_rated", "P_cm_rated",
                             "altitude", "m_fuel"],
            promotes_outputs=["arch_weights", "P_conv_rated", "m_propulsion",
                              "m_fuel_system", "fuel_lhv", "eta_fuel_cruise",
                              "fuel_capable", "penalty_discreteness"],
        )

        split_in = ["arch_weights", "P_cruise", "altitude", "P_cm_rated",
                    "P_lift_rated", "P_conv_rated"]
        self.add_subsystem(
            "cruise_split", CruiseSplitComp(base_manager=mgr, technology=tech),
            promotes_inputs=split_in + ["k_electric"],
            promotes_outputs=["P_batt_cruise", "fuel_flow_cruise"],
        )

        # ── Mission energy: the corridor ─────────────────────────────────
        energy_in = ["m_tow", "A_rotor", "S_ref", "AR", "e_oswald", "C_D0",
                     "V_cruise", "altitude", "P_cruise", "P_climb", "P_lift_rated",
                     "P_cm_rated", "fuel_capable", "fuel_lhv", "m_battery", "m_fuel"]
        energy_out = ["E_battery_wh", "E_fuel_wh", "E_vtol_wh", "E_transition_wh",
                      "E_climb_wh", "E_cruise_wh", "E_descent_wh", "t_climb_s",
                      "t_cruise_s", "t_mission_s", "d_cruise_m", "headwind_cruise",
                      "k_effective", "energy_mission_wh", "energy_available_wh",
                      "SOC_final", "P_battery_peak_w", "E_battery_used_wh",
                      "E_battery_terminal_wh", "m_fuel_carried", "m_fuel_burned",
                      "E_fuel_burned_wh", "g2_energy_margin", "g6_battery_power",
                      "g10_fuel_energy"]
        energy_out = energy_out + ([("g3_soc_margin", "g3_soc_margin_analytic")]
                                   if opt["coupled"] else ["g3_soc_margin"])
        self.add_subsystem(
            "energy",
            EnergyComp(technology=tech, **ctx.mission_options()),
            promotes_inputs=energy_in + ["P_batt_cruise", "fuel_flow_cruise"],
            promotes_outputs=energy_out,
        )

        # ── Mission energy: the design-range capability point ────────────
        self.has_design_point = bool(opt["design_point"] and ctx.has_design_point)
        if self.has_design_point:
            self.add_subsystem(
                "cruise_split_design", CruiseSplitComp(base_manager=mgr, technology=tech),
                promotes_inputs=split_in + [("k_electric", "k_electric_design")],
                promotes_outputs=[("P_batt_cruise", "P_batt_cruise_design"),
                                  ("fuel_flow_cruise", "fuel_flow_cruise_design")],
            )
            self.add_subsystem(
                "energy_design",
                EnergyComp(technology=tech,
                           **ctx.mission_options(range_m=ctx.design_range_m)),
                promotes_inputs=energy_in + [("P_batt_cruise", "P_batt_cruise_design"),
                                             ("fuel_flow_cruise", "fuel_flow_cruise_design")],
                promotes_outputs=[(n, n + "_design") for n in DESIGN_POINT_OUTPUTS],
            )

        # ── m1 terrain -> cruise-altitude constraint ─────────────────────
        if terrain is not None:
            self.add_subsystem(
                "terrain_clearance",
                TerrainClearanceComp(h_terrain_max_m=terrain.h_terrain_max_m,
                                     clearance_cruise_m=terrain.clearance_cruise_m),
                promotes_inputs=["altitude"],
                promotes_outputs=["agl_cruise", "g5_terrain_clearance"],
            )

        self.add_subsystem(
            "rotor_fit", RotorFitComp(),
            promotes_inputs=["rotor_diameter", "span", "fuse_diameter"],
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

        # Closes the loop.
        self.add_subsystem(
            "weights", WeightsComp(payload_kg=opt["payload_kg"]),
            promotes_inputs=["m_empty", "m_propulsion", "m_battery", "m_rotor_group",
                             ("m_fuel", "m_fuel_carried")],
            promotes_outputs=["m_tow"],
        )

        self.add_subsystem(
            "objective", ObjectiveComp(penalty_scale=opt["penalty_scale"],
                                       metric=opt["objective_metric"]),
            promotes_inputs=["energy_mission_wh", "E_battery_used_wh",
                             "m_fuel_burned", "m_fuel_carried", "fuel_lhv",
                             "m_battery", "m_propulsion", "penalty_discreteness"],
            promotes_outputs=(
                [("objective", "objective_analytic"),
                 ("energy_primary_wh", "energy_primary_wh_analytic"),
                 "energy_system_mass"]
                if opt["coupled"] else
                ["objective", "energy_primary_wh", "energy_system_mass"]),
        )

        # Shared design variables: one default each, declared at group level.
        self.set_input_defaults("m_battery", val=1.0, units="kg")
        self.set_input_defaults("m_fuel", val=0.2, units="kg")
        self.set_input_defaults("AR", val=12.0)
        self.set_input_defaults("wing_loading", val=300.0, units="N/m**2")
        self.set_input_defaults("disk_loading", val=120.0, units="N/m**2")
        self.set_input_defaults("taper_ratio", val=1.0)
        self.set_input_defaults("k_electric", val=0.5)
        if self.has_design_point:
            self.set_input_defaults("k_electric_design", val=0.5)

        # Fixed-point iteration over the mass-closure loop, with Aitken
        # relaxation. Newton was tried and drives wing area negative on the
        # first step from a poor guess; Gauss-Seidel evaluates only physical
        # iterates.
        self.nonlinear_solver = om.NonlinearBlockGS(
            maxiter=200 if opt["coupled"] else opt["maxiter"],
            atol=1e-10, rtol=1e-10, use_aitken=True, iprint=0,
            err_on_non_converge=False,
        )
        self.linear_solver = om.DirectSolver()
