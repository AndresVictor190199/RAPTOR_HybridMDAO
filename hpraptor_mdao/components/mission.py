"""
Mission energy and storage sizing, plus the mass-closure component.

EnergyComp flies the design mission segment by segment and sizes the
battery and fuel against it. WeightsComp closes the mass loop — its `m_tow`
output feeds back to m2/m3, and that feedback edge is what the group's
nonlinear solver converges.

The mission (EnergyComp)
------------------------
Seven segments, the same sequence the dymos trajectory flies, each with the
physics that governs it:

    1  VTOL climb        axial momentum theory with the climb term T*V_c
    2  transition fwd    rotor thrust handed to the wing as speed builds;
                         Glauert edgewise inflow; propeller accelerates
    3  wingborne climb   to cruise altitude at the climb angle
    4  cruise            the architecture's own battery/fuel split
    5  wingborne descent at the same angle; the propeller only makes up
                         the drag gravity does not
    6  transition back   rotors take the weight back as the wing slows
    7  VTOL descent      Leishman vortex-ring-state fit

All of it at ground speed — airspeed less the headwind the m1 wind model
gives along the route's true bearing — and all of it charged through the
motor, ESC and battery losses, at the ratings the mass model bought.

What this replaced: hover power for the whole VTOL time with no climb work
(lifting the aircraft 100 m cost nothing beyond hovering); a climb from the
pad that double-counted the VTOL leg; no descent (the 830 m drop to the
Cumbaya pad was flown as level cruise); no transitions; no wind; cruise and
climb drawn from the battery at SHAFT power with no motor loss; and fuel
converted to shaft energy through a separately estimated efficiency rather
than the architecture's own fuel flow.
"""

from __future__ import annotations
import numpy as np
import openmdao.api as om

from hpraptor.m5_propulsion.smooth import smooth_max, smooth_min
from hpraptor.m5_propulsion import battery_catalogue as batcat
from hpraptor.m5_propulsion import rotor_power as rp
from hpraptor.m5_propulsion.technology import DEFAULT_TECHNOLOGY, PowertrainTechnology
from hpraptor.m6_dynamics.eom_np import headwind

G = 9.80665

#: 5-point Gauss-Legendre rule on [0, 1], for the transition integrals.
_GL_X = 0.5 * (1.0 + np.array([-0.9061798459386640, -0.5384693101056831, 0.0,
                               0.5384693101056831, 0.9061798459386640]))
_GL_W = 0.5 * np.array([0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
                        0.4786286704993665, 0.2369268850561891])


class EnergyComp(om.ExplicitComponent):
    """The design mission, flown segment by segment, against the storage bought."""

    def initialize(self):
        o = self.options
        o.declare("cell", default=batcat.DEFAULT_CELL, types=str,
                  desc="Catalogue key for the battery cell format")
        o.declare("pack_overhead", default=batcat.DEFAULT_PACK_OVERHEAD, types=float)
        o.declare("technology", default=DEFAULT_TECHNOLOGY, types=PowertrainTechnology)
        o.declare("reserve_soc", default=0.15, types=float)
        o.declare("fuel_reserve", default=0.10, types=float)
        o.declare("range_m", default=50000.0, types=float,
                  desc="Ground distance origin -> destination for this mission")
        o.declare("h_origin_m", default=2900.0, types=float,
                  desc="Departure pad elevation AMSL, from the DEM")
        o.declare("h_dest_m", default=2900.0, types=float,
                  desc="Arrival pad elevation AMSL, from the DEM")
        o.declare("h_ground_mean_m", default=2900.0, types=float,
                  desc="Mean terrain under the route, for the cruise wind AGL")
        o.declare("vtol_climb_m", default=100.0, types=float)
        o.declare("vtol_descent_m", default=100.0, types=float)
        o.declare("vtol_climb_rate", default=3.0, types=float, desc="m/s")
        o.declare("vtol_descent_rate", default=2.5, types=float, desc="m/s")
        o.declare("fw_climb_angle_deg", default=8.0, types=float)
        o.declare("fw_descent_angle_deg", default=8.0, types=float)
        o.declare("transition_accel", default=1.5, types=float,
                  desc="Mean acceleration in the forward transition [m/s^2]")
        o.declare("transition_decel", default=1.5, types=float,
                  desc="Mean deceleration in the back transition [m/s^2]")
        o.declare("eta_prop", default=0.75, types=float)
        o.declare("kappa_i", default=1.15, types=float)
        o.declare("sigma_rotor", default=0.07, types=float)
        o.declare("C_d_blade", default=0.012, types=float)
        o.declare("V_tip", default=120.0, types=float)
        # Wind along the route (m1). Zero speed gives still air.
        o.declare("wind_speed_ref", default=0.0, types=float)
        o.declare("wind_from_deg", default=0.0, types=float)
        o.declare("track_deg", default=0.0, types=float,
                  desc="True bearing origin -> destination")
        o.declare("wind_z0", default=0.1, types=float)
        o.declare("wind_h_ref", default=10.0, types=float)
        o.declare("wind_log_profile", default=True, types=bool)
        # Constant smoothing scales (must not depend on the iterate, or
        # complex step and finite differences disagree on the derivative).
        o.declare("mass_scale_kg", default=1.0, types=float)
        o.declare("fuel_ref_kg", default=0.1, types=float,
                  desc="Normalisation of the fuel-balance constraint")

    def setup(self):
        a = self.add_input
        a("m_tow", val=10.0, units="kg")
        a("A_rotor", val=1.0, units="m**2")
        a("S_ref", val=0.3, units="m**2")
        a("AR", val=15.0)
        a("e_oswald", val=0.8)
        a("C_D0", val=0.025)
        a("V_cruise", val=30.0, units="m/s")
        a("altitude", val=3100.0, units="m", desc="Cruise altitude AMSL (design variable)")
        a("P_cruise", val=200.0, units="W", desc="Cruise shaft power")
        a("P_climb", val=700.0, units="W", desc="Wingborne climb shaft power")
        a("P_lift_rated", val=1500.0, units="W")
        a("P_cm_rated", val=800.0, units="W")
        a("P_batt_cruise", val=230.0, units="W",
          desc="Battery terminal power at cruise, from the architecture split")
        a("fuel_flow_cruise", val=0.0, units="kg/s",
          desc="Fuel flow at cruise, from the architecture split")
        a("fuel_capable", val=1.0, desc="Blended ability to burn fuel; 0 for all-electric")
        a("fuel_lhv", val=43.0e6, units="J/kg")
        # Storage masses are DESIGN VARIABLES. The optimizer buys the energy
        # and the reserve; g3/g6/g10 are what make it buy enough.
        a("m_battery", val=1.0, units="kg", desc="Battery pack mass (design variable)")
        a("m_fuel", val=0.3, units="kg", desc="Usable fuel mass (design variable)")

        o = self.add_output
        o("E_battery_wh", val=100.0, units="W*h", desc="Installed pack energy")
        o("E_fuel_wh", val=0.0, units="W*h", desc="Chemical energy of the fuel carried")
        o("E_vtol_wh", val=20.0, units="W*h", desc="Shaft energy, both VTOL legs")
        o("E_transition_wh", val=5.0, units="W*h", desc="Shaft energy, both transitions")
        o("E_climb_wh", val=10.0, units="W*h", desc="Shaft energy, wingborne climb")
        o("E_cruise_wh", val=20.0, units="W*h", desc="Shaft energy, cruise")
        o("E_descent_wh", val=0.0, units="W*h", desc="Shaft energy, wingborne descent")
        o("t_climb_s", val=60.0, units="s")
        o("t_cruise_s", val=400.0, units="s")
        o("t_mission_s", val=600.0, units="s")
        o("d_cruise_m", val=10000.0, units="m", ref=1.0e4)
        o("headwind_cruise", val=0.0, units="m/s")
        o("k_effective", val=1.0,
          desc="Battery share of the cruise drive demand actually delivered")
        o("energy_mission_wh", val=60.0, units="W*h",
          desc="Shaft energy the mission demands, all segments")
        o("energy_available_wh", val=90.0, units="W*h",
          desc="Usable battery energy above the reserve, at the terminals")
        o("SOC_final", val=0.3, desc="Battery state of charge at touchdown")
        o("P_battery_peak_w", val=1200.0, units="W",
          desc="Largest terminal power the pack must sustain")
        o("E_battery_used_wh", val=60.0, units="W*h",
          desc="Chemical energy drawn from the pack (terminal energy / discharge eff.)")
        o("E_battery_terminal_wh", val=58.0, units="W*h",
          desc="Electrical energy delivered at the pack terminals")
        o("m_fuel_carried", val=0.0, units="kg",
          desc="Fuel actually carried, after architecture gating")
        o("m_fuel_burned", val=0.0, units="kg", desc="Fuel burned on this mission")
        o("E_fuel_burned_wh", val=0.0, units="W*h",
          desc="Chemical energy of the fuel burned")
        o("g2_energy_margin", val=-0.3,
          desc="Diagnostic: E_used / E_usable - 1. Equivalent to g3, so not "
               "declared as a constraint (two copies of one constraint break LICQ)")
        o("g3_soc_margin", val=-0.2,
          desc="reserve_soc - SOC_final; <= 0 keeps the reserve at touchdown")
        o("g6_battery_power", val=0.0,
          desc="P_peak / pack continuous limit - 1; <= 0 means the pack can "
               "deliver the VTOL climb at its cell C-rate")
        o("g10_fuel_energy", val=0.0,
          desc="(fuel required incl. reserve - fuel carried) / fuel_ref; <= 0 "
               "means the tank holds what the cruise burns")

        self.declare_partials("*", "*", method="cs")

    # ── helpers ──────────────────────────────────────────────────────────

    def _hw(self, agl):
        o = self.options
        if o["wind_speed_ref"] == 0.0:
            return 0.0 * agl
        return headwind(o["wind_speed_ref"], o["wind_from_deg"], o["track_deg"], agl,
                        o["wind_z0"], o["wind_h_ref"], o["wind_log_profile"])

    def compute(self, inputs, outputs):
        o = self.options
        t = o["technology"]
        g = lambda k: inputs[k][0]

        m = g("m_tow"); W = m * G
        A = g("A_rotor"); S = g("S_ref"); AR = g("AR"); e = g("e_oswald")
        CD0 = g("C_D0"); V = g("V_cruise"); h_cr = g("altitude")
        P_lift_r = g("P_lift_rated"); P_cm_r = g("P_cm_rated")
        eta_p = o["eta_prop"]; kap = o["kappa_i"]

        def lift_bus(P):
            return rp.drive_electrical_power(P, P_lift_r, t.motor_eta_max, t.motor_k_loss, t.esc_eta)

        def cm_bus(P):
            return rp.drive_electrical_power(P, P_cm_r, t.motor_eta_max, t.motor_k_loss, t.esc_eta)

        cell = batcat.get_cell(o["cell"])
        m_batt = g("m_battery")
        n_cells = batcat.pack_cell_count(m_batt, cell, o["pack_overhead"])

        def chem(P_bus):
            """Chemical energy rate behind a terminal power."""
            return P_bus / rp.battery_discharge_efficiency(
                P_bus, n_cells, cell.voltage_nom_v, cell.r_internal_ohm)

        def prof(rho):
            return rp.profile_power(rho, A, o["sigma_rotor"], o["C_d_blade"], o["V_tip"])

        h_o, h_d = o["h_origin_m"], o["h_dest_m"]
        dh_vc, dh_vd = o["vtol_climb_m"], o["vtol_descent_m"]
        h_t1 = h_o + dh_vc                       # top of the VTOL climb
        h_t2 = h_d + dh_vd                       # start of the VTOL descent
        rho_cr = rp.isa_density_smooth(h_cr)
        C_L_cr = W / (0.5 * rho_cr * V ** 2 * S)
        k_ind = 1.0 / (np.pi * AR * e)

        # 1. VTOL climb ----------------------------------------------------
        Vc = o["vtol_climb_rate"]
        rho1 = rp.isa_density_smooth(h_o + 0.5 * dh_vc)
        P_vc = rp.axial_climb_power(W, Vc, rho1, A, kap) + prof(rho1)
        t_vc = dh_vc / Vc
        bus_vc = lift_bus(P_vc)

        # 7. VTOL descent --------------------------------------------------
        Vd = o["vtol_descent_rate"]
        rho7 = rp.isa_density_smooth(h_d + 0.5 * dh_vd)
        P_vd = rp.axial_descent_power(W, Vd, rho7, A, kap) + prof(rho7)
        t_vd = dh_vd / Vd
        bus_vd = lift_bus(P_vd)

        # 2 & 6. Transitions — Gauss-Legendre over the speed ramp ------------
        def transition(h_tr, accel, forward):
            rho = rp.isa_density_smooth(h_tr)
            dur = V / accel
            Esh = 0.0; Ebus = 0.0; Echem = 0.0; Pmax = 0.0 * W
            for x, wq in zip(_GL_X, _GL_W):
                Vj = x * V
                q = 0.5 * rho * Vj ** 2
                L = q * S * C_L_cr
                Tr = smooth_max(W - L, 0.0, scale=W)
                Pr = (rp.edgewise_rotor_power(Tr, Vj, rho, A, kap)
                      + prof(rho) * rp.rotor_running_gate(Tr, W))
                D = q * S * (CD0 + k_ind * C_L_cr ** 2)
                Tp = (m * accel + D) if forward else smooth_max(D - m * accel, 0.0, scale=W)
                Pp = Tp * Vj / eta_p
                bus = lift_bus(Pr) + cm_bus(Pp)
                Esh = Esh + wq * (Pr + Pp) * dur
                Ebus = Ebus + wq * bus * dur
                Echem = Echem + wq * chem(bus) * dur
                Pmax = smooth_max(Pmax, bus, scale=W)
            hw = self._hw(h_tr - (h_o if forward else h_d))
            dist = dur * (0.5 * V - hw)
            return Esh, Ebus, Echem, dur, dist, Pmax

        E_tf_sh, E_tf_bus, E_tf_chem, t_tf, d_tf, Pmax_tf = transition(
            h_t1, o["transition_accel"], True)
        E_tb_sh, E_tb_bus, E_tb_chem, t_tb, d_tb, Pmax_tb = transition(
            h_t2, o["transition_decel"], False)

        # 3. Wingborne climb -----------------------------------------------
        gam_c = np.radians(o["fw_climb_angle_deg"])
        dh_c = smooth_max(h_cr - h_t1, 0.0, scale=o["mass_scale_kg"] * 100.0)
        t_cl = dh_c / (V * np.sin(gam_c))
        hw_cl = self._hw(0.5 * (h_t1 + h_cr) - h_o)
        d_cl = t_cl * (V * np.cos(gam_c) - hw_cl)
        P_cl = g("P_climb")
        bus_cl = cm_bus(P_cl)

        # 5. Wingborne descent ---------------------------------------------
        gam_d = np.radians(o["fw_descent_angle_deg"])
        dh_d = smooth_max(h_cr - h_t2, 0.0, scale=o["mass_scale_kg"] * 100.0)
        t_de = dh_d / (V * np.sin(gam_d))
        hw_de = self._hw(0.5 * (h_t2 + h_cr) - h_d)
        d_de = t_de * (V * np.cos(gam_d) - hw_de)
        P_cr = g("P_cruise")
        # Thrust makes up only the drag that gravity does not; a steeper
        # descent is held by a windmilling (not regenerating) propeller.
        P_de = smooth_max(P_cr - W * V * np.sin(gam_d) / eta_p, 0.0, scale=P_cr)
        bus_de = cm_bus(P_de)

        # 4. Cruise ---------------------------------------------------------
        R = o["range_m"]
        d_cr = smooth_max(R - d_tf - d_cl - d_de - d_tb, 0.05 * R, scale=100.0)
        hw_cr = self._hw(h_cr - o["h_ground_mean_m"])
        t_cr = d_cr / (V - hw_cr)
        P_bc = g("P_batt_cruise")
        bus_cr_total = cm_bus(P_cr)

        # ── Battery energy, at the terminals and chemically ───────────────
        E_term_J = (bus_vc * t_vc + E_tf_bus + bus_cl * t_cl + P_bc * t_cr
                    + bus_de * t_de + E_tb_bus + bus_vd * t_vd)
        E_chem_J = (chem(bus_vc) * t_vc + E_tf_chem + chem(bus_cl) * t_cl
                    + chem(P_bc) * t_cr + chem(bus_de) * t_de + E_tb_chem
                    + chem(bus_vd) * t_vd)
        E_used_wh = E_chem_J / 3600.0
        E_batt_wh = batcat.pack_energy_wh(m_batt, cell, o["pack_overhead"])
        SOC_final = 1.0 - E_used_wh / E_batt_wh

        # ── Fuel ─────────────────────────────────────────────────────────
        fuel_capable = g("fuel_capable")
        m_fuel = fuel_capable * g("m_fuel")
        m_burn = g("fuel_flow_cruise") * t_cr
        m_req = m_burn / (1.0 - o["fuel_reserve"])
        LHV = g("fuel_lhv")

        # ── Pack power ───────────────────────────────────────────────────
        P_peak = smooth_max(smooth_max(bus_vc, bus_vd, scale=W),
                            smooth_max(Pmax_tf, Pmax_tb, scale=W), scale=W)
        P_limit = batcat.pack_power_limit_w(m_batt, cell, o["pack_overhead"])

        # ── Shaft energy by segment ──────────────────────────────────────
        E_vtol = (P_vc * t_vc + P_vd * t_vd) / 3600.0
        E_tr = (E_tf_sh + E_tb_sh) / 3600.0
        E_cl = P_cl * t_cl / 3600.0
        E_crs = P_cr * t_cr / 3600.0
        E_des = P_de * t_de / 3600.0

        outputs["E_battery_wh"] = E_batt_wh
        outputs["E_fuel_wh"] = m_fuel * LHV / 3600.0
        outputs["E_vtol_wh"] = E_vtol
        outputs["E_transition_wh"] = E_tr
        outputs["E_climb_wh"] = E_cl
        outputs["E_cruise_wh"] = E_crs
        outputs["E_descent_wh"] = E_des
        outputs["t_climb_s"] = t_cl
        outputs["t_cruise_s"] = t_cr
        outputs["t_mission_s"] = t_vc + t_tf + t_cl + t_cr + t_de + t_tb + t_vd
        outputs["d_cruise_m"] = d_cr
        outputs["headwind_cruise"] = hw_cr
        outputs["k_effective"] = P_bc / bus_cr_total
        outputs["energy_mission_wh"] = E_vtol + E_tr + E_cl + E_crs + E_des
        outputs["energy_available_wh"] = E_batt_wh * (1.0 - o["reserve_soc"])
        outputs["SOC_final"] = SOC_final
        outputs["P_battery_peak_w"] = P_peak
        outputs["E_battery_used_wh"] = E_used_wh
        outputs["E_battery_terminal_wh"] = E_term_J / 3600.0
        outputs["m_fuel_carried"] = m_fuel
        outputs["m_fuel_burned"] = m_burn
        outputs["E_fuel_burned_wh"] = m_burn * LHV / 3600.0
        outputs["g2_energy_margin"] = E_used_wh / (E_batt_wh * (1.0 - o["reserve_soc"])) - 1.0
        outputs["g3_soc_margin"] = o["reserve_soc"] - SOC_final
        outputs["g6_battery_power"] = P_peak / P_limit - 1.0
        # A scaled difference, not a ratio: an architecture with no fuel path
        # has required == carried == 0, and 0/0 would poison the Jacobian.
        outputs["g10_fuel_energy"] = (m_req - m_fuel) / o["fuel_ref_kg"]


class WeightsComp(om.ExplicitComponent):
    """
    Mass buildup and closure.

    `m_tow` here is an OUTPUT that feeds back to the geometry/structures
    inputs of the same group — that feedback edge is the sizing loop, and
    the group's NonlinearBlockGS solver is what converges it.
    """

    def initialize(self):
        self.options.declare("payload_kg", default=5.0, types=float)

    def setup(self):
        self.add_input("m_empty", val=8.0, units="kg",
                       desc="Airframe empty mass (from StructuresComp)")
        self.add_input("m_propulsion", val=2.5, units="kg",
                       desc="Motors, ESCs, converters and fuel system")
        self.add_input("m_battery", val=1.5, units="kg")
        self.add_input("m_fuel", val=1.2, units="kg")
        self.add_input("m_rotor_group", val=0.3, units="kg")

        # A lower bound gives BoundsEnforceLS something to enforce: MTOW
        # cannot fall below the payload.
        self.add_output("m_tow", val=20.0, units="kg", desc="Closed MTOW",
                        lower=self.options["payload_kg"] + 0.5)

        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        outputs["m_tow"] = (inputs["m_empty"][0] + inputs["m_propulsion"][0]
                            + inputs["m_battery"][0] + inputs["m_fuel"][0]
                            + inputs["m_rotor_group"][0]
                            + self.options["payload_kg"])


class ObjectiveComp(om.ExplicitComponent):
    """
    Scalarized objective, with a choice of what "cost" means.

    ``primary_energy`` (default) — energy drawn from OUTSIDE the aircraft to
    fly the corridor mission: the chemical energy taken from the pack
    divided by the charging efficiency, plus the chemical energy of the fuel
    BURNED. Every architecture is charged for its own losses at the same
    boundary. (The fuel term used to be the fuel CARRIED, which charged a
    tank sized for the design-range mission against a corridor that burns a
    fraction of it.)

    ``energy_mass`` — battery + fuel + powerplant mass.

    ``shaft_energy`` — the shaft energy the mission demands; for
    single-architecture studies only, since it hides conversion losses.

    The discreteness penalty is MULTIPLICATIVE: cost * (1 + mu * sum
    w_i(1-w_i)). An absolute penalty (the old lambda = 200 Wh) is 3.5x the
    objective at 13 km and 4 % of it at 600 km, so its strength depended on
    the mission rather than on the relaxation. A relative one does not.
    """

    METRICS = ("primary_energy", "energy_mass", "shaft_energy")

    def initialize(self):
        self.options.declare("penalty_scale", default=1.0, types=float,
                             desc="mu, the relative weight of sum w_i(1-w_i)")
        self.options.declare("metric", default="primary_energy",
                             values=("primary_energy", "energy_mass", "shaft_energy"),
                             desc="Which cost the optimizer minimises")
        self.options.declare("eta_charging", default=0.90, types=float,
                             desc="Grid-to-battery charging efficiency")

    def setup(self):
        self.add_input("energy_mission_wh", val=60.0, units="W*h",
                       desc="Shaft energy the mission demands")
        self.add_input("E_battery_used_wh", val=60.0, units="W*h",
                       desc="Chemical energy drawn from the pack")
        self.add_input("m_fuel_burned", val=0.0, units="kg")
        self.add_input("m_fuel_carried", val=0.0, units="kg")
        self.add_input("fuel_lhv", val=43.0e6, units="J/kg")
        self.add_input("m_battery", val=1.0, units="kg")
        self.add_input("m_propulsion", val=1.0, units="kg")
        self.add_input("penalty_discreteness", val=0.8)

        self.add_output("energy_primary_wh", val=70.0, units="W*h",
                        desc="Energy drawn from outside the aircraft")
        self.add_output("energy_system_mass", val=2.0, units="kg",
                        desc="Battery + fuel + powerplant mass")
        self.add_output("objective", val=70.0)
        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        opt = self.options
        e_batt = inputs["E_battery_used_wh"][0] / opt["eta_charging"]
        e_fuel = inputs["m_fuel_burned"][0] * inputs["fuel_lhv"][0] / 3600.0
        energy_primary = e_batt + e_fuel
        energy_system_mass = (inputs["m_battery"][0] + inputs["m_fuel_carried"][0]
                              + inputs["m_propulsion"][0])

        outputs["energy_primary_wh"] = energy_primary
        outputs["energy_system_mass"] = energy_system_mass

        metric = opt["metric"]
        if metric == "primary_energy":
            cost = energy_primary
        elif metric == "energy_mass":
            cost = energy_system_mass
        else:
            cost = inputs["energy_mission_wh"][0]
        outputs["objective"] = cost * (1.0 + opt["penalty_scale"]
                                       * inputs["penalty_discreteness"][0])


class TerrainClearanceComp(om.ExplicitComponent):
    """
    Terrain clearance as an explicit optimization constraint.

    Previously the sizing problem never saw the terrain: cruise altitude was
    a hardcoded 2900 m default, and nothing checked it against the ridge the
    route actually crosses. Sizing therefore used the wrong air density and
    could return a design whose cruise altitude flew through a mountain.

    With cruise altitude promoted to a design variable, this component turns
    the terrain profile from m1 into the constraint that bounds it from
    below, while climb energy (EnergyComp) bounds it from above. The
    optimizer then trades altitude against energy with the ridge as a hard
    floor, which is the physically meaningful formulation.
    """

    def initialize(self):
        self.options.declare("h_terrain_max_m", default=3026.0, types=float,
                             desc="Highest terrain on the direct route, from the DEM")
        self.options.declare("clearance_cruise_m", default=100.0, types=float,
                             desc="Required AGL during cruise")

    def setup(self):
        self.add_input("altitude", val=2900.0, units="m",
                       desc="Cruise altitude AMSL (design variable)")

        self.add_output("agl_cruise", val=100.0, units="m",
                        desc="Height above the route's highest terrain")
        self.add_output("g5_terrain_clearance", val=0.0,
                        desc="(h_terrain_max + clearance_required - h_cruise) "
                             "normalised; <= 0 means the cruise altitude "
                             "clears the ridge with the required margin")

        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        opt = self.options
        h_cruise = inputs["altitude"][0]
        h_required = opt["h_terrain_max_m"] + opt["clearance_cruise_m"]

        outputs["agl_cruise"] = h_cruise - opt["h_terrain_max_m"]
        # Normalised by the required clearance so the constraint is O(1) and
        # scales sensibly for the driver.
        outputs["g5_terrain_clearance"] = (h_required - h_cruise) / opt["clearance_cruise_m"]


class RotorFitComp(om.ExplicitComponent):
    """
    Geometric admissibility: the rotors have to fit on the airframe.

    Uses the quadplane H layout every consumer shares
    (``m2_geometry.rotor.boom_layout``): booms at the innermost station
    where the discs clear the fuselage, y_b = (D(1+c) + d_fus)/2, and the
    outboard rotor tip inside the wing tip, y_b + D/2 <= b/2. The previous
    constraint assumed two rotors side by side on each semi-span (2.2 D <=
    b), a layout the 3D model did not draw and no quadplane uses.

    Modelling it as a constraint rather than a bound means the limit moves
    with the wing instead of being a constant the optimizer leans on.
    """

    def initialize(self):
        self.options.declare("clearance_frac", default=0.10, types=float,
                             desc="Tip-to-tip gap as a fraction of diameter")

    def setup(self):
        self.add_input("rotor_diameter", val=0.45, units="m")
        self.add_input("span", val=2.5, units="m")
        self.add_input("fuse_diameter", val=0.16, units="m")

        self.add_output("rotor_span_required", val=2.0, units="m",
                        desc="Span the rotor array needs, tip to tip")
        self.add_output("g7_rotor_fit", val=0.0,
                        desc="rotor_span_required/span - 1; <= 0 means the "
                             "rotors fit within the wingspan")

        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        D = inputs["rotor_diameter"][0]
        y_b = 0.5 * (D * (1.0 + self.options["clearance_frac"]) + inputs["fuse_diameter"][0])
        required = 2.0 * (y_b + 0.5 * D)
        outputs["rotor_span_required"] = required
        outputs["g7_rotor_fit"] = required / inputs["span"][0] - 1.0


class AdmissibilityComp(om.ExplicitComponent):
    """
    Limits that come from the models being valid, not from taste.

    Chord Reynolds number: below ~2e5 a small-UAV wing runs laminar
    separation bubbles that a turbulent flat-plate skin-friction buildup
    does not model, so C_D0 becomes optimistic exactly where the optimizer
    wants to go. Constraining Re keeps the answer inside the region where
    the drag polar being optimised against is credible — it is what caps
    aspect ratio.

    Kinematic viscosity is now the ISA/Sutherland value the drag buildup
    itself uses. It used to be nu_sl * rho_sl / rho (viscosity held at its
    sea-level value), which overstates nu by ~6 % at 3100 m — on an active
    constraint, that is a 6 % error in the chord the design is allowed.
    """

    def initialize(self):
        self.options.declare("Re_min", default=2.0e5, types=float,
                             desc="Lowest chord Reynolds number the drag "
                                  "polar is trusted at")

    def setup(self):
        self.add_input("chord_mean", val=0.25, units="m")
        self.add_input("V_cruise", val=30.0, units="m/s")
        self.add_input("altitude", val=3000.0, units="m")

        self.add_output("Re_cruise", val=5.0e5, ref=1.0e5,
                        desc="Chord Reynolds number at the cruise condition")
        self.add_output("g8_reynolds", val=0.0,
                        desc="1 - Re_cruise/Re_min; <= 0 keeps the design "
                             "inside the drag polar's validity range")

        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        from hpraptor.core.atmosphere import isa_density, isa_viscosity

        alt = inputs["altitude"][0]
        nu = isa_viscosity(alt) / isa_density(alt)
        Re = inputs["V_cruise"][0] * inputs["chord_mean"][0] / nu

        outputs["Re_cruise"] = Re
        outputs["g8_reynolds"] = 1.0 - Re / self.options["Re_min"]


class RotorGroupMassComp(om.ExplicitComponent):
    """
    Mass of the lift-rotor group, so a bigger disk is not free.

    Disk loading sat on its lower bound in every run: hover power falls as
    the disk grows, and the mass model only counted motors (by specific
    power), never the blades, hubs or the booms that carry them. Growing the
    rotors was therefore pure gain, which is not a property of any real
    aircraft.

    Blades, hubs and booms are estimated together from total disk area. The
    coefficient is a statistical figure for small multirotor hardware, not a
    structural calculation — enough to put a real price on disk area and let
    the optimizer find an interior trade, which is the point.
    """

    def initialize(self):
        self.options.declare("rotor_mass_per_area", default=0.45, types=float,
                             desc="Blade + hub + boom mass per m^2 of disk [kg/m^2]")

    def setup(self):
        self.add_input("A_rotor", val=0.65, units="m**2")
        self.add_output("m_rotor_group", val=0.3, units="kg",
                        desc="Blades, hubs and booms — excludes the motors, "
                             "which ArchitectureComp sizes by specific power")
        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        outputs["m_rotor_group"] = (self.options["rotor_mass_per_area"]
                                    * inputs["A_rotor"][0])
