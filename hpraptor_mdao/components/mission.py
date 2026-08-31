"""
Mission energy and storage sizing, plus the mass-closure component.

EnergyComp sizes the battery (VTOL energy + peak power) and the fuel
(Breguet), and reports the mission energy that the optimizer minimizes.
WeightsComp closes the mass loop — its `m_tow` output feeds back to
m2/m3, and that feedback edge is what the group's nonlinear solver
converges.
"""

from __future__ import annotations
import numpy as np
import openmdao.api as om

from hpraptor.m5_propulsion.architecture_np import smooth_max
from hpraptor.m5_propulsion import battery_catalogue as batcat

G = 9.80665


class EnergyComp(om.ExplicitComponent):
    """Battery and fuel sizing, and total mission energy."""

    def initialize(self):
        self.options.declare("cell", default=batcat.DEFAULT_CELL, types=str,
                             desc="Catalogue key for the battery cell format")
        self.options.declare("pack_overhead", default=batcat.DEFAULT_PACK_OVERHEAD,
                             types=float)
        self.options.declare("eta_motor", default=0.93, types=float)
        self.options.declare("eta_overall_cruise", default=0.30, types=float)
        self.options.declare("reserve_soc", default=0.15, types=float)
        self.options.declare("fuel_reserve", default=0.10, types=float)
        self.options.declare("t_vtol_total", default=95.0, types=float, desc="s")
        self.options.declare("range_m", default=50000.0, types=float)
        self.options.declare("h_origin_m", default=2900.0, types=float,
                             desc="Departure pad elevation AMSL, from the DEM")
        self.options.declare("eta_prop", default=0.75, types=float)
        self.options.declare("fw_climb_angle_deg", default=8.0, types=float)
        # Characteristic magnitudes for the smooth max/min blends. These MUST
        # be constants, not derived from the current iterate: a data-dependent
        # smoothing scale makes eps a function of the inputs, so complex step
        # (which sees the scale frozen at its real part) and finite difference
        # (which recomputes it) disagree on the derivative.
        self.options.declare("mass_scale_kg", default=1.0, types=float)

    def setup(self):
        self.add_input("P_hover", val=3000.0, units="W")
        self.add_input("P_cruise", val=450.0, units="W")
        self.add_input("P_climb", val=1500.0, units="W")
        self.add_input("L_D", val=17.0)
        self.add_input("m_tow", val=20.0, units="kg")
        self.add_input("fuel_lhv", val=43.0e6, units="J/kg")
        self.add_input("V_cruise", val=30.0, units="m/s")
        self.add_input("altitude", val=2900.0, units="m",
                       desc="Cruise altitude AMSL (design variable)")
        self.add_input("k_electric", val=0.5,
                       desc="Commanded share of cruise power drawn from the battery")
        self.add_input("fuel_capable", val=1.0,
                       desc="Blended ability to burn fuel; 0 for all-electric")
        # Storage masses are DESIGN VARIABLES, not sizing formulas. Deriving
        # them from the requirement made g3 satisfied by construction; as
        # variables the optimizer must earn the reserve, and g3/g6 become
        # constraints that genuinely bind.
        self.add_input("m_battery", val=1.0, units="kg",
                       desc="Battery pack mass (design variable)")
        self.add_input("m_fuel", val=0.3, units="kg",
                       desc="Usable fuel mass (design variable)")

        self.add_output("E_battery_wh", val=250.0, units="W*h")
        self.add_output("E_fuel_wh", val=14000.0, units="W*h")
        self.add_output("E_vtol_wh", val=80.0, units="W*h")
        self.add_output("E_climb_wh", val=60.0, units="W*h")
        self.add_output("E_cruise_wh", val=900.0, units="W*h")
        self.add_output("t_climb_s", val=60.0, units="s")
        self.add_output("k_effective", val=0.5,
                        desc="Battery share actually used after architecture "
                             "gating; forced to 1 where no fuel path exists")
        self.add_output("energy_mission_wh", val=1000.0, units="W*h",
                        desc="Shaft energy required by the design mission")
        self.add_output("energy_available_wh", val=1000.0, units="W*h",
                        desc="Usable shaft energy from the sized battery + fuel")
        self.add_output("SOC_final", val=0.2,
                        desc="Battery state of charge at mission end")
        self.add_output("g2_energy_margin", val=0.0,
                        desc="energy_mission/energy_available - 1; <= 0 means "
                             "the vehicle can actually fly the mission")
        self.add_output("g3_soc_margin", val=0.0,
                        desc="reserve_soc - SOC_final; <= 0 means the battery "
                             "still holds its reserve at touchdown")
        self.add_output("g6_battery_power", val=0.0,
                        desc="P_peak_elec/pack power limit - 1; <= 0 means the "
                             "pack can actually deliver the hover peak at its "
                             "cell C-rate")
        self.add_output("P_battery_peak_w", val=3000.0, units="W",
                        desc="Peak electrical draw the pack must sustain")
        self.add_output("m_fuel_carried", val=0.3, units="kg",
                        desc="Fuel actually carried, after architecture gating")
        self.add_output("E_battery_used_wh", val=200.0, units="W*h",
                        desc="Electrical energy drawn from the pack over the mission")

        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        opt = self.options
        P_hover = inputs["P_hover"][0]
        P_cruise = inputs["P_cruise"][0]
        P_climb = inputs["P_climb"][0]
        L_D = inputs["L_D"][0]
        m_tow = inputs["m_tow"][0]
        LHV = inputs["fuel_lhv"][0]
        V = inputs["V_cruise"][0]
        h_cruise = inputs["altitude"][0]
        fuel_capable = inputs["fuel_capable"][0]

        k_e = inputs["k_electric"][0]
        R = opt["range_m"]

        # ── Architecture gating ──────────────────────────────────────────
        # An architecture with no fuel converter cannot serve ANY of the
        # cruise from fuel, whatever k_electric says. Without this the
        # optimizer happily sized an all-electric aircraft that drew half
        # its energy from gasoline it had no way to burn.
        k_eff = 1.0 - fuel_capable * (1.0 - k_e)

        # ── Shaft energy the design mission demands ──────────────────────
        E_vtol_wh = P_hover * opt["t_vtol_total"] / 3600.0

        # Climb from the departure pad to cruise altitude. This is what
        # gives cruise altitude a cost, so the terrain-clearance constraint
        # (g5) binds from below instead of the altitude floating free.
        dh_climb = smooth_max(h_cruise - opt["h_origin_m"], 0.0,
                              scale=opt["mass_scale_kg"] * 100.0)
        roc = V * np.sin(np.radians(opt["fw_climb_angle_deg"]))
        t_climb = dh_climb / roc
        E_climb_wh = P_climb * t_climb / 3600.0

        # Cruise covers the range less the ground distance used climbing.
        d_climb = t_climb * V * np.cos(np.radians(opt["fw_climb_angle_deg"]))
        d_cruise = smooth_max(R - d_climb, 0.05 * R, scale=100.0)
        E_cruise_wh = P_cruise * (d_cruise / V) / 3600.0

        energy_mission_wh = E_vtol_wh + E_climb_wh + E_cruise_wh

        # ── Battery: all of VTOL and climb (powered lift and climb are
        #    electric on every architecture here), plus k_eff of cruise.
        E_batt_required_wh = E_vtol_wh + E_climb_wh + k_eff * E_cruise_wh

        # Installed energy now follows from the MASS the optimizer chose and
        # the cell format's pack-level specific energy, rather than being
        # back-solved from the requirement. That inversion is what turns g3
        # from an identity into a constraint.
        cell = batcat.get_cell(opt["cell"])
        m_battery = inputs["m_battery"][0]
        m_fuel_dv = inputs["m_fuel"][0]
        E_battery_wh = batcat.pack_energy_wh(m_battery, cell, opt["pack_overhead"])

        # ── Fuel: only the share the architecture can actually burn.
        #    Carried fuel is gated so a battery-only aircraft has no dead
        #    fuel mass even if the optimizer leaves m_fuel non-zero.
        eta = opt["eta_overall_cruise"]
        m_fuel = fuel_capable * m_fuel_dv

        # ── Usable shaft energy actually on board ────────────────────────
        E_fuel_wh = m_fuel * LHV / 3600.0
        usable_batt = E_battery_wh * (1.0 - opt["reserve_soc"])
        usable_fuel = E_fuel_wh * eta * (1.0 - opt["fuel_reserve"])
        energy_available_wh = usable_batt + usable_fuel

        # ── State of charge at touchdown ─────────────────────────────────
        SOC_final = 1.0 - E_batt_required_wh / E_battery_wh

        # ── Can the pack physically deliver the hover peak? ───────────────
        # A pack sized on energy alone can be incapable of the hover draw at
        # its cells' C-rate. Energy-dense cylindricals in particular need
        # several times the energy-sized mass just to sustain power.
        P_peak_elec = P_hover / opt["eta_motor"]
        P_pack_limit = batcat.pack_power_limit_w(m_battery, cell, opt["pack_overhead"])

        outputs["P_battery_peak_w"] = P_peak_elec
        outputs["g6_battery_power"] = P_peak_elec / P_pack_limit - 1.0
        outputs["m_fuel_carried"] = m_fuel
        outputs["E_battery_used_wh"] = E_batt_required_wh
        outputs["E_battery_wh"] = E_battery_wh
        outputs["E_fuel_wh"] = E_fuel_wh
        outputs["E_vtol_wh"] = E_vtol_wh
        outputs["E_climb_wh"] = E_climb_wh
        outputs["E_cruise_wh"] = E_cruise_wh
        outputs["t_climb_s"] = t_climb
        outputs["k_effective"] = k_eff
        outputs["energy_mission_wh"] = energy_mission_wh
        outputs["energy_available_wh"] = energy_available_wh
        outputs["SOC_final"] = SOC_final
        outputs["g2_energy_margin"] = energy_mission_wh / energy_available_wh - 1.0
        outputs["g3_soc_margin"] = opt["reserve_soc"] - SOC_final


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
        self.add_input("m_propulsion", val=2.5, units="kg")
        self.add_input("m_battery", val=1.5, units="kg")
        self.add_input("m_fuel", val=1.2, units="kg")
        self.add_input("m_rotor_group", val=0.3, units="kg")

        # A lower bound is what gives BoundsEnforceLS something to enforce.
        # Without it a Newton step can drive MTOW negative, and the geometry
        # model then takes the square root of a negative area and fills the
        # whole Jacobian with NaN — which is exactly how the coupled solve
        # failed before this bound existed. MTOW cannot be below the payload.
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

    Comparing architectures on SHAFT energy is not neutral. A hybrid pays
    its conversion losses inside the number (fuel -> shaft at ~30%), while a
    battery aircraft's stored energy is already electrical, so it does not.
    Ranking on shaft Wh therefore flatters the battery path for reasons that
    have nothing to do with the design.

    Three metrics are offered, and the default is the fair one:

    ``primary_energy``  (default)
        Energy drawn from OUTSIDE the aircraft to fly the mission: battery
        energy divided by charging efficiency, plus the fuel's full chemical
        energy. Every architecture is charged for its own losses at the same
        boundary, which is what makes the comparison like-for-like.

    ``energy_mass``
        Mass of the energy system (battery + fuel + powerplant). For an
        aircraft this is often the decision that matters more than energy,
        and it is completely architecture-neutral.

    ``shaft_energy``
        The previous behaviour, kept so earlier results stay reproducible.
        Use it for single-architecture studies, not for ranking.
    """

    METRICS = ("primary_energy", "energy_mass", "shaft_energy")

    def initialize(self):
        self.options.declare("penalty_scale", default=200.0, types=float,
                             desc="lambda, in objective units per unit of sum w_i(1-w_i)")
        self.options.declare("metric", default="primary_energy", values=("primary_energy", "energy_mass", "shaft_energy"),
                             desc="Which cost the optimizer minimises")
        self.options.declare("eta_charging", default=0.90, types=float,
                             desc="Grid-to-battery charging efficiency")

    def setup(self):
        self.add_input("energy_mission_wh", val=1000.0, units="W*h",
                       desc="Shaft energy the mission demands")
        self.add_input("E_battery_used_wh", val=200.0, units="W*h",
                       desc="Electrical energy drawn from the pack")
        self.add_input("m_fuel_carried", val=0.3, units="kg")
        self.add_input("fuel_lhv", val=43.0e6, units="J/kg")
        self.add_input("m_battery", val=1.0, units="kg")
        self.add_input("m_propulsion", val=2.5, units="kg")
        self.add_input("penalty_discreteness", val=0.8)

        self.add_output("energy_primary_wh", val=1000.0, units="W*h",
                        desc="Energy drawn from outside the aircraft")
        self.add_output("energy_system_mass", val=3.0, units="kg",
                        desc="Battery + fuel + powerplant mass")
        self.add_output("objective", val=1000.0)
        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        opt = self.options

        # Primary energy: charge the battery path for charging losses, and
        # the fuel path for its full chemical content. Same boundary for
        # both, which is the whole point.
        e_batt_primary = inputs["E_battery_used_wh"][0] / opt["eta_charging"]
        e_fuel_primary = inputs["m_fuel_carried"][0] * inputs["fuel_lhv"][0] / 3600.0
        energy_primary = e_batt_primary + e_fuel_primary

        energy_system_mass = (inputs["m_battery"][0]
                              + inputs["m_fuel_carried"][0]
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

        outputs["objective"] = cost + opt["penalty_scale"] * inputs["penalty_discreteness"][0]


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

    Disk loading was pinned at its LOWER bound in every run, because hover
    power falls as the disk grows and nothing in the model stopped the disk
    from growing. The real limit is not a number typed into the bounds, it
    is that four rotors of diameter D must physically fit around a wing of
    a given span without overlapping each other or the fuselage.

    Modelling it as a constraint rather than a bound means the limit moves
    correctly when the wing does, instead of being a constant the optimizer
    silently leans on.
    """

    def initialize(self):
        self.options.declare("n_rotors", default=4, types=int)
        self.options.declare("clearance_frac", default=0.10, types=float,
                             desc="Tip-to-tip gap as a fraction of diameter")

    def setup(self):
        self.add_input("rotor_diameter", val=0.45, units="m")
        self.add_input("span", val=2.5, units="m")
        self.add_input("fuse_length", val=1.2, units="m")

        self.add_output("rotor_span_required", val=2.0, units="m",
                        desc="Lateral extent the rotor array needs")
        self.add_output("g7_rotor_fit", val=0.0,
                        desc="rotor_span_required/span - 1; <= 0 means the "
                             "rotors fit within the wingspan")

        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        D = inputs["rotor_diameter"][0]
        span = inputs["span"][0]
        pitch = D * (1.0 + self.options["clearance_frac"])

        # A quad carries two rotors per side; they must sit within the
        # semi-span without their tips meeting at the centreline.
        per_side = self.options["n_rotors"] / 2.0
        required = per_side * pitch

        outputs["rotor_span_required"] = required
        outputs["g7_rotor_fit"] = required / span - 1.0


class AdmissibilityComp(om.ExplicitComponent):
    """
    Limits that come from the models being valid, not from taste.

    Two bounds kept binding at the optimum because the physics that should
    have stopped them was missing:

    **Aspect ratio** ran to whatever ceiling it was given. Spar stress did
    not stop it (the spar sits on its manufacturing thickness floor), so
    nothing did. The real limit is chord Reynolds number: at AR 22 on a
    1.9 m span the chord is 86 mm, and at that scale the drag polar this
    model uses — built from turbulent flat-plate skin friction — no longer
    describes the flow. Constraining Re keeps the design inside the region
    where the aerodynamics it is being optimised against are meaningful.

    **Disk loading** ran to its floor because a bigger rotor always lowered
    hover power and nothing charged for it. Rotor mass is handled in
    RotorGroupMassComp; this component covers the Reynolds side.
    """

    def initialize(self):
        # 2e5, not 1e5. Below roughly this figure a small-UAV wing runs
        # laminar separation bubbles that a turbulent flat-plate skin-friction
        # buildup does not model, so C_D0 becomes optimistic exactly where the
        # optimizer wants to go. Constraining Re keeps the answer inside the
        # region where the drag polar being optimised against is credible.
        self.options.declare("Re_min", default=2.0e5, types=float,
                             desc="Lowest chord Reynolds number the drag "
                                  "polar is trusted at")
        self.options.declare("nu_sea_level", default=1.46e-5, types=float,
                             desc="Kinematic viscosity at sea level [m^2/s]")

    def setup(self):
        self.add_input("chord_mean", val=0.25, units="m")
        self.add_input("V_cruise", val=30.0, units="m/s")
        self.add_input("altitude", val=3000.0, units="m")

        self.add_output("Re_cruise", val=5.0e5,
                        desc="Chord Reynolds number at the cruise condition")
        self.add_output("g8_reynolds", val=0.0,
                        desc="1 - Re_cruise/Re_min; <= 0 keeps the design "
                             "inside the drag polar's validity range")

        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        from hpraptor.core.atmosphere import isa_density

        rho = isa_density(inputs["altitude"][0])
        # Kinematic viscosity scales roughly with 1/rho at constant
        # temperature; adequate for a validity check, and smooth.
        nu = self.options["nu_sea_level"] * 1.225 / rho
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
