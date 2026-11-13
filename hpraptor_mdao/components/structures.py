"""
m3_structures as an OpenMDAO discipline.

Note the formulation change relative to hpraptor.core.initial_sizing:
that pipeline calls `minimum_feasible_t_spar_mm()`, which runs a BISECTION
SEARCH inside the analysis to find the thinnest feasible spar. A root-find
nested inside a component is exactly what you avoid in a gradient-based
MDO framework — it is non-smooth and its derivative is meaningless.

The OpenMDAO-native equivalent is to expose `t_spar_mm` as a DESIGN
VARIABLE and the stress margin as an INEQUALITY CONSTRAINT (g4 <= 0), and
let the outer optimizer drive the thickness down until the constraint
binds. Same answer, real gradients, and the constraint activity is visible.
"""

from __future__ import annotations
import numpy as np
import openmdao.api as om

from hpraptor.m2_geometry.planform import WingPlanform
from hpraptor.m3_structures.spar_sizing import WingLoadCase, WingStructuralSizer
from hpraptor.m3_structures.mass_buildup import compute_wing_mass_buildup
from hpraptor.m5_propulsion.rotor_power import isa_density_smooth
from hpraptor.m5_propulsion.smooth import smooth_max

G = 9.80665
RHO_SL = 1.225


def gust_load_factor(wing_loading_pa, rho, V_tas, mac, cl_alpha,
                     U_de: float = 15.24):
    """
    Limit gust load factor, CS-23.341 / FAR-23.341 (Pratt-Walker).

        mu_g = 2 (W/S) / (rho c a g)
        K_g  = 0.88 mu_g / (5.3 + mu_g)
        n    = 1 + K_g rho_0 U_de V_EAS a / (2 W/S)

    U_de = 15.24 m/s (50 ft/s) is the design gust at V_C. A light,
    low-wing-loading aircraft is gust-critical rather than manoeuvre-critical:
    at 30 m/s and ~330 N/m^2 this gives ~4.3 g against a 3.8 g manoeuvre
    envelope, so the manoeuvre value alone under-sized the load case. It is
    also the coupling that lets wing loading reach the structure through the
    loads and not only through the geometry.
    """
    mu_g = 2.0 * wing_loading_pa / (rho * mac * cl_alpha * G)
    K_g = 0.88 * mu_g / (5.3 + mu_g)
    V_eas = V_tas * np.sqrt(rho / RHO_SL)
    return 1.0 + K_g * RHO_SL * U_de * V_eas * cl_alpha / (2.0 * wing_loading_pa)


class StructuresComp(om.ExplicitComponent):
    """Wing spar sizing and structural mass for a given thickness."""

    def initialize(self):
        self.options.declare("t_c", default=0.12, types=float)
        # 3.8 is the CS-23 normal-category LIMIT manoeuvre load factor; the
        # 1.5 factor of safety is applied on the allowable stress, so the
        # spar is checked at an ultimate 5.7 g. (The option name predates
        # this note and is kept for compatibility.)
        self.options.declare("load_factor_ultimate", default=3.8, types=float,
                             desc="LIMIT manoeuvre load factor (CS-23 normal)")
        self.options.declare("safety_factor", default=1.5, types=float)
        self.options.declare("gust_velocity", default=15.24, types=float,
                             desc="Design gust U_de at V_C [m/s]")
        self.options.declare("f_empty_nonwing", default=0.25, types=float,
                             desc="Non-wing empty fraction (fuselage/tail/gear/avionics/"
                                  "wiring). The WING's structural mass is computed here "
                                  "from real geometry and loads and added on top, so this "
                                  "fraction must NOT include it.")

    def setup(self):
        self.add_input("S_ref", val=0.7, units="m**2")
        self.add_input("AR", val=10.0)
        self.add_input("m_tow", val=20.0, units="kg")
        self.add_input("t_spar_mm", val=2.0, units="mm", desc="Spar web thickness (design var)")
        self.add_input("taper_ratio", val=1.0, desc="c_tip / c_root")
        self.add_input("V_cruise", val=30.0, units="m/s")
        self.add_input("altitude", val=2900.0, units="m")

        self.add_output("n_gust", val=4.0, desc="CS-23 Pratt gust load factor")
        self.add_output("n_limit", val=4.0,
                        desc="Governing limit load factor, max(manoeuvre, gust)")
        self.add_output("m_wing_structure", val=3.0, units="kg",
                        desc="Spar + ribs + skin")
        self.add_output("spar_mass", val=1.0, units="kg")
        self.add_output("g4_stress_margin", val=-0.5,
                        desc="sigma_max/sigma_allow - 1; <= 0 is feasible")
        # Airframe empty mass is emitted here (rather than in WeightsComp)
        # purely so the mass-closure cycle stays a clean loop through the
        # disciplines: WeightsComp would otherwise need m_tow both as an
        # input and as its own output, which is a degenerate self-cycle.
        self.add_output("m_empty", val=8.0, units="kg",
                        desc="f_empty_nonwing * m_tow + wing structure")

        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        wing = WingPlanform(S=inputs["S_ref"][0], AR=inputs["AR"][0],
                            t_c=self.options["t_c"],
                            taper_ratio=inputs["taper_ratio"][0])
        m_tow = inputs["m_tow"][0]
        rho = isa_density_smooth(inputs["altitude"][0])
        n_gust = gust_load_factor(m_tow * G / wing.S, rho, inputs["V_cruise"][0],
                                  wing.mac, wing.lift_curve_slope,
                                  U_de=self.options["gust_velocity"])
        n_limit = smooth_max(self.options["load_factor_ultimate"], n_gust, scale=1.0)
        load_case = WingLoadCase(
            mtow_kg=m_tow,
            load_factor_ultimate=n_limit,
            safety_factor=self.options["safety_factor"],
        )
        sizer = WingStructuralSizer(wing, load_case)
        spar = sizer.size_spar(inputs["t_spar_mm"][0])
        wing_mass = compute_wing_mass_buildup(wing, spar.spar_mass_kg)

        outputs["n_gust"] = n_gust
        outputs["n_limit"] = n_limit
        outputs["spar_mass"] = spar.spar_mass_kg
        outputs["m_wing_structure"] = wing_mass.total_kg
        outputs["g4_stress_margin"] = spar.stress_margin
        outputs["m_empty"] = (self.options["f_empty_nonwing"] * inputs["m_tow"][0]
                              + wing_mass.total_kg)
