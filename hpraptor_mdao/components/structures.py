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


class StructuresComp(om.ExplicitComponent):
    """Wing spar sizing and structural mass for a given thickness."""

    def initialize(self):
        self.options.declare("t_c", default=0.12, types=float)
        self.options.declare("load_factor_ultimate", default=3.8, types=float)
        self.options.declare("safety_factor", default=1.5, types=float)
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
        load_case = WingLoadCase(
            mtow_kg=inputs["m_tow"][0],
            load_factor_ultimate=self.options["load_factor_ultimate"],
            safety_factor=self.options["safety_factor"],
        )
        sizer = WingStructuralSizer(wing, load_case)
        spar = sizer.size_spar(inputs["t_spar_mm"][0])
        wing_mass = compute_wing_mass_buildup(wing, spar.spar_mass_kg)

        outputs["spar_mass"] = spar.spar_mass_kg
        outputs["m_wing_structure"] = wing_mass.total_kg
        outputs["g4_stress_margin"] = spar.stress_margin
        outputs["m_empty"] = (self.options["f_empty_nonwing"] * inputs["m_tow"][0]
                              + wing_mass.total_kg)
