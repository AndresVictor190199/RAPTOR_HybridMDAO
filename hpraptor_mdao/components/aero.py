"""
m4_aero as an OpenMDAO discipline.

Parasite drag from real wetted areas (component buildup), plus the
cruise drag polar. This is the "Geometry -> Parasite Drag -> Power
feedback" edge of the mission sizing loop, and in the N2 diagram it is
the link that makes wing area actually affect required power.
"""

from __future__ import annotations
import numpy as np
import openmdao.api as om

from hpraptor.core.atmosphere import isa_density
from hpraptor.m4_aero.parasite_drag import parasite_cd0_from_wetted

G = 9.80665


class AeroComp(om.ExplicitComponent):
    """Parasite drag buildup and the cruise drag polar."""

    def initialize(self):
        self.options.declare("t_c", default=0.12, types=float)
        # Kept only as an override. e is now an INPUT, computed from AR
        # and taper by GeometryComp -- a constant e made the drag polar
        # blind to planform shape, which would have made a taper design
        # variable a null direction. Set this to a float to pin it.
        self.options.declare("e_oswald", default=None,
                             types=float, allow_none=True)
        self.options.declare("C_L_max", default=1.6, types=float)

    def setup(self):
        self.add_input("S_ref", val=0.7, units="m**2")
        self.add_input("AR", val=10.0)
        self.add_input("fuse_length", val=1.2, units="m")
        self.add_input("fuse_diameter", val=0.2, units="m")
        # Wetted areas are CONSUMED, not re-derived. This component used to
        # rebuild the geometry from length and diameter and compute its own
        # wetted areas, which meant swapping GeometryComp for the AeroSandbox
        # assembly changed nothing at all downstream: the measured areas were
        # produced and then ignored. Taking them as inputs is what actually
        # connects the geometry solver to the drag polar.
        self.add_input("wetted_wing", val=1.5, units="m**2")
        self.add_input("wetted_fuse", val=0.75, units="m**2")
        self.add_input("chord_mean", val=0.27, units="m")
        self.add_input("m_tow", val=20.0, units="kg")
        self.add_input("e_oswald", val=0.78,
                       desc="Span efficiency from AR and taper (m2 geometry)")
        self.add_input("V_cruise", val=30.0, units="m/s")
        self.add_input("altitude", val=2900.0, units="m")

        self.add_output("C_D0", val=0.02, desc="Zero-lift drag coefficient")
        self.add_output("C_L_cruise", val=0.6)
        self.add_output("C_D_cruise", val=0.04)
        self.add_output("L_D", val=17.0, desc="Cruise lift-to-drag ratio")
        self.add_output("D_cruise", val=12.0, units="N")
        self.add_output("g1_cl_margin", val=-0.5,
                        desc="C_L_cruise/(0.8*C_L_max) - 1; <= 0 keeps stall margin")

        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        S = inputs["S_ref"][0]
        AR = inputs["AR"][0]
        V = inputs["V_cruise"][0]
        alt = inputs["altitude"][0]
        W = inputs["m_tow"][0] * G
        e = (self.options["e_oswald"] if self.options["e_oswald"] is not None
             else inputs["e_oswald"][0])

        C_D0 = parasite_cd0_from_wetted(
            S_ref=S,
            chord_mean=inputs["chord_mean"][0],
            wetted_wing=inputs["wetted_wing"][0],
            fuse_length=inputs["fuse_length"][0],
            wetted_fuse=inputs["wetted_fuse"][0],
            airspeed_ms=V, altitude_m=alt,
        )

        rho = isa_density(alt)
        q = 0.5 * rho * V ** 2
        C_L = W / (q * S)
        C_D = C_D0 + C_L ** 2 / (np.pi * AR * e)

        outputs["C_D0"] = C_D0
        outputs["C_L_cruise"] = C_L
        outputs["C_D_cruise"] = C_D
        outputs["L_D"] = C_L / C_D
        outputs["D_cruise"] = q * S * C_D
        # Stall margin: keep cruise C_L below 80% of C_L_max.
        outputs["g1_cl_margin"] = C_L / (0.8 * self.options["C_L_max"]) - 1.0
