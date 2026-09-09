"""
m2_geometry as an OpenMDAO discipline.

Wraps hpraptor.m2_geometry's real functions (WingPlanform, fuselage,
rotor, tail sizing) rather than re-deriving them, so this component and
the analytical pipeline in hpraptor.core.initial_sizing cannot drift.
"""

from __future__ import annotations
import numpy as np
import openmdao.api as om

from hpraptor.m2_geometry.planform import WingPlanform
from hpraptor.m2_geometry.fuselage import estimate_fuselage_geometry
from hpraptor.m2_geometry.rotor import size_rotors_from_disk_loading
from hpraptor.m2_geometry.tail_sizing import size_tail_from_wing

G = 9.80665


class GeometryComp(om.ExplicitComponent):
    """
    Vehicle geometry from MTOW and the geometric design variables.

    Wing area follows from the wing-loading design variable rather than
    being a free variable itself (S = W / (W/S)), matching the
    constraint-diagram formulation used by the analytical sizer.
    """

    def initialize(self):
        self.options.declare("n_rotors", default=4, types=int)
        self.options.declare("t_c", default=0.12, types=float)

    def setup(self):
        self.add_input("m_tow", val=20.0, units="kg", desc="Maximum takeoff mass")
        self.add_input("wing_loading", val=270.0, units="N/m**2", desc="W/S")
        self.add_input("AR", val=10.0, desc="Wing aspect ratio")
        self.add_input("disk_loading", val=300.0, units="N/m**2", desc="Rotor W/A")
        self.add_input("taper_ratio", val=1.0, desc="c_tip / c_root")

        self.add_output("S_ref", val=0.7, units="m**2", desc="Wing reference area")
        self.add_output("span", val=2.7, units="m")
        self.add_output("chord_mean", val=0.27, units="m",
                        desc="Mean geometric chord, S/b")
        self.add_output("chord_root", val=0.27, units="m")
        self.add_output("mac", val=0.27, units="m",
                        desc="Mean aerodynamic chord; equals chord_mean only "
                             "for a rectangular wing")
        self.add_output("e_oswald", val=0.78,
                        desc="Span efficiency, from AR and taper")
        self.add_output("wetted_wing", val=1.5, units="m**2")
        self.add_output("fuse_length", val=1.2, units="m")
        self.add_output("fuse_diameter", val=0.2, units="m")
        self.add_output("wetted_fuse", val=0.75, units="m**2")
        self.add_output("rotor_diameter", val=0.45, units="m")
        self.add_output("A_rotor", val=0.65, units="m**2", desc="Total rotor disk area")
        self.add_output("S_h", val=0.09, units="m**2", desc="Horizontal tail area")
        self.add_output("S_v", val=0.07, units="m**2", desc="Vertical tail area")
        self.add_output("tail_arm", val=1.07, units="m")

        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        m_tow = inputs["m_tow"][0]
        W = m_tow * G
        S = W / inputs["wing_loading"][0]
        AR = inputs["AR"][0]

        wing = WingPlanform(S=S, AR=AR, t_c=self.options["t_c"],
                            taper_ratio=inputs["taper_ratio"][0])
        fuse = estimate_fuselage_geometry(m_tow)
        rotor = size_rotors_from_disk_loading(W, inputs["disk_loading"][0],
                                              self.options["n_rotors"])
        tail = size_tail_from_wing(wing)

        outputs["S_ref"] = S
        outputs["span"] = wing.span
        outputs["chord_mean"] = wing.chord_mean
        outputs["chord_root"] = wing.chord_root
        outputs["mac"] = wing.mac
        outputs["e_oswald"] = wing.oswald_factor
        outputs["wetted_wing"] = wing.wetted_area
        outputs["fuse_length"] = fuse.length_m
        outputs["fuse_diameter"] = fuse.diameter_m
        outputs["wetted_fuse"] = fuse.wetted_area
        outputs["rotor_diameter"] = rotor.diameter_m
        outputs["A_rotor"] = rotor.disk_area_total_m2
        outputs["S_h"] = tail.S_h
        outputs["S_v"] = tail.S_v
        outputs["tail_arm"] = tail.l_t
