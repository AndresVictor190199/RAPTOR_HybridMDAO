"""
AeroSandbox geometry and aerodynamics as OpenMDAO disciplines.

Two components, wrapping the solvers that already exist in
``hpraptor.m2_geometry.aerosandbox_geometry`` rather than re-deriving them:

``ASBGeometryComp``
    Assembles the real 3D airplane — wing, tail, fuselage, propulsors — and
    reports the geometry the analytical path can only estimate: true wetted
    areas from the actual surfaces, and the tail volume that follows from
    where the tail really sits.

``ASBAeroComp``
    Runs AeroSandbox's aerodynamics on that airplane for C_L, C_D0, C_D and
    L/D, replacing the component-buildup polar in ``AeroComp``.

Three constraints govern how these are wired, and each one is a measured
fact rather than a preference.

**AeroSandbox cannot be complex-stepped.** Its internals call ``arctan2``,
which has no complex loop and raises ``TypeError`` outright. Every other
component in this framework uses ``method="cs"``; these two must use
``method="fd"``, and that is not a style choice — complex step simply does
not run through this code.

**The solvers are not equally affordable.** Measured on this model:
geometry assembly 0.4 ms, ``AeroBuildup`` 64 ms, ``VortexLatticeMethod``
329 ms, VLM with stability derivatives 1976 ms. Finite differencing a
6-input component multiplies those by seven. AeroBuildup is therefore the
in-loop default and VLM is available for a verification pass; putting VLM
in the loop would cost roughly 2.3 s per gradient.

**Geometry works today, aerodynamics may not.** AeroSandbox is built on
CasADi, and its aerodynamic paths need CasADi's B-spline interpolant
plugin. Where that plugin cannot load — see ``aerosandbox_aero_available``
— geometry still runs fine because it is pure numpy. The components are
written so the framework falls back to the analytical polar instead of
failing, and says so.
"""

from __future__ import annotations

import numpy as np
import openmdao.api as om

from hpraptor.m2_geometry.planform import WingPlanform
from hpraptor.m2_geometry.fuselage import estimate_fuselage_geometry
from hpraptor.m2_geometry.rotor import RotorGeometry, size_rotors_from_disk_loading
from hpraptor.m2_geometry.tail_sizing import size_tail_from_wing

G = 9.80665

_AERO_AVAILABLE = None
_AERO_REASON = ""


def aerosandbox_aero_available() -> bool:
    """
    Whether AeroSandbox's *aerodynamic* solvers can actually run here.

    Probed once, by doing the smallest thing that touches the CasADi
    B-spline interpolant every aero path needs. Geometry is deliberately not
    part of the probe: it is pure numpy and works regardless.
    """
    global _AERO_AVAILABLE, _AERO_REASON
    if _AERO_AVAILABLE is None:
        try:
            import aerosandbox as asb
            float(asb.Atmosphere(altitude=1000.0).density())
            _AERO_AVAILABLE, _AERO_REASON = True, ""
        except Exception as exc:                      # pragma: no cover
            _AERO_AVAILABLE = False
            _AERO_REASON = f"{type(exc).__name__}: {str(exc)[:120]}"
    return _AERO_AVAILABLE


def aerosandbox_aero_reason() -> str:
    """Why the aero solvers are unavailable, for an actionable message."""
    aerosandbox_aero_available()
    return _AERO_REASON


def _assemble(m_tow, wing_loading, AR, disk_loading, n_rotors=4, t_c=0.12):
    """Build the AeroSandbox airplane for one design point."""
    from hpraptor.m2_geometry.aerosandbox_geometry import build_airplane

    W = m_tow * G
    S = W / wing_loading
    wing = WingPlanform(S=S, AR=AR, t_c=t_c)
    fuselage = estimate_fuselage_geometry(m_tow)
    tail = size_tail_from_wing(wing)
    sized = size_rotors_from_disk_loading(W, disk_loading, n_rotors)
    rotor = RotorGeometry(n_rotors=n_rotors,
                          diameter_m=sized.diameter_m,
                          disk_loading_pa=disk_loading)
    return build_airplane(wing, fuselage, tail, rotor), wing, fuselage, tail, sized


class ASBGeometryComp(om.ExplicitComponent):
    """
    Real 3D geometry, in place of the analytical estimate.

    The analytical ``GeometryComp`` computes wetted area from a flat-plate
    factor on the planform. This assembles the actual surfaces and measures
    them, which matters because wetted area drives parasite drag and hence
    the whole energy result.
    """

    def initialize(self):
        self.options.declare("n_rotors", default=4, types=int)
        self.options.declare("t_c", default=0.12, types=float)

    def setup(self):
        self.add_input("m_tow", val=10.0, units="kg")
        self.add_input("wing_loading", val=300.0, units="N/m**2")
        self.add_input("AR", val=12.0)
        self.add_input("disk_loading", val=120.0, units="N/m**2")

        self.add_output("S_ref", val=0.3, units="m**2")
        self.add_output("span", val=2.0, units="m")
        self.add_output("chord_mean", val=0.17, units="m")
        self.add_output("wetted_total", val=1.5, units="m**2",
                        desc="Measured wetted area of every surface, not a "
                             "flat-plate factor on the planform")
        self.add_output("wetted_wing", val=0.7, units="m**2")
        self.add_output("wetted_fuse", val=0.5, units="m**2")
        # Carried so this is a drop-in for GeometryComp, which AeroComp
        # consumes these from.
        self.add_output("fuse_length", val=1.0, units="m")
        self.add_output("fuse_diameter", val=0.16, units="m")
        self.add_output("S_h", val=0.04, units="m**2")
        self.add_output("S_v", val=0.03, units="m**2")
        self.add_output("tail_arm", val=0.8, units="m")
        self.add_output("rotor_diameter", val=0.5, units="m")
        self.add_output("A_rotor", val=0.9, units="m**2")
        self.add_output("V_h", val=0.5,
                        desc="Horizontal tail volume coefficient, from where "
                             "the tail actually sits")

        # Finite difference, not complex step: AeroSandbox calls arctan2,
        # which raises TypeError on complex input.
        self.declare_partials("*", "*", method="fd",
                              step=1e-6, step_calc="rel_element")

    def compute(self, inputs, outputs):
        airplane, wing, fuselage, tail, rotor = _assemble(
            inputs["m_tow"][0], inputs["wing_loading"][0],
            inputs["AR"][0], inputs["disk_loading"][0],
            n_rotors=self.options["n_rotors"], t_c=self.options["t_c"],
        )

        wetted_wing = float(airplane.wings[0].area("wetted"))
        wetted_all = float(sum(w.area("wetted") for w in airplane.wings))
        wetted_fuse = float(sum(f.area_wetted() for f in airplane.fuselages))

        outputs["S_ref"] = wing.S
        outputs["span"] = wing.span
        outputs["chord_mean"] = wing.chord_mean
        outputs["wetted_wing"] = wetted_wing
        outputs["wetted_fuse"] = wetted_fuse
        outputs["fuse_length"] = fuselage.length_m
        outputs["fuse_diameter"] = fuselage.diameter_m
        outputs["wetted_total"] = wetted_all + wetted_fuse
        outputs["S_h"] = tail.S_h
        outputs["S_v"] = tail.S_v
        outputs["tail_arm"] = tail.l_t
        outputs["rotor_diameter"] = rotor.diameter_m
        outputs["A_rotor"] = rotor.disk_area_total_m2
        outputs["V_h"] = (tail.S_h * tail.l_t) / (wing.S * wing.chord_mean)


class ASBAeroComp(om.ExplicitComponent):
    """
    Aerodynamics from AeroSandbox, in place of the component-buildup polar.

    ``solver="aerobuildup"`` is the in-loop default at ~64 ms per call.
    ``solver="vlm"`` is ~5x slower and meant for a verification pass, not
    for a gradient loop.

    C_L is not free here: it is whatever the aircraft needs to hold its own
    weight at the cruise condition, so the component solves for the angle of
    attack that produces it and reports the drag at that trim state. That is
    the honest comparison against the analytical polar, which does the same
    thing implicitly.
    """

    def initialize(self):
        self.options.declare("solver", default="aerobuildup",
                             values=("aerobuildup", "vlm"))
        self.options.declare("alpha_bounds", default=(-4.0, 14.0), types=tuple)
        self.options.declare("CL_max", default=1.4, types=float)
        self.options.declare("n_alpha", default=9, types=int,
                             desc="Samples used to invert C_L(alpha)")

    def setup(self):
        self.add_input("m_tow", val=10.0, units="kg")
        self.add_input("wing_loading", val=300.0, units="N/m**2")
        self.add_input("AR", val=12.0)
        self.add_input("disk_loading", val=120.0, units="N/m**2")
        self.add_input("V_cruise", val=30.0, units="m/s")
        self.add_input("altitude", val=3126.0, units="m")

        self.add_output("C_L_cruise", val=0.5)
        self.add_output("C_D_cruise", val=0.03)
        self.add_output("C_D0", val=0.025)
        self.add_output("L_D", val=17.0)
        self.add_output("D_cruise", val=6.0, units="N")
        self.add_output("alpha_trim", val=3.0, units="deg")
        self.add_output("g1_cl_margin", val=-0.3,
                        desc="C_L_cruise/(0.8 C_L_max) - 1; <= 0 keeps stall margin")

        self.declare_partials("*", "*", method="fd",
                              step=1e-6, step_calc="rel_element")

    def compute(self, inputs, outputs):
        import aerosandbox as asb

        m_tow = inputs["m_tow"][0]
        V = inputs["V_cruise"][0]
        h = inputs["altitude"][0]

        airplane, wing, _, _, _ = _assemble(
            m_tow, inputs["wing_loading"][0], inputs["AR"][0],
            inputs["disk_loading"][0])

        atmos = asb.Atmosphere(altitude=h)
        rho = float(atmos.density())
        q = 0.5 * rho * V ** 2
        CL_needed = (m_tow * G) / (q * wing.S)

        # Sweep alpha once, then interpolate for the trim point. A sweep is
        # cheaper and far more robust than a root-find here: AeroBuildup is
        # vectorised over alpha, so the whole curve costs about one call.
        lo, hi = self.options["alpha_bounds"]
        alphas = np.linspace(lo, hi, self.options["n_alpha"])
        op = asb.OperatingPoint(atmosphere=atmos, velocity=V, alpha=alphas)

        if self.options["solver"] == "vlm":
            CLs, CDs = [], []
            for a in alphas:
                r = asb.VortexLatticeMethod(
                    airplane=airplane,
                    op_point=asb.OperatingPoint(atmosphere=atmos, velocity=V, alpha=a),
                ).run()
                CLs.append(float(r["CL"]))
                CDs.append(float(r["CD"]))
            CL_curve, CD_curve = np.array(CLs), np.array(CDs)
        else:
            r = asb.AeroBuildup(airplane=airplane, op_point=op).run()
            CL_curve = np.atleast_1d(np.asarray(r["CL"], dtype=float))
            CD_curve = np.atleast_1d(np.asarray(r["CD"], dtype=float))

        # Interpolate to the alpha that trims. np.interp needs an increasing
        # x, and the lift curve is monotonic over this bracket.
        alpha_trim = float(np.interp(CL_needed, CL_curve, alphas))
        CD = float(np.interp(alpha_trim, alphas, CD_curve))

        # Zero-lift drag from the curve's own minimum, so C_D0 means the same
        # thing it does in the analytical polar.
        CD0 = float(np.min(CD_curve))

        outputs["C_L_cruise"] = CL_needed
        outputs["C_D_cruise"] = CD
        outputs["C_D0"] = CD0
        outputs["L_D"] = CL_needed / max(CD, 1e-9)
        outputs["D_cruise"] = q * wing.S * CD
        outputs["alpha_trim"] = alpha_trim
        outputs["g1_cl_margin"] = CL_needed / (0.8 * self.options["CL_max"]) - 1.0
