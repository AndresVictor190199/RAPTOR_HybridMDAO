"""
Parasite Drag Buildup — Component Skin-Friction Method
===========================================================

Computes the zero-lift (parasite) drag coefficient C_D0 from the wing
and fuselage wetted areas, using the standard component buildup
method: turbulent flat-plate skin friction (Prandtl-Schlichting) times
a form factor per component, summed and normalized by wing reference
area.

This closes the "Geometry -> Parasite Drag -> Power Feedback" link in
the mission sizing loop. Before this module, C_D0 was a hardcoded
constant (0.025) in every vehicle preset and in initial_sizing.py, with
no dependence on actual vehicle geometry — so changing wing area or
MTOW never changed drag, which is physically wrong and breaks the
sizing loop's feedback path.

References
----------
[1] Raymer, D. (2018). Aircraft Design: A Conceptual Approach. Ch.12
    (component buildup drag method).
[2] Hoerner, S.F. (1965). Fluid-Dynamic Drag. (typical form factors)

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from hpraptor.core.atmosphere import isa_density, isa_viscosity
from hpraptor.m2_geometry.planform import WingPlanform
from hpraptor.m2_geometry.fuselage import FuselageGeometry


def flat_plate_cf(reynolds: float) -> float:
    """
    Turbulent flat-plate skin friction coefficient
    (Prandtl-Schlichting formula): Cf = 0.455 / (log10(Re))^2.58.

    The Re >= 1 floor is applied on the REAL part so this stays valid
    under complex-step differentiation (OpenMDAO `method='cs'`); Python's
    max() raises on complex operands. Real-input behavior is unchanged.
    """
    if np.iscomplexobj(reynolds):
        reynolds = np.where(np.real(reynolds) > 1.0, reynolds, 1.0 + 0.0 * reynolds)
    else:
        reynolds = max(reynolds, 1.0)
    return 0.455 / (np.log10(reynolds) ** 2.58)


@dataclass
class ParasiteDragBuildup:
    """Component-by-component parasite drag breakdown."""
    reynolds_wing: float
    reynolds_fuselage: float
    cf_wing: float
    cf_fuselage: float
    cd0_wing: float
    cd0_fuselage: float
    cd0_misc: float
    cd0_total: float

    def summary(self) -> str:
        return (
            f"Parasite drag buildup: C_D0 = {self.cd0_total:.4f}\n"
            f"  Wing:     Cf={self.cf_wing:.5f}  (Re={self.reynolds_wing:.2e})  "
            f"-> C_D0={self.cd0_wing:.4f}\n"
            f"  Fuselage: Cf={self.cf_fuselage:.5f}  (Re={self.reynolds_fuselage:.2e})  "
            f"-> C_D0={self.cd0_fuselage:.4f}\n"
            f"  Misc (booms/gear/hubs): C_D0={self.cd0_misc:.4f}"
        )


def compute_parasite_drag(
    wing: WingPlanform,
    fuselage: FuselageGeometry,
    airspeed_ms: float,
    altitude_m: float,
    form_factor_wing: float = 1.25,
    form_factor_fuselage: float = 1.15,
    misc_drag_fraction: float = 0.08,
) -> ParasiteDragBuildup:
    """
    Compute C_D0 for a given wing + fuselage geometry at a flight
    condition, via the component buildup method.

    Parameters
    ----------
    wing : WingPlanform
    fuselage : FuselageGeometry
    airspeed_ms : float
        True airspeed [m/s] — sets the Reynolds number.
    altitude_m : float
        Altitude [m AMSL] — sets air density/viscosity via ISA.
    form_factor_wing, form_factor_fuselage : float
        Component form factors accounting for pressure drag beyond
        pure skin friction (thickness, shape). Typical values from
        Hoerner/Raymer for moderate-thickness wings and slender bodies.
    misc_drag_fraction : float
        Additional drag (booms, landing gear, rotor hubs, surface
        gaps/steps) as a fraction of the wing+fuselage clean C_D0.
    """
    rho = isa_density(altitude_m)
    mu = isa_viscosity(altitude_m)

    reynolds_wing = rho * airspeed_ms * wing.chord_mean / mu
    reynolds_fuselage = rho * airspeed_ms * fuselage.length_m / mu

    cf_wing = flat_plate_cf(reynolds_wing)
    cf_fuselage = flat_plate_cf(reynolds_fuselage)

    cd0_wing = cf_wing * form_factor_wing * wing.wetted_area / wing.S
    cd0_fuselage = cf_fuselage * form_factor_fuselage * fuselage.wetted_area / wing.S

    cd0_clean = cd0_wing + cd0_fuselage
    cd0_misc = cd0_clean * misc_drag_fraction
    cd0_total = cd0_clean + cd0_misc

    return ParasiteDragBuildup(
        reynolds_wing=reynolds_wing,
        reynolds_fuselage=reynolds_fuselage,
        cf_wing=cf_wing,
        cf_fuselage=cf_fuselage,
        cd0_wing=cd0_wing,
        cd0_fuselage=cd0_fuselage,
        cd0_misc=cd0_misc,
        cd0_total=cd0_total,
    )


def parasite_cd0_from_wetted(
    S_ref: float,
    chord_mean: float,
    wetted_wing: float,
    fuse_length: float,
    wetted_fuse: float,
    airspeed_ms: float,
    altitude_m: float,
    form_factor_wing: float = 1.25,
    form_factor_fuselage: float = 1.15,
    misc_drag_fraction: float = 0.08,
):
    """
    C_D0 from wetted areas that were MEASURED elsewhere.

    ``compute_parasite_drag`` takes geometry objects and derives wetted area
    from them. That is fine when the analytical planform is the only source,
    but it makes the buildup deaf to a better one: an AeroSandbox assembly
    can measure the real surfaces, and on this airframe it finds 24% less
    wetted fuselage than the analytical formula assumes. Passing areas in
    rather than deriving them is what lets that measurement reach the drag.

    Complex-step safe: no branches, no comparisons, no geometry objects.
    """
    rho = isa_density(altitude_m)
    mu = isa_viscosity(altitude_m)

    cf_wing = flat_plate_cf(rho * airspeed_ms * chord_mean / mu)
    cf_fuse = flat_plate_cf(rho * airspeed_ms * fuse_length / mu)

    cd0_wing = cf_wing * form_factor_wing * wetted_wing / S_ref
    cd0_fuse = cf_fuse * form_factor_fuselage * wetted_fuse / S_ref
    clean = cd0_wing + cd0_fuse
    return clean * (1.0 + misc_drag_fraction)
