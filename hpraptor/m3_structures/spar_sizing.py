"""
Wing Spar Sizing — Thin-Wall Box Beam Bending Model
=======================================================

Analytical, preliminary-design-level structural model for the wing spar:
given a WingPlanform, an ultimate load case, and a spar material, computes
the root bending moment, resulting bending stress in a thin-walled box
spar of a given web thickness, the stress margin against the g4 constraint
from uav_mdo_framework_spec.md, and the resulting spar mass.

Modeling assumptions (preliminary/conceptual design fidelity — matches
the rest of hpraptor, not a substitute for FEA):
  - Spanwise lift distribution is elliptical, giving the standard
    root bending moment M_root = n_ult * W * b / (3*pi) for a wing of
    span b carrying weight W at ultimate load factor n_ult.
  - The spar is idealized as a thin-walled rectangular box beam of
    depth h = t_c * chord_mean (the local airfoil thickness) and width
    w = spar_chord_fraction * chord_mean, with uniform wall thickness
    t_spar (the spec's design variable) and constant section along span.
  - Only the spar's primary bending strength is sized here; secondary
    structure (ribs, skin) mass is handled separately in mass_buildup.py.

References
----------
[1] Kroo, I. (2001). Aircraft Design: Synthesis and Analysis. Ch. 9.
[2] Niu, M.C.Y. (1988). Airframe Structural Design.
[3] uav_mdo_framework_spec.md — g4 (wing stress) constraint and the
    t_spar / N_ribs design variables (Sec. 3.2-3.3).

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from hpraptor.m2_geometry.planform import WingPlanform
from .materials import SparMaterial, CFRP_UNIDIRECTIONAL

G = 9.80665  # m/s^2


@dataclass
class WingLoadCase:
    """
    Ultimate structural load case for wing spar sizing.

    Attributes
    ----------
    mtow_kg : float
        Maximum takeoff mass [kg] — the weight the wing must carry.
    load_factor_ultimate : float
        Ultimate load factor n_ult [-]. Default 3.8 matches the spec's
        g4 constraint (a 2.5g limit load with a standard 1.5x ultimate
        factor gives n_ult = 3.75; 3.8 is the conventional CS-23/FAR-23
        rounded value used in the spec).
    safety_factor : float
        Additional margin applied to material yield stress to get the
        allowable stress (sigma_allow = yield / safety_factor). Default
        1.5, matching the spec's g4 constraint exactly.
    """
    mtow_kg: float
    load_factor_ultimate: float = 3.8
    safety_factor: float = 1.5

    @property
    def W_ultimate_N(self) -> float:
        """Ultimate design load [N] = n_ult * MTOW * g."""
        return self.load_factor_ultimate * self.mtow_kg * G


@dataclass
class SparSizingResult:
    """Result of sizing/checking a wing spar for a given web thickness."""
    t_spar_mm: float
    root_bending_moment_Nm: float
    spar_depth_m: float
    spar_width_m: float
    I_xx_m4: float
    sigma_max_pa: float
    sigma_allow_pa: float
    stress_margin: float   # g4: sigma_max/sigma_allow - 1.0 <= 0 is feasible
    is_feasible: bool
    spar_mass_kg: float

    def summary(self) -> str:
        status = "FEASIBLE" if self.is_feasible else "INFEASIBLE"
        return (
            f"Spar sizing (t_spar={self.t_spar_mm:.2f} mm): {status}\n"
            f"  Root bending moment: {self.root_bending_moment_Nm:.1f} N*m\n"
            f"  Section: depth={self.spar_depth_m*1000:.1f} mm, "
            f"width={self.spar_width_m*1000:.1f} mm, I_xx={self.I_xx_m4:.3e} m^4\n"
            f"  Stress: {self.sigma_max_pa/1e6:.1f} MPa "
            f"(allow {self.sigma_allow_pa/1e6:.1f} MPa, "
            f"margin g4={self.stress_margin:+.3f})\n"
            f"  Spar mass: {self.spar_mass_kg:.3f} kg"
        )


class WingStructuralSizer:
    """
    Sizes/checks a wing spar against the g4 stress constraint for a given
    WingPlanform and WingLoadCase.

    Example
    -------
    >>> planform = WingPlanform(S=0.66, AR=10.0, t_c=0.12)
    >>> load_case = WingLoadCase(mtow_kg=18.9)
    >>> sizer = WingStructuralSizer(planform, load_case)
    >>> result = sizer.size_spar(t_spar_mm=3.0)
    >>> print(result.summary())
    """

    def __init__(self, planform: WingPlanform, load_case: WingLoadCase,
                 material: SparMaterial = CFRP_UNIDIRECTIONAL,
                 spar_chord_fraction: float = 0.5):
        """
        Parameters
        ----------
        planform : WingPlanform
        load_case : WingLoadCase
        material : SparMaterial
            Spar material. Default unidirectional CFRP.

            Aluminium 6061-T6 was the previous default and is not what
            this class of aircraft is built from: a 10 kg VTOL carries a
            filament-wound carbon tube, not an aluminium box. CFRP has
            3.7x the specific strength (375 vs 102 kN.m/kg), so the
            aluminium spar was simultaneously heavier than reality and
            sitting on its manufacturing floor with 26-40% stress margin,
            which is why the structure never pushed back on aspect ratio.
        spar_chord_fraction : float
            Spar box width as a fraction of the mean chord [-]. Default
            0.5 (a single/dual-spar box spanning roughly the front half
            of the chord — a standard preliminary-design assumption).
        """
        self.planform = planform
        self.load_case = load_case
        self.material = material
        self.spar_chord_fraction = spar_chord_fraction

    def root_bending_moment(self) -> float:
        """
        Wing root bending moment [N*m] assuming an elliptical spanwise
        lift distribution: M_root = n_ult * W_ultimate * b / (3*pi).
        """
        b = self.planform.span
        return self.load_case.W_ultimate_N * b / (3.0 * np.pi)

    def size_spar(self, t_spar_mm: float) -> SparSizingResult:
        """
        Compute bending stress, g4 margin, and mass for a given spar web
        thickness (the spec's t_spar design variable, bounds [1, 6] mm).
        """
        t = t_spar_mm / 1000.0  # mm -> m
        # Sized at the ROOT chord, which is where the bending moment this
        # method checks actually acts. Using the mean chord was equivalent
        # while every wing was rectangular, but understates a tapered
        # wing's root spar: taper deepens the root box exactly where the
        # load peaks, and that structural credit is a real part of why
        # taper pays. At taper_ratio = 1 chord_root == chord_mean, so
        # rectangular wings are unaffected.
        c_root = self.planform.chord_root
        h = self.planform.t_c * c_root                  # spar depth
        w = self.spar_chord_fraction * c_root           # spar width

        # Thin-walled rectangular box beam: two flanges (width w) at
        # +-h/2 plus two webs (height h), all of wall thickness t.
        I_xx = w * t * h**2 / 2.0 + t * h**3 / 6.0

        M_root = self.root_bending_moment()
        # Guard the degenerate I_xx <= 0 case (zero/negative geometry) on the
        # REAL part, so this expression also survives complex-step
        # differentiation (OpenMDAO `method='cs'`) — a bare `I_xx > 0`
        # comparison raises on complex operands. For any physical geometry
        # the guard does not bind and the result is unchanged.
        I_xx_real = np.real(I_xx)
        if np.iscomplexobj(I_xx):
            sigma_max = np.where(I_xx_real > 0, M_root * (h / 2.0) / I_xx,
                                 np.inf + 0.0 * I_xx)
        else:
            sigma_max = M_root * (h / 2.0) / I_xx if I_xx_real > 0 else np.inf

        sigma_allow = self.material.yield_stress_pa / self.load_case.safety_factor
        stress_margin = sigma_max / sigma_allow - 1.0  # g4

        # Mass integrates along the span, so it scales with the MEAN chord,
        # not the root chord that sizes the stress above. A real spar box
        # tapers with the wing: its section area falls roughly linearly with
        # local chord, and integrating that over the span gives the mean.
        # Multiplying the root section by the full span instead makes a
        # tapered wing carry a root-sized box all the way to the tip -- it
        # put 54% of extra mass on a lambda = 0.3 wing and completely buried
        # the 6% L/D that taper actually buys.
        h_mean = self.planform.t_c * self.planform.chord_mean
        w_mean = self.spar_chord_fraction * self.planform.chord_mean
        cross_section_area = 2 * t * (w_mean + h_mean)
        spar_mass = cross_section_area * self.planform.span * self.material.density_kg_m3

        return SparSizingResult(
            t_spar_mm=t_spar_mm,
            root_bending_moment_Nm=M_root,
            spar_depth_m=h,
            spar_width_m=w,
            I_xx_m4=I_xx,
            sigma_max_pa=sigma_max,
            sigma_allow_pa=sigma_allow,
            stress_margin=stress_margin,
            # Feasibility is a real-valued predicate; take the real part so a
            # complex-step evaluation still yields a usable flag.
            is_feasible=bool(np.real(stress_margin) <= 0.0),
            spar_mass_kg=spar_mass,
        )

    def minimum_feasible_t_spar_mm(self, t_min: float = 1.0, t_max: float = 6.0,
                                   tol: float = 1e-3) -> float:
        """
        Bisection search for the thinnest (lightest) spar web thickness
        within [t_min, t_max] mm that still satisfies the g4 stress
        constraint. Raises ValueError if even t_max is infeasible.
        """
        if not self.size_spar(t_max).is_feasible:
            raise ValueError(
                f"No feasible spar within bounds: even t_spar={t_max} mm "
                f"violates the stress constraint (margin="
                f"{self.size_spar(t_max).stress_margin:+.3f})."
            )
        lo, hi = t_min, t_max
        if self.size_spar(lo).is_feasible:
            return lo
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if self.size_spar(mid).is_feasible:
                hi = mid
            else:
                lo = mid
        return hi
