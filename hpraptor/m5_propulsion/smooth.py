"""
Smooth, complex-step-safe primitives shared by every differentiable model.

Kept in their own module so the propulsion physics (``architecture_np``) and
the rotor/motor/battery physics (``rotor_power``) can both use them without
importing each other. ``architecture_np`` re-exports them under their old
names, so existing imports keep working.

COMPLEX-STEP RULE: any comparison is taken on the real part only; values
returned are never truncated to it.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _re(x: Any) -> Any:
    """Real part, for comparisons only (never in returned values)."""
    return np.real(x)


def _smoothing_eps(scale: float) -> float:
    """
    Smoothing epsilon sized to the operands' magnitude.

    smooth_max/smooth_min deviate from the true max/min by at most
    sqrt(eps)/2, so an absolute eps is only meaningful relative to the
    operands. ``(1e-3 * scale)^2`` keeps the error at 0.05 % of ``scale``.
    """
    return (1e-3 * scale) ** 2


def smooth_max(a: Any, b: Any, scale: Any = 1.0, eps: float = None) -> Any:
    """Twice-differentiable max(a, b); complex-step safe."""
    if eps is None:
        eps = _smoothing_eps(scale)
    return 0.5 * (a + b + np.sqrt((a - b) ** 2 + eps))


def smooth_min(a: Any, b: Any, scale: Any = 1.0, eps: float = None) -> Any:
    """Twice-differentiable min(a, b); complex-step safe."""
    if eps is None:
        eps = _smoothing_eps(scale)
    return 0.5 * (a + b - np.sqrt((a - b) ** 2 + eps))


def smooth_switch(P: Any, width: float = 10.0, cap: float = 30.0) -> Any:
    """
    Smooth 0->1 gate, tanh(P / width), clamped so the complex tanh cannot
    overflow (it is evaluated through sinh/cosh, which overflow past ~350).
    """
    arg = P / width
    arg = np.where(_re(arg) > cap, cap + 0.0 * arg, arg)
    arg = np.where(_re(arg) < -cap, -cap + 0.0 * arg, arg)
    return np.tanh(arg)
