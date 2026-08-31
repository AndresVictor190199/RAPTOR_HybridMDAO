"""
Optimization problem formulation, read back from the model itself.

Prints the standard statement

    minimize    f(x)
    by varying  x_i_lower <= x_i <= x_i_upper      i = 1..n_x
    subject to  g_j(x) <= 0                        j = 1..n_g
                h_l(x)  = 0                        l = 1..n_h

by INTROSPECTING the OpenMDAO problem — design variables, objective, and
constraints are read from what was actually declared on the model, not
from a hand-maintained list. A formulation written out separately drifts
from the code the first time a bound changes; this one cannot.

Constraints declared with `upper=` or `lower=` are reported as
inequalities and normalized to the g(x) <= 0 convention; those declared
with `equals=` are reported as equalities h(x) = 0.
"""

from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np
import openmdao.api as om

BIG = 1e29  # OpenMDAO's stand-in for "unbounded"


def _derivative_scheme(model) -> str:
    """
    Report which approximation schemes the components actually declare.

    This used to be the hard-coded claim "per-component complex step", which
    stopped being true the moment an AeroSandbox component entered the model:
    AeroSandbox calls ``arctan2``, which has no complex loop, so those
    components must declare ``method="fd"``. A formulation printout that
    states the wrong differentiation scheme is worse than one that omits it,
    so this reads the schemes off the built model instead of asserting them.
    """
    schemes = set()
    for sub in model.system_iter(recurse=True, typ=om.ExplicitComponent):
        schemes.update(getattr(sub, "_approx_schemes", {}).keys())
    label = {"cs": "complex step", "fd": "finite difference"}
    if not schemes:
        return "per-component analytic partials"
    named = sorted(label.get(x, x) for x in schemes)
    return "per-component " + " + ".join(named)


def _fmt(v, width: int = 12) -> str:
    """Format a bound, showing an unbounded side as a dash."""
    if v is None:
        return "-".rjust(width)
    a = np.atleast_1d(np.asarray(v, dtype=float))
    if np.all(np.abs(a) >= BIG):
        return ("-inf" if a.flat[0] < 0 else "+inf").rjust(width)
    if a.size == 1:
        return f"{a.flat[0]:.4g}".rjust(width)
    return f"[{a.min():.3g}, {a.max():.3g}]".rjust(width)


def _size(meta: Dict) -> int:
    for key in ("global_size", "size"):
        if meta.get(key):
            return int(meta[key])
    val = meta.get("val")
    return int(np.atleast_1d(val).size) if val is not None else 1


def describe(problem, title: str = "OPTIMIZATION PROBLEM", notes: Optional[List[str]] = None) -> str:
    """
    Render the formulation of an OpenMDAO problem.

    `problem.final_setup()` must have been called (or the problem run) so
    that declared metadata is populated.
    """
    model = problem.model
    dvs = model.get_design_vars(recurse=True, get_sizes=True, use_prom_ivc=True)
    objs = model.get_objectives(recurse=True, get_sizes=True, use_prom_ivc=True)
    cons = model.get_constraints(recurse=True, get_sizes=True, use_prom_ivc=True)

    ineq, eq = [], []
    for name, meta in cons.items():
        if meta.get("equals") is not None:
            eq.append((name, meta))
        else:
            ineq.append((name, meta))

    n_x = sum(_size(m) for m in dvs.values())
    n_g = sum(_size(m) for _, m in ineq)
    n_h = sum(_size(m) for _, m in eq)

    W = 78
    L = ["=" * W, title, "=" * W, ""]

    obj_name = next(iter(objs), "(none declared)")
    L += [
        f"  minimize     f(x)  =  {obj_name}",
        f"  by varying   x_i,lower <= x_i <= x_i,upper       i = 1 .. {n_x}",
        f"  subject to   g_j(x) <= 0                         j = 1 .. {n_g}",
    ]
    if n_h:
        L.append(f"               h_l(x)  = 0                         l = 1 .. {n_h}")
    L.append("")

    # ── Design variables ─────────────────────────────────────────────────
    L += ["-" * W, f"  DESIGN VARIABLES   n_x = {n_x}", "-" * W,
          f"  {'#':<10}{'variable':<24}{'lower':>12}{'upper':>12}   {'size':>4}"]
    idx = 1
    for name, meta in dvs.items():
        size = _size(meta)
        label = f"x{idx}" if size == 1 else f"x{idx}..{idx + size - 1}"
        L.append(f"  {label:<10}{name:<24}{_fmt(meta.get('lower'))}"
                 f"{_fmt(meta.get('upper'))}   {size:>4}")
        idx += size
    L.append("")

    # ── Inequality constraints ───────────────────────────────────────────
    L += ["-" * W, f"  INEQUALITY CONSTRAINTS   n_g = {n_g}", "-" * W]
    if ineq:
        L.append(f"  {'#':<5}{'expression':<52}{'size':>4}")
        j = 1
        for name, meta in ineq:
            size = _size(meta)
            up, lo = meta.get("upper"), meta.get("lower")
            if up is not None and np.all(np.abs(np.atleast_1d(up)) < BIG):
                expr = f"{name} - ({_fmt(up, 0).strip()}) <= 0"
                L.append(f"  g{j:<4}{expr:<52}{size:>4}")
                j += size
            if lo is not None and np.all(np.abs(np.atleast_1d(lo)) < BIG):
                expr = f"({_fmt(lo, 0).strip()}) - {name} <= 0"
                L.append(f"  g{j:<4}{expr:<52}{size:>4}")
                j += size
    else:
        L.append("  (none declared)")
    L.append("")

    # ── Equality constraints ─────────────────────────────────────────────
    if eq:
        L += ["-" * W, f"  EQUALITY CONSTRAINTS   n_h = {n_h}", "-" * W,
              f"  {'#':<5}{'expression':<52}{'size':>4}"]
        for l_i, (name, meta) in enumerate(eq, start=1):
            expr = f"{name} - ({_fmt(meta['equals'], 0).strip()}) = 0"
            L.append(f"  h{l_i:<4}{expr:<52}{_size(meta):>4}")
        L.append("")

    # ── Solver / driver ──────────────────────────────────────────────────
    driver = type(problem.driver).__name__
    try:
        opt = problem.driver.options["optimizer"]
    except (KeyError, AttributeError):
        opt = "n/a"
    L += ["-" * W, "  SOLUTION METHOD", "-" * W,
          f"  driver           {driver} ({opt})",
          f"  derivatives      {_derivative_scheme(model)}; totals via OpenMDAO's "
          f"unified\n                   derivatives equation ({problem._mode} mode)"]
    nl = getattr(model, "nonlinear_solver", None)
    if nl is not None:
        try:
            maxiter = nl.options["maxiter"]
        except (KeyError, AttributeError):
            maxiter = "?"
        L.append(f"  coupling solver  {type(nl).__name__} (maxiter={maxiter})")
    L.append("")

    if notes:
        L += ["-" * W, "  NOTES", "-" * W]
        L += [f"  - {n}" for n in notes]
        L.append("")

    L.append("=" * W)
    return "\n".join(L)


def print_formulation(problem, title: str = "OPTIMIZATION PROBLEM",
                      notes: Optional[List[str]] = None) -> None:
    print(describe(problem, title, notes))
