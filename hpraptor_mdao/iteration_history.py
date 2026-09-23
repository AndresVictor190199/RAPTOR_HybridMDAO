"""
What the driver did on the way to the answer.

A converged objective says where the optimizer stopped. It does not say
whether it walked there or fell into the first feasible corner it met, and
those are different claims — the same distinction `solver_stats` draws
between a KKT exit and an iteration-limit exit, but resolved per iteration
instead of once at the end.

This module records the driver's own iterates and renders them two ways:

  * `format_history`  — a terminal table, for when you are watching a run;
  * `plot_history`    — objective and worst constraint violation against
                        iteration, for when you are explaining one.

Both read the same recorded case file, so the picture and the table can
never disagree about what happened.

Why this exists at all: the `all_electric` rows in the campaign converge to
points ~0.8% apart depending only on floating-point path, because two of
their design variables (`k_electric`, `m_fuel`) are gated out of every
response and leave a null space in the QP subproblem. That is invisible in
a results table and obvious in an iteration trace.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import openmdao.api as om


@dataclass
class IterationHistory:
    """The driver's iterates, as arrays."""

    objective: np.ndarray                      # (n_iter,)
    constraints: Dict[str, np.ndarray]         # name -> (n_iter,) worst element
    design_vars: Dict[str, np.ndarray]         # name -> (n_iter,) norm
    objective_name: str = "objective"
    source: str = ""

    @property
    def n_iter(self) -> int:
        return len(self.objective)

    @property
    def violation(self) -> np.ndarray:
        """
        Worst constraint violation at each iterate.

        Every constraint in this problem is posed as g <= 0, so the
        violation is max(0, max_i g_i) and feasibility is exactly zero.
        Reported rather than the raw margins because one number per
        iterate is what makes the feasibility restoration visible.
        """
        if not self.constraints:
            return np.zeros_like(self.objective)
        stacked = np.vstack([v for v in self.constraints.values()])
        return np.maximum(stacked.max(axis=0), 0.0)


def attach_recorder(prob: om.Problem, path: str) -> str:
    """
    Record the driver's iterates to `path`.

    Call this after `build_problem` and BEFORE `prob.setup()`; OpenMDAO
    binds recorders during setup, so a recorder attached afterwards
    silently captures nothing.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # A stale file would be appended to, and the history would then contain
    # two runs with no marker between them.
    if os.path.exists(path):
        os.remove(path)

    rec = om.SqliteRecorder(path)
    prob.driver.add_recorder(rec)
    prob.driver.recording_options["record_desvars"] = True
    prob.driver.recording_options["record_objectives"] = True
    prob.driver.recording_options["record_constraints"] = True
    prob.driver.recording_options["includes"] = ["*"]
    return path


def read_history(path: str) -> IterationHistory:
    """Read a recorded case file back into arrays."""
    cr = om.CaseReader(path)
    cases = cr.list_cases("driver", out_stream=None)
    if not cases:
        raise RuntimeError(f"{path} contains no driver iterations")

    obj_name: Optional[str] = None
    obj: List[float] = []
    cons: Dict[str, List[float]] = {}
    dvs: Dict[str, List[float]] = {}

    for cid in cases:
        c = cr.get_case(cid)
        objectives = c.get_objectives()
        if obj_name is None:
            obj_name = list(objectives.keys())[0]
        obj.append(float(np.asarray(objectives[obj_name]).ravel()[0]))

        for k, v in c.get_constraints().items():
            # Worst element, so a vector constraint stays one number.
            cons.setdefault(k, []).append(float(np.max(np.asarray(v))))
        for k, v in c.get_design_vars().items():
            arr = np.asarray(v, dtype=float).ravel()
            dvs.setdefault(k, []).append(
                float(arr[0]) if arr.size == 1 else float(np.linalg.norm(arr)))

    return IterationHistory(
        objective=np.asarray(obj),
        constraints={k: np.asarray(v) for k, v in cons.items()},
        design_vars={k: np.asarray(v) for k, v in dvs.items()},
        objective_name=obj_name or "objective",
        source=path,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Terminal view
# ═══════════════════════════════════════════════════════════════════════════

def format_history(hist: IterationHistory, max_rows: int = 40,
                   show_dvs: bool = True) -> str:
    """
    The iteration trace as a terminal table.

    Long runs are thinned rather than truncated: the first and last
    iterates are always shown, because the interesting part of a run is
    usually where it started and where it stopped.
    """
    n = hist.n_iter
    idx = (np.arange(n) if n <= max_rows
           else np.unique(np.concatenate([[0], np.linspace(
               0, n - 1, max_rows - 1).astype(int), [n - 1]])))

    viol = hist.violation
    obj = hist.objective
    dv_names = sorted(hist.design_vars) if show_dvs else []
    # Keep the table inside a terminal: the six design variables that move
    # most across the run are the ones worth a column.
    if len(dv_names) > 6:
        spread = {k: float(np.ptp(v) / (np.abs(v).max() + 1e-30))
                  for k, v in hist.design_vars.items()}
        dv_names = sorted(sorted(spread, key=spread.get, reverse=True)[:6])

    head = f"{'iter':>5}{'objective':>14}{'d(obj)':>12}{'max viol':>12}  feas"
    head += "".join(f"{k.split('.')[-1][:11]:>12}" for k in dv_names)
    lines = [head, "-" * len(head)]

    for i in idx:
        d = "" if i == 0 else f"{obj[i] - obj[i - 1]:+.3e}"
        feas = "  ok " if viol[i] <= 1e-6 else "  NO "
        row = f"{i:>5}{obj[i]:>14.6g}{d:>12}{viol[i]:>12.3e}{feas}"
        row += "".join(f"{hist.design_vars[k][i]:>12.4g}" for k in dv_names)
        lines.append(row)

    lines.append("-" * len(head))
    lines.append(
        f"  {n} iterations   objective {obj[0]:.6g} -> {obj[-1]:.6g}"
        f"   ({(obj[-1] - obj[0]) / abs(obj[0]) * 100:+.2f}%)"
        f"   final violation {viol[-1]:.2e}")
    if n > len(idx):
        lines.append(f"  (thinned to {len(idx)} of {n} rows; "
                     f"first and last always shown)")

    # Which constraints are active at the end is the part a reader acts on.
    active = sorted(k for k, v in hist.constraints.items() if abs(v[-1]) < 1e-6)
    if active:
        lines.append("  active at exit: "
                     + ", ".join(a.split(".")[-1] for a in active))
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Graphical view
# ═══════════════════════════════════════════════════════════════════════════

#: Categorical slots 1-3 of the project palette, in fixed order. Never
#: cycled: a fourth series folds into "other" rather than inventing a hue.
_SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_GRID = "#dcdbd6"
_SURFACE = "#fcfcfb"


def plot_history(hist: IterationHistory, out_png: str,
                 title: str = "Driver iteration history") -> str:
    """
    Objective and feasibility against iteration, as a two-panel figure.

    Two panels rather than two y-axes on one: the objective is in Wh and
    the violation is dimensionless, and overlaying different scales on one
    frame is the single most misleading thing a chart can do.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = hist.n_iter
    it = np.arange(n)
    viol = hist.violation

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.5, 6.4), sharex=True, height_ratios=[2, 1])
    fig.patch.set_facecolor(_SURFACE)

    # ── objective ────────────────────────────────────────────────────────
    ax1.plot(it, hist.objective, color=_SERIES[0], lw=2.0,
             marker="o", ms=4, mfc=_SURFACE, mew=1.2, zorder=3)
    ax1.set_ylabel("objective  [Wh]", color=_INK_2, fontsize=10)
    ax1.set_title(title, color=_INK, fontsize=13, loc="left", pad=12)
    # The converged value is the number a reader takes away, so it is the
    # one that gets a direct label rather than a legend entry.
    ax1.annotate(f"{hist.objective[-1]:.4g} Wh",
                 xy=(it[-1], hist.objective[-1]),
                 xytext=(-8, 10), textcoords="offset points",
                 ha="right", color=_INK, fontsize=10, fontweight="bold")
    if n > 1 and hist.objective[0] > 0:
        ax1.annotate(f"start {hist.objective[0]:.4g} Wh", xy=(0, hist.objective[0]),
                     xytext=(8, 6), textcoords="offset points",
                     ha="left", color=_INK_2, fontsize=9)
    # A start value orders of magnitude above the optimum flattens the
    # whole trace; log scale keeps both ends readable.
    if hist.objective.min() > 0 and (
            hist.objective.max() / hist.objective.min() > 30):
        ax1.set_yscale("log")

    # ── feasibility ──────────────────────────────────────────────────────
    feas_floor = 1e-12
    ax2.plot(it, np.maximum(viol, feas_floor), color=_SERIES[1], lw=2.0,
             marker="o", ms=4, mfc=_SURFACE, mew=1.2, zorder=3)
    ax2.set_yscale("log")
    ax2.axhline(1e-6, color=_INK_2, lw=1.0, ls="--", zorder=2)
    # Left-anchored: the trace ends at the right, and a right-anchored
    # label sat on top of the final approach to the threshold.
    ax2.annotate("feasible  (g <= 1e-6)", xy=(0.006, 1e-6),
                 xycoords=("axes fraction", "data"),
                 xytext=(0, 5), textcoords="offset points",
                 ha="left", va="bottom", color=_INK_2, fontsize=9)
    ax2.set_ylabel("max violation", color=_INK_2, fontsize=10)
    ax2.set_xlabel("driver iteration", color=_INK_2, fontsize=10)

    for ax in (ax1, ax2):
        ax.set_facecolor(_SURFACE)
        ax.grid(True, color=_GRID, lw=0.8, alpha=0.9, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_GRID)
        ax.tick_params(colors=_INK_2, labelsize=9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=160, facecolor=_SURFACE)
    plt.close(fig)
    return out_png
