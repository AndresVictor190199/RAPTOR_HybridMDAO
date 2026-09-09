"""
Run every architecture through the sizing MDO and tabulate what happened.

One study answers "what is the best design?". A campaign answers the two
questions a methodology section actually has to answer: *how does the answer
change across the design space you claim to cover*, and *did the optimizer
converge in each case or merely stop*. The second is why every row carries
its exit status and iteration count rather than only its objective — a
result that hit an iteration limit is a different claim from one that met
its KKT test, and a table that reports only the objective hides which is
which.

Each architecture is run **pinned**, not relaxed. Pinning holds ``z_arch``
one-hot and removes it from the design vector, so the row is a clean
single-architecture optimum. The relaxed run is a separate row: it lets the
optimizer choose, and the interesting comparison is whether it lands on the
same design as the pinned winner.

Usage
-----
    python -m hpraptor_mdao.run all          # campaign + every diagram
    python -m hpraptor_mdao.run campaign     # just the table
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

RESULTS = Path("results")

#: Columns of the summary table: (result key, header, format, width).
#: ``solver.*`` keys are read out of the nested solver-stats block.
COLUMNS = [
    ("architecture",       "architecture",  "{:s}",     "<22"),
    ("aero",               "aero",          "{:s}",     "<12"),
    ("m_tow",              "MTOW",          "{:.3f}",   ">8"),
    ("S_ref",              "S_ref",         "{:.4f}",   ">8"),
    ("AR",                 "AR",            "{:.2f}",   ">6"),
    ("L_D",                "L/D",           "{:.2f}",   ">6"),
    ("C_D0",               "C_D0",          "{:.5f}",   ">8"),
    ("energy_mission_wh",  "E_miss",        "{:.2f}",   ">8"),
    ("energy_primary_wh",  "E_prim",        "{:.2f}",   ">8"),
    ("m_battery",          "m_batt",        "{:.3f}",   ">7"),
    ("m_fuel_carried",     "m_fuel",        "{:.3f}",   ">7"),
    ("k_electric",         "k_elec",        "{:.3f}",   ">7"),
    ("m_propulsion",       "m_prop",        "{:.3f}",   ">7"),
    ("solver.nit",         "iters",         "{:d}",     ">6"),
    ("solver.nfev",        "nfev",          "{:d}",     ">6"),
    ("solver.njev",        "njev",          "{:d}",     ">6"),
    ("wall_s",             "time_s",        "{:.1f}",   ">7"),
    ("solver.exit_status", "exit",          "{:s}",     "<9"),
]

#: Constraints reported in the second table. Anything > 0 is a violation.
CONSTRAINTS = [
    ("g1_cl_margin",         "g1 stall"),
    ("g2_energy_margin",     "g2 energy"),
    ("g3_soc_margin",        "g3 SOC"),
    ("g4_stress_margin",     "g4 stress"),
    ("g5_terrain_clearance", "g5 terrain"),
    ("g6_battery_power",     "g6 C-rate"),
    ("g7_rotor_fit",         "g7 rotor"),
    ("g8_reynolds",          "g8 Re"),
    ("g10_fuel_energy",      "g10 fuel"),
]


def _dig(row: Dict, key: str):
    """Fetch ``a.b`` from a nested dict, returning None when absent."""
    node = row
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def run_campaign(
    mission=None,
    terrain=None,
    architectures: Optional[Sequence[Optional[str]]] = None,
    aero_sources: Sequence[str] = ("analytical", "aerosandbox"),
    verbose: bool = True,
) -> List[Dict]:
    """
    Size the vehicle for every (architecture, solver) pair.

    ``architectures`` may contain ``None``, which means "do not pin" — the
    relaxed run where ``z_arch`` stays in the design vector and the optimizer
    picks. That row is labelled ``relaxed`` and reports which architecture it
    converged onto.
    """
    from hpraptor_mdao import ARCH_NAMES, build_problem, run_optimization

    if architectures is None:
        architectures = list(ARCH_NAMES) + [None]

    rows: List[Dict] = []
    total = len(architectures) * len(aero_sources)
    n = 0

    for source in aero_sources:
        for arch in architectures:
            n += 1
            label = arch or "relaxed"
            if verbose:
                print(f"  [{n:>2}/{total}] {label:<22} {source}", end="", flush=True)

            prob = build_problem(
                mission=mission, terrain=terrain, fixed_architecture=arch,
                geometry_source=source, aero_source=source,
            )
            prob.setup()
            t0 = time.perf_counter()
            try:
                result = run_optimization(prob, verbose=False)
            except Exception as exc:                       # pragma: no cover
                # One failing architecture must not lose the other thirteen
                # rows -- the table is the deliverable, and a gap in it is
                # more informative than an aborted run.
                if verbose:
                    print(f"   FAILED  {type(exc).__name__}: {exc}")
                rows.append({"architecture": label, "aero": source,
                             "failed": f"{type(exc).__name__}: {exc}"})
                continue
            wall = time.perf_counter() - t0

            result["architecture"] = label
            result["pinned"] = arch is not None
            result["aero"] = source
            result["wall_s"] = wall
            rows.append(result)

            if verbose:
                status = _dig(result, "solver.exit_status") or "?"
                print(f"   {result['energy_primary_wh']:8.2f} Wh   "
                      f"{_dig(result, 'solver.nit') or -1:>3} it   "
                      f"{wall:5.1f} s   {status}")
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# Tables
# ═══════════════════════════════════════════════════════════════════════════

def format_campaign_table(rows: List[Dict]) -> str:
    """The main results table, one line per (architecture, solver)."""
    header = "  ".join(f"{h:{w}}" for _, h, _, w in COLUMNS)
    out = [header, "-" * len(header)]

    for row in rows:
        if "failed" in row:
            out.append(f"{row['architecture']:<22}  {row['aero']:<12}  "
                       f"FAILED: {row['failed']}")
            continue
        cells = []
        for key, _, fmt, width in COLUMNS:
            val = _dig(row, key)
            if val is None:
                cells.append(f"{'-':{width}}")
            elif fmt.endswith("s}"):
                cells.append(f"{str(val):{width}}")
            else:
                try:
                    cells.append(f"{fmt.format(val):{width}}")
                except (ValueError, TypeError):
                    cells.append(f"{str(val):{width}}")
        out.append("  ".join(cells))
    return "\n".join(out)


def format_constraint_table(rows: List[Dict]) -> str:
    """Constraint activity, so an active set can be read off at a glance."""
    heads = [f"{'architecture':<22}", f"{'aero':<12}"]
    heads += [f"{lbl:>11}" for _, lbl in CONSTRAINTS]
    header = "  ".join(heads)
    out = [header, "-" * len(header)]

    for row in rows:
        if "failed" in row:
            continue
        cells = [f"{row['architecture']:<22}", f"{row['aero']:<12}"]
        for key, _ in CONSTRAINTS:
            val = row.get(key)
            if val is None:
                cells.append(f"{'-':>11}")
            else:
                # A trailing marker beats colour here: this table is read in
                # a terminal, pasted into a log, and diffed.
                mark = "!" if val > 1e-6 else ("*" if abs(val) < 1e-6 else " ")
                cells.append(f"{val:>10.4f}{mark}")
        out.append("  ".join(cells))
    out.append("")
    out.append("  * = active (|g| < 1e-6)      ! = VIOLATED (g > 0)")
    return "\n".join(out)


def format_best(rows: List[Dict]) -> str:
    """Which architecture won, per solver path."""
    out = []
    for source in sorted({r["aero"] for r in rows if "failed" not in r}):
        pinned = [r for r in rows
                  if r.get("aero") == source and r.get("pinned")
                  and "failed" not in r]
        if not pinned:
            continue
        best = min(pinned, key=lambda r: r["energy_primary_wh"])
        relaxed = next((r for r in rows if r.get("aero") == source
                        and not r.get("pinned", True) and "failed" not in r), None)
        out.append(f"  {source}:")
        out.append(f"    best pinned    {best['architecture']:<20} "
                   f"{best['energy_primary_wh']:.2f} Wh")
        if relaxed is not None:
            agree = relaxed.get("dominant_architecture") == best["architecture"]
            out.append(f"    relaxed chose  "
                       f"{relaxed.get('dominant_architecture','?'):<20} "
                       f"{relaxed['energy_primary_wh']:.2f} Wh"
                       f"   {'(agrees)' if agree else '(DISAGREES)'}")
    return "\n".join(out)


def campaign_report(rows: List[Dict]) -> str:
    """Everything, as one block of text."""
    W = 118
    return "\n".join([
        "=" * W,
        "ARCHITECTURE CAMPAIGN — sizing MDO, every architecture, both solver paths",
        "=" * W, "",
        format_campaign_table(rows), "",
        "-" * W,
        "CONSTRAINT ACTIVITY",
        "-" * W, "",
        format_constraint_table(rows), "",
        "-" * W,
        "SELECTION",
        "-" * W, "",
        format_best(rows), "",
        "=" * W,
    ])


def best_row(rows):
    """
    The best FEASIBLE design in the campaign, highest-fidelity path first.

    Feasibility is checked here rather than trusted: an optimizer that exits
    FAIL still returns a design vector, and at long range the all-electric
    runs return one that violates its own energy balance. Picking the
    minimum objective without checking would hand the vehicle renderer an
    aircraft that cannot fly the mission.
    """
    gates = ("g1_cl_margin", "g2_energy_margin", "g3_soc_margin",
             "g4_stress_margin", "g6_battery_power", "g7_rotor_fit",
             "g10_fuel_energy")
    for source in ("aerosandbox", "analytical"):
        pool = [r for r in rows
                if r.get("aero") == source and "failed" not in r
                and r.get("success")
                and all(r.get(g, -1.0) <= 1e-4 for g in gates)]
        if pool:
            return min(pool, key=lambda r: r["energy_primary_wh"])
    return None


def save_campaign(rows: List[Dict], stem: str = "campaign") -> Dict[str, str]:
    """Write the raw rows, the rendered report, and the winning design."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    js = RESULTS / (stem + ".json")
    txt = RESULTS / (stem + "_table.txt")
    with open(js, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, default=float)
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write(campaign_report(rows) + chr(10))

    paths = {"json": str(js), "table": str(txt)}

    # The winner, on its own, so the vehicle renderer has a single design to
    # draw without re-deriving which row won.
    best = best_row(rows)
    if best is not None:
        bp = RESULTS / (stem + "_best.json")
        with open(bp, "w", encoding="utf-8") as fh:
            json.dump(best, fh, indent=2, default=float)
        paths["best"] = str(bp)
    return paths
