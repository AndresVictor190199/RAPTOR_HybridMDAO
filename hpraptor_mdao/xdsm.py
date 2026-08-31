"""
XDSM diagrams of the optimization problem.

Two complementary views, because they answer different questions:

**pyXDSM** (``write_conceptual_xdsm``) draws the diagram by hand, from a
description written here. It shows the problem the way a paper would: which
disciplines exist, what each hands the next, and where the loops close. It
is readable, it is publication quality, and it is exactly as correct as the
description below — so the description is kept beside the code it describes.

**OpenMDAO-XDSM** (``write_model_xdsm``) introspects a live ``om.Problem``
and draws what is *actually wired*, variable by variable. Nothing is
asserted by hand, so it cannot flatter the implementation: a connection
that was never made does not appear, and a variable nobody consumes shows
up as a dead end. This is the one to trust when checking the model.

Read them together. The conceptual diagram tells you what the framework is
meant to do; the automatic one tells you what it does.

Usage
-----
    python -m hpraptor_mdao.run xdsm              # both, for the coupled problem
    python -m hpraptor_mdao.run xdsm --study sizing
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

REPORTS = Path("reports")

#: MiKTeX ships poppler's pdftocairo but does not put it on PATH.
_PDFTOCAIRO_CANDIDATES = [
    "pdftocairo",
    r"C:\Users\vic_a\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdftocairo.exe",
]


def _find_pdftocairo() -> Optional[str]:
    for cand in _PDFTOCAIRO_CANDIDATES:
        found = shutil.which(cand) if os.sep not in cand else (
            cand if Path(cand).exists() else None)
        if found:
            return found
    return None


def pdf_to_png(pdf_path: Path, dpi: int = 200) -> Optional[Path]:
    """
    Rasterise a pyXDSM PDF so the diagram can be embedded in a report.

    pyXDSM renders through LaTeX, which only produces PDF. Returns None when
    no converter is available rather than raising: the PDF is still the real
    deliverable, and losing the PNG should not fail a diagram build.
    """
    tool = _find_pdftocairo()
    if tool is None or not pdf_path.exists():
        return None
    out_stem = pdf_path.with_suffix("")
    subprocess.run([tool, "-png", "-r", str(dpi), "-singlefile",
                    str(pdf_path), str(out_stem)],
                   check=False, capture_output=True)
    png = out_stem.with_suffix(".png")
    return png if png.exists() else None


# ═══════════════════════════════════════════════════════════════════════════
# 1. CONCEPTUAL DIAGRAM  (pyXDSM, hand-described)
# ═══════════════════════════════════════════════════════════════════════════

def write_conceptual_xdsm(name: str = "xdsm_conceptual",
                          outdir: Path = REPORTS) -> Path:
    """
    The coupled problem as a paper would present it.

    Follows the standard XDSM conventions: the optimizer on the diagonal
    drives everything, grey parallelograms are data, and the thin line is
    the process order while the thick one is data flow.
    """
    from pyxdsm.XDSM import XDSM, OPT, SOLVER, FUNC, LEFT

    outdir.mkdir(parents=True, exist_ok=True)
    x = XDSM(use_sfmath=True)

    # -- Systems ---------------------------------------------------------
    # m1 sits at step 0, outside the optimization loop, because the mission
    # is data the optimizer is solved *against*, not something it changes:
    # the DEM is fetched, the corridor is cut and the terrain surrogate is
    # fitted once, and every iteration afterwards reads the same numbers.
    x.add_system("mis", FUNC, r"\text{0: m1 Mission (NASADEM)}")
    x.add_system("opt", OPT, r"\text{1, 12} \to \text{2: Optimizer (SLSQP)}")
    x.add_system("mda", SOLVER, r"\text{2, 9} \to \text{3: MDA (NLBGS)}")
    x.add_system("geo", FUNC, r"\text{3: m2 Geometry}")
    x.add_system("str", FUNC, r"\text{4: m3 Structures}")
    x.add_system("aero", FUNC, r"\text{5: m4 Aerodynamics}")
    x.add_system("prop", FUNC, r"\text{6: m5 Propulsion}")
    x.add_system("arch", FUNC, r"\text{7: m5 Architecture}")
    x.add_system("mass", FUNC, r"\text{8: Energy \& mass closure}")
    x.add_system("traj", SOLVER, r"\text{10: m7 Trajectory (dymos)}")
    x.add_system("clos", FUNC, r"\text{11: Coupling closure}")

    # -- What the mission hands the rest of the problem -------------------
    x.connect("mis", "traj", r"\text{terrain}(x),\ h_0,\ h_f")
    x.connect("mis", "mass", r"R_{route},\ t_{hover},\ m_{pay}")
    x.connect("mis", "opt", r"h^{lo},\ h^{hi}")

    # -- Design variables leaving the optimizer --------------------------
    x.connect("opt", "geo", r"W/S,\ AR,\ W/A")
    x.connect("opt", "str", r"t_{spar}")
    x.connect("opt", "aero", r"AR,\ h_{cruise}")
    x.connect("opt", "prop", r"h_{cruise}")
    x.connect("opt", "arch", r"z_{arch},\ k_{elec}")
    x.connect("opt", "mass", r"m_{bat},\ m_{fuel}")
    x.connect("opt", "traj", r"z_{arch},\ \text{node controls}")

    # -- Discipline-to-discipline data -----------------------------------
    x.connect("geo", "str", r"S_{ref},\ b")
    x.connect("geo", "aero", r"S_{ref},\ S_{wet},\ \bar{c}")
    x.connect("geo", "prop", r"A_{rotor}")
    x.connect("aero", "prop", r"D_{cruise}")
    x.connect("prop", "arch", r"P_{cruise},\ P_{motor}")
    x.connect("str", "mass", r"m_{empty}")
    x.connect("arch", "mass", r"m_{prop},\ \eta_{arch}")

    # -- The MDA feedback edge: MTOW -------------------------------------
    # This is the only true cycle in the sizing loop, and it is why an MDA
    # is needed at all: mass sets the geometry that sets the mass.
    x.connect("mass", "mda", r"m_{TOW}")
    x.connect("mda", "geo", r"m_{TOW}")
    x.connect("mda", "str", r"m_{TOW}")
    x.connect("mda", "aero", r"m_{TOW}")
    x.connect("mda", "prop", r"m_{TOW}")

    # -- Converged sizing feeds the trajectory ---------------------------
    x.connect("geo", "traj", r"S_{ref},\ A_{rotor}")
    x.connect("aero", "traj", r"C_{D0},\ e")
    x.connect("arch", "traj", r"w_{arch}")
    x.connect("mass", "traj", r"m_{TOW},\ E_{batt}")

    # -- Trajectory feeds the closure, closure feeds the optimizer -------
    x.connect("traj", "clos", r"SOC_f,\ m_f,\ E")
    x.connect("mass", "clos", r"m_{TOW}")
    x.connect("clos", "opt", r"f,\ g_3,\ g_9,\ h_{mass}")
    x.connect("str", "opt", r"g_4")
    x.connect("aero", "opt", r"g_1,\ g_8")
    x.connect("geo", "opt", r"g_7")
    x.connect("mass", "opt", r"g_5,\ g_6")
    x.connect("traj", "opt", r"g_{AGL},\ g_{C_L}")

    # -- External inputs and outputs -------------------------------------
    x.add_input("mis", r"\text{lat/lon endpoints, DEM}")
    x.add_input("geo", r"n_{rotors}")
    x.add_input("aero", r"\text{airfoil, } Re_{min}")
    x.add_input("prop", r"\eta_{motor},\ \eta_{prop}")
    x.add_input("arch", r"\text{6 architectures}")
    x.add_input("mass", r"\text{cell spec}")
    x.add_input("traj", r"\text{wind}")
    x.add_input("opt", r"x^{(0)}")

    x.add_output("opt", r"x^*", side=LEFT)
    x.add_output("clos", r"E^*_{primary}", side=LEFT)
    x.add_output("traj", r"\text{flight path}", side=LEFT)
    x.add_output("mis", r"\text{corridor profile}", side=LEFT)

    x.add_process(
        ["mis", "opt", "mda", "geo", "str", "aero", "prop", "arch", "mass",
         "mda", "traj", "clos", "opt"],
        arrow=True,
    )

    # pyXDSM writes an \input{"<path>.tikz"} line verbatim into the .tex. On
    # Windows that path carries backslashes, which LaTeX reads as undefined
    # control sequences, and the build dies. Running from inside the output
    # directory leaves a bare filename with no separator at all.
    cwd = Path.cwd()
    try:
        os.chdir(outdir)
        x.write(name, build=True, cleanup=True, quiet=True)
    finally:
        os.chdir(cwd)
    return outdir / f"{name}.pdf"


# ═══════════════════════════════════════════════════════════════════════════
# 2. AUTOMATIC DIAGRAM  (OpenMDAO-XDSM, introspected)
# ═══════════════════════════════════════════════════════════════════════════

#: xdsmjs' own pseudo-node for everything outside the model. It is never
#: listed in ``nodes``, so it must not be treated as a dangling reference.
_XDSMJS_EXTERNAL = "_U_"


def _repair_xdsm_html(path: Path) -> int:
    """
    Drop edges whose endpoints are not in the node list, and report how many.

    xdsmjs looks every edge endpoint up in the node table while it lays the
    diagram out. A reference to a node that is not there throws, the
    ``DOMContentLoaded`` handler dies, and the page renders **completely
    blank** — no error visible, just white. That failure is silent and
    indistinguishable from a broken file, so it is worth guarding against
    rather than trusting the writer to be consistent.

    omxdsm produces exactly this when ``include_indepvarcomps=False``: it
    removes the IndepVarComp nodes, then only rewires the ones that appear as
    a *source* in the pruned connection table, leaving each design-variable
    edge from the driver pointing at a node it just deleted. Keeping the IVCs
    avoids that path entirely, and this pass is the backstop for anything
    else that slips through.

    Returns the number of edges removed.
    """
    import html as _html
    import json

    raw = path.read_text(encoding="utf-8")
    match = re.search(r'data-mdo="(.*?)"></div>', raw, re.S)
    if match is None:
        return 0

    data = json.loads(_html.unescape(match.group(1)))
    known = {n["id"] for n in data.get("nodes", [])} | {_XDSMJS_EXTERNAL}
    edges = data.get("edges", [])
    kept = [e for e in edges if e["from"] in known and e["to"] in known]
    removed = len(edges) - len(kept)
    if removed == 0:
        return 0

    data["edges"] = kept
    patched = _html.escape(json.dumps(data), quote=True)
    path.write_text(raw[:match.start(1)] + patched + raw[match.end(1):],
                    encoding="utf-8")
    return removed


def write_model_xdsm(problem, name: str = "xdsm_model",
                     outdir: Path = REPORTS,
                     out_format: str = "html",
                     recurse: bool = True,
                     include_indepvarcomps: bool = True):
    """
    Draw what the model is actually wired as, from the live Problem.

    Nothing here is asserted by hand — omxdsm walks the connection graph, so
    a discipline that was never connected simply will not have an arrow, and
    an output nobody consumes shows as a dead end. That is precisely what
    makes it worth generating alongside the conceptual one.

    ``out_format='html'`` produces an interactive xdsmjs page (no LaTeX);
    ``'pdf'`` goes through pyXDSM and needs a working LaTeX toolchain.
    """
    from omxdsm import write_xdsm

    outdir.mkdir(parents=True, exist_ok=True)
    problem.final_setup()

    write_xdsm(
        problem,
        filename=str(outdir / name),
        out_format=out_format,
        show_browser=False,
        recurse=recurse,
        include_indepvarcomps=include_indepvarcomps,
        include_solver=True,
        show_parallel=True,
        quiet=True,
    )
    suffix = {"html": ".html", "pdf": ".pdf", "tex": ".tex"}.get(out_format, "")
    out = outdir / f"{name}{suffix}"

    if out_format == "html" and out.exists():
        dropped = _repair_xdsm_html(out)
        if dropped:
            print(f"    [xdsm] dropped {dropped} dangling edge(s) from "
                  f"{out.name} so the page renders")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 3. VARIABLE INVENTORY  (what the diagrams cannot show in a box)
# ═══════════════════════════════════════════════════════════════════════════

def variable_inventory(problem) -> str:
    """
    Every variable the model computes, grouped by the discipline that owns it.

    The XDSM boxes name the disciplines and the arrows name the couplings,
    but neither can carry sixty variables legibly. This is the companion
    listing, read from the live model so it cannot drift.
    """
    import numpy as np

    problem.final_setup()
    lines = []
    seen = set()

    def walk(system, depth=0):
        name = system.pathname or "<model>"
        try:
            outs = system.list_outputs(val=True, units=True, desc=True,
                                       prom_name=True, out_stream=None)
        except Exception:
            return
        own = [(n, m) for n, m in outs
               if n.count(".") == name.count(".") + (1 if name != "<model>" else 0)]
        if own:
            lines.append(f"\n{'  ' * depth}{name}  ({len(own)} outputs)")
            for abs_name, meta in own:
                prom = meta.get("prom_name", abs_name)
                if prom in seen:
                    continue
                seen.add(prom)
                val = meta["val"]
                shown = (f"{float(np.ravel(val)[0]):.4g}" if np.size(val) == 1
                         else f"array({np.size(val)})")
                units = meta.get("units") or "-"
                desc = (meta.get("desc") or "").split(".")[0][:52]
                lines.append(f"{'  ' * depth}    {prom:<26s} {shown:>12s} "
                             f"{units:<8s} {desc}")
        for sub in getattr(system, "_subsystems_myproc", []):
            walk(sub, depth + 1)

    walk(problem.model)
    return "\n".join(lines)
