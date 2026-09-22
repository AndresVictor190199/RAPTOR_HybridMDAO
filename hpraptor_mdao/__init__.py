"""
hpraptor_mdao — OpenMDAO Multidisciplinary Design Optimization
================================================================

Gradient-based MDO over the hpraptor discipline modules, with the
continuous propulsion-architecture relaxation as a first-class design
variable alongside the geometric and structural ones.

Model structure (one component per discipline)
----------------------------------------------
    GeometryComp        m2_geometry     wing / fuselage / rotor / tail
    StructuresComp      m3_structures   spar sizing, g4 stress margin
    AeroComp            m4_aero         parasite drag buildup, drag polar
    PowerRequiredComp   m5_propulsion   hover / cruise / climb power
    ArchitectureComp    m5_propulsion   continuous architecture relaxation
    EnergyComp                          battery + fuel sizing, mission energy
    WeightsComp                         mass closure
    ObjectiveComp                       energy + discreteness penalty

The `m_tow` feedback edge makes this a genuinely coupled MDA, converged
by NonlinearBlockGS; derivatives across the converged loop come from
OpenMDAO's unified derivatives equation, with per-component partials by
complex step.

What changed from the previous version of this package
-------------------------------------------------------
It used to contain five standalone components with their own toy physics
(flat efficiencies, a hardcoded C_D0), disconnected from m2-m5 entirely,
and its "optimizer" never ran: design variables were declared as
IndepVarComp outputs, `add_design_var`/`add_objective`/`add_constraint`
were never called, and the runner invoked `run_model()` rather than
`run_driver()`. Every component here wraps the real hpraptor module
instead, and the driver is actually configured and run.

Usage
-----
    from hpraptor_mdao import build_problem, run_optimization

    prob = build_problem(payload_kg=5.0, altitude=2900.0)
    result = run_optimization(prob)
    print(result["dominant_architecture"], result["arch_weights"])

    # Pinned-architecture baseline, for the discrete comparison:
    prob = build_problem(fixed_architecture="series")

Author: Victor Berrazueta (LUAS-EPN)
"""

from __future__ import annotations
# ── OpenMDAO's automatic reports, off by default ─────────────────────────
# Every Problem OpenMDAO builds writes an N2 + scaling + optimizer report
# into a `<script>_out/` directory beside the working directory. Across a
# session of CLI runs and test invocations that silently accumulated 125 of
# them at the repo root. They are regenerable on demand (`--n2`, or the
# `xdsm` study), so the default is off; set OPENMDAO_REPORTS=1 to restore
# them for a single run.
import os as _os

_os.environ.setdefault("OPENMDAO_REPORTS", "0")

# OPENMDAO_REPORTS only governs the HTML reports. A Problem also creates its
# output directory for *coloring* files, which declare_coloring writes on
# every run -- so `__main__NN_out/` folders kept appearing at the repo root
# even with reports off. Those files are worth keeping (they cache the
# Jacobian sparsity and make reruns markedly faster), they just need one
# home instead of one folder per invocation. OPENMDAO_WORKDIR gives them
# that; the directory is gitignored.
_os.environ.setdefault("OPENMDAO_WORKDIR", ".openmdao")

try:
    import openmdao.api as om  # noqa: F401
    HAS_OPENMDAO = True
except ImportError:  # pragma: no cover
    HAS_OPENMDAO = False

if HAS_OPENMDAO:
    from .components import (
        GeometryComp, StructuresComp, AeroComp,
        PowerRequiredComp, ArchitectureComp,
        EnergyComp, WeightsComp, ObjectiveComp,
    )
    from .groups import HybridVTOLGroup
    from .problem import build_problem, run_optimization, ARCH_NAMES

    __all__ = [
        "HAS_OPENMDAO",
        "GeometryComp", "StructuresComp", "AeroComp",
        "PowerRequiredComp", "ArchitectureComp",
        "EnergyComp", "WeightsComp", "ObjectiveComp",
        "HybridVTOLGroup",
        "build_problem", "run_optimization", "ARCH_NAMES",
    ]
else:  # pragma: no cover
    __all__ = ["HAS_OPENMDAO"]
