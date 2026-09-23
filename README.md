# RAPTOR HybridMDAO

**Terrain-aware multidisciplinary design optimization for hybrid-electric VTOL UAVs.**

Given two coordinates and a payload, the framework sizes the lightest hybrid VTOL that can
actually fly that route over the real terrain — and lets the optimizer *choose the propulsion
architecture* rather than comparing six hand-built designs.

Architecture selection is posed as a continuous design variable (a softmax relaxation over six
powertrains, with a discreteness penalty that forces a buildable one-hot answer) solved in the
same gradient-based problem as the wing, the structure and the battery.

**[FRAMEWORK.md](FRAMEWORK.md)** is the full write-up: the model and what has been verified
about it, an orientation to the code, and an honest list of what is known to be wrong. Read
Part III before quoting any number from this repository.

---

## Install

```bash
pip install -e ".[all]"
setx OPENTOPOGRAPHY_API_KEY <your-free-key>     # Windows; export on POSIX
```

The key is optional but strongly recommended. Without it the DEM builder silently falls back
from NASADEM (~31 m posting) to keyless SRTM (~111 m) **and overwrites your cached DEM**. It
prints which source it chose on every run.

---

## Run it

```bash
python -m hpraptor_mdao.run all          # everything: campaign, diagrams, figures  (~6 min)
```

Or the pieces:

```bash
python -m hpraptor_mdao.run campaign             # every architecture x both fidelity paths
python -m hpraptor_mdao.run campaign --quick     # analytical only, ~2 s
python -m hpraptor_mdao.run sizing --save mydesign
python -m hpraptor_mdao.run sizing --arch turbo_electric
python -m hpraptor_mdao.run sizing --geometry-source aerosandbox --aero-source aerosandbox
python -m hpraptor_mdao.run mission              # five-phase dymos trajectory
python -m hpraptor_mdao.run coupled              # sizing + trajectory in one loop
python -m hpraptor_mdao.run xdsm                 # model diagrams
python -m hpraptor_mdao.run profiles             # mission-profile comparison, 2D + 3D
python -m hpraptor_mdao.run sizing --history     # iteration trace, terminal + figure
python -m hpraptor_mdao.run sizing --dry-run     # print the formulation, solve nothing
```

Visualization:

```bash
python -m hpraptor.postprocessing.trajectory_3d --views all --interactive
python -m hpraptor.postprocessing.aircraft_3d --result results/campaign_best.json --views all
python -m hpraptor.postprocessing.serve          # serve reports/ on localhost
```

The interactive 3D pages must be opened over `http://localhost`, not by double-clicking the
file: Chromium browsers give `file://` pages an opaque origin, which blocks what vtk.js needs.
`hpraptor-serve` exists for exactly that.

### A different route

Change **two coordinate pairs and one cache path** in a copy of `configs/quito_mission.yaml`,
then:

```bash
python -m hpraptor.m1_mission.srtm_downloader --mission configs/mine.yaml \
    --source auto --resolution native --output data/dem/mine.npz
python -m hpraptor_mdao.run all --mission configs/mine.yaml
```

The corridor bounding box is derived from the endpoints; `altitude_amsl` is only a fallback,
since pad elevations are anchored to the DEM itself.

---

## The optimization problem

**Minimize** primary energy — battery energy divided by charging efficiency plus the fuel's full
chemical content, so every architecture is charged for its own losses at the same boundary —
plus a discreteness penalty on the architecture weights.

**9 design variables** (14 scalars): wing loading, aspect ratio, taper ratio, spar thickness,
disk loading, battery mass, fuel mass, electric fraction, and the 6-vector `z_arch`. Cruise
altitude joins them when terrain is loaded.

**9 constraints**, each added in response to a specific way the optimizer was caught cheating:

| | Enforces | Stopped |
|---|---|---|
| g₁ | stall margin | shrinking the wing past reachable lift |
| g₂ | battery covers its share | shrinking the aircraft to nothing |
| g₃ | SOC reserve at touchdown | landing with a flat pack |
| g₄ | spar below yield | unbounded aspect ratio |
| g₅ | cruise clears the ridge | flying through a mountain |
| g₆ | pack C-rate | a pack unable to lift the aircraft |
| g₇ | rotors fit the span | overlapping rotors |
| g₈ | Reynolds ≥ 2×10⁵ | a wing the airfoil data no longer covers |
| g₁₀ | fuel carried supplies its share | burning fuel that existed in no tank |

---

## Package layout

```
hpraptor/                 physics library
├── core/                 config, mission loader, atmosphere, flight path/segments
├── m1_mission/           NASADEM via OpenTopography, terrain surrogate, path building
├── m2_geometry/          wing planform, fuselage, tail, rotors; AeroSandbox assembly
├── m3_structures/        spar sizing, mass buildup, stability
├── m4_aero/              parasite drag buildup, AeroSandbox/VLM interface
├── m5_propulsion/        motors, ICE, fuel cell, battery catalogue, architecture blending
├── m6_dynamics/          differentiable 3-DoF equations of motion
└── postprocessing/       2D dashboards, 3D corridor + vehicle renderers, local server

hpraptor_mdao/            the OpenMDAO layer
├── components/           each discipline as an ExplicitComponent
├── trajectory/           dymos phases and ODEs
├── groups.py             the coupled sizing group (NLBGS on the mass loop)
├── problem.py            design variables, constraints, objective, driver
├── coupled.py            sizing + trajectory in one problem
├── campaign.py           every architecture x both fidelity paths, tabulated
├── xdsm.py               conceptual and introspected model diagrams
└── run.py                CLI entry point
```

Two interchangeable fidelity paths run through m2/m4: an analytical buildup (milliseconds) and
AeroSandbox's real 3D geometry and aerodynamics (tens of seconds). They agree on the
architecture ranking and disagree on the numbers, which is the useful property — explore with
the cheap one, publish with the expensive one.

---

## Outputs

| Directory | Contents |
|---|---|
| `results/` | campaign JSON, the rendered results table, the winning design |
| `reports/` | XDSM diagrams, variable inventory, 3D plates and interactive pages |
| `figures/` | per-architecture dashboards and the comparison sweep |
| `data/dem/` | cached DEM (gitignored — one API call rebuilds it) |

---

## Status

Working: the sizing MDO converges to KKT on all 14 campaign runs; NASADEM ingestion for any
global endpoints; the architecture relaxation recovers the discrete winner; 3D corridor and
vehicle visualization; 299 tests.

Open: see [FRAMEWORK.md Part III](FRAMEWORK.md#part-iii--what-is-known-to-be-wrong). The two
that most affect a quoted number: the design mission is the **13.58 km corridor**, not the
50 km `design_range_km` the YAML states (terrain supersedes it), and the `all_electric` rows
are path-dependent because `k_electric` and `m_fuel` are inert for that architecture. The
coupled sizing-plus-trajectory problem still exits on iteration limit rather than reaching
KKT.

## License

MIT — see LICENSE.
