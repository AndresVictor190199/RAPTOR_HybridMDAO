# RAPTOR HybridMDAO — framework summary

**Terrain-aware multidisciplinary design optimization for hybrid-electric VTOL UAVs.**

Given two coordinates and a payload, the framework sizes the lightest hybrid VTOL that can
fly that route over real terrain, and lets the optimizer *choose the propulsion architecture*
rather than comparing hand-built designs. Architecture selection is posed as a continuous
design variable — a softmax relaxation over six powertrains with a discreteness penalty that
forces a buildable one-hot answer — solved in the same gradient-based problem as the wing, the
structure and the battery.

This document has two halves. **Part I** is the model and what has been verified about it,
written for a reviewer or co-author. **Part II** is the code, written for someone about to
work in it. **Part III** lists what is known to be wrong or unfinished, because a summary that
omits that is not a summary.

---

# Part I — The model

## 1. The optimization problem

**Minimize** primary energy — battery energy divided by charging efficiency, plus the fuel's
full chemical content — so every architecture is charged for its own losses at the same
boundary. Plus a discreteness penalty `λ·Σ wᵢ(1−wᵢ)` that vanishes only at a one-hot
architecture.

Comparing architectures on *shaft* energy would not be neutral: a hybrid pays its conversion
losses inside that number while a battery aircraft's stored energy is already electrical.
Primary energy is the like-for-like boundary.

**Design variables — 9 (14 scalars):** wing loading, aspect ratio, taper ratio, spar thickness,
disk loading, battery mass, fuel mass, electric power fraction, and the 6-vector `z_arch`.
Cruise altitude joins them when terrain is loaded.

**Constraints — 9.** Each was added in response to a specific way the optimizer was caught
exploiting the model:

| | Enforces | Stopped |
|---|---|---|
| g₁ | stall margin | shrinking the wing past reachable lift |
| g₂ | battery covers its own share | shrinking the aircraft to nothing |
| g₃ | SOC reserve at touchdown | landing with a flat pack |
| g₄ | spar below yield | unbounded aspect ratio |
| g₅ | cruise clears the ridge | flying through a mountain |
| g₆ | pack C-rate | a pack unable to lift the aircraft |
| g₇ | rotors fit the span | overlapping rotors |
| g₈ | Reynolds ≥ 2×10⁵ | a wing the airfoil data no longer covers |
| g₁₀ | fuel carried supplies its share | burning fuel that existed in no tank |

Bounds are stated as *admissibility* limits, not tuning knobs. A bound that binds at the
optimum is the model declining to answer the question, so where a previous bound was simply
where the optimizer stopped, it was widened and a physical constraint added in its place.

Driver: SLSQP via `ScipyOptimizeDriver`, reverse mode, per-component complex-step partials
assembled through OpenMDAO's unified derivatives equation.

## 2. The discipline chain

```
m1 mission     NASADEM via OpenTopography → terrain surrogate → corridor geometry
m2 geometry    wing planform, fuselage, tail, rotors
m3 structures  spar sizing (CFRP), mass buildup, stability
m4 aero        parasite-drag buildup or AeroSandbox VLM/AeroBuildup
m5 propulsion  motors, ICE, fuel cell, gas turbine, battery catalogue, architecture blending
m6 dynamics    differentiable 3-DoF equations of motion (dymos ODEs)
```

The mass loop (`m_tow` feeding back into m2/m3) is closed by a `NonlinearBlockGS` solver inside
the sizing group; that feedback edge is the only cycle in the model.

## 3. The architecture relaxation — the contribution

`ArchitectureComp` turns a 6-vector into softmax weights over
`{all_electric, series, parallel, series_parallel, turbo_electric, fuel_cell}` and blends each
architecture's propulsion mass, fuel flow, bus power, fuel LHV and **fuel-path efficiency**.
Everything downstream sees one blended vehicle, so the discrete choice becomes a gradient-based
one.

Two details make it work rather than merely typecheck:

- **`fuel_capable`** gates the fuel path. Without it the optimizer sized an "optimal"
  all-electric aircraft that drew 49% of its energy from gasoline it had no way to burn.
- **`eta_fuel_cruise`** carries each architecture's real conversion efficiency, evaluated from
  the same Willans line / turbine lapse / polarization curve the trajectory ODEs use. It is
  blended by `wᵢ·LHVᵢ` rather than `wᵢ`, which makes
  `blend_fuel_lhv × blend_fuel_efficiency` equal `Σ wᵢ·LHVᵢ·ηᵢ` at *every* weight vector — not
  only at the one-hot vertices the penalty eventually reaches, but everywhere the optimizer
  actually searches.

## 4. Two fidelity paths

m2 and m4 each have an analytical implementation (milliseconds) and an AeroSandbox
implementation (tens of seconds) selected by `--geometry-source` / `--aero-source`. They agree
on the architecture ranking and disagree on the numbers, which is the useful property: explore
with the cheap one, publish with the expensive one.

## 5. What is verified

| Check | Result |
|---|---|
| Test suite | **299 passed** |
| Total derivatives vs central FD | worst relative error **~3×10⁻⁶** (FD truncation, not model error) |
| Unconnected inputs in the sizing model | **0** |
| Campaign reproducibility | identical to printed digits across repeated runs |
| CasADi ↔ numpy architecture parity | asserted per-architecture in `tests/test_m5_architecture_np.py` |

## 6. Results — the Quito → Cumbayá corridor

13.58 km, 610 m net descent, crossing a 3026 m ridge. NASADEM 1 arc-second, no synthetic
terrain anywhere in the framework.

Primary energy, all 14 runs converging to a KKT exit:

| architecture | analytical | aerosandbox | η fuel |
|---|---|---|---|
| **all_electric** | **57.64** | **54.68** | — |
| series | 63.75 | 60.42 | 0.094 |
| parallel | 61.60 | 58.55 | 0.096 |
| series_parallel | 63.75 | 60.42 | 0.097 |
| turbo_electric | 60.78 | 57.85 | 0.033 |
| fuel_cell | 68.98 | 66.54 | 0.160 |
| relaxed | 57.15 | 54.71 | → all_electric |

The relaxation recovers the best discrete architecture on both fidelity paths.

**Read this table carefully.** At 13.58 km every architecture converges to `m_fuel = 0`: the
hybrids decline to use their own fuel path, so the ranking is driven entirely by powerplant
mass, not by energy conversion. The corridor is far inside all-electric territory and cannot
discriminate architectures on efficiency.

Where it *does* discriminate — a range sweep on the analytical path, no terrain:

| range | best |
|---|---|
| 50–350 km | all_electric |
| 450 km | fuel_cell (3584 Wh) |
| 600 km | fuel_cell (5709 Wh) |

All-electric stays feasible to ~350 km and is infeasible at 450 km on the 8 kg battery bound.
Beyond the crossover the fuel cell wins; turbo-electric, once charged its real ~0.047 part-load
efficiency, is the worst option by roughly 5×.

## 7. The mission profile: three sources that do not agree

The framework constructs a vertical profile in three separate places, and they are **not** one
chain:

1. **`m1_mission.builder.PathBuilder`** builds three candidate profiles from the DEM
   (HIGH_OVERFLY, TERRAIN_FOLLOW, MINIMAL_ENERGY). It is used by the legacy `run_mission.py`
   pipeline, the tests, and the 3D viewer — **never by the MDAO**.
2. **The sizing MDO** reduces the whole corridor to four scalars (range, pad elevation, highest
   terrain, required clearance). The "profile" is one cruise altitude, held above the route's
   single highest point by g₅. On this corridor that constraint is active, so the altitude is
   set by terrain, not traded.
3. **The dymos trajectory** re-derives a profile over five phases against a 60-Gaussian
   differentiable terrain surrogate (lifted 15 m so it never under-predicts the ridge), with
   AGL as a path constraint at every node.

Only (3) is an optimized trajectory in the usual sense. The lateral track is never optimized
anywhere — it is a straight line in latitude/longitude from origin to destination.

**The two optimizers disagree about the architecture.** `run mission` converges (SLSQP exit
mode 0, 539 iterations, ~1.5 h wall) to a one-hot **`series_parallel`**, while the sizing MDO
converges to `all_electric`. The disagreement is not physical: the trajectory burns 0.46 g of
fuel over the whole mission, so every architecture behaves almost identically in it and the
choice is decided by the discreteness penalty pushing off a symmetric start rather than by
energy. The framework's own dry-run note says this outright — the penalty has exactly zero
gradient at `z_arch = 0`, so which vertex a relaxed run falls into is floating-point
asymmetry and `--multistart` is required to break the tie. **Do not read the trajectory's
architecture as a second opinion on the sizing result.**

Converged trajectory, for reference: 13584 m in 402 s, SOC 0.617 at touchdown, phase
durations 20.8 / 31.0 / 317.5 / 7.9 / 25.0 s.

Measured clearance, at each path's own coordinates rather than by resampling the straight-line
profile:

| profile | time | peak | min AGL | nodes below ground |
|---|---|---|---|---|
| high_overfly | 9.9 min | 3156 m | −0.4 m | 1 (touchdown pixel) |
| terrain_follow | 8.2 min | 3003 m | **−149.5 m** | 20 |
| minimal_energy | 8.6 min | 3126 m | **−165.6 m** | 4 |
| sizing MDO cruise | — | 3126 m | +100.0 m | 0 |

Two of the three candidates fly *through* the ridge. This is detected, not ignored:
`run_mission.py` refuses to rank on energy when nothing clears terrain and selects the least
unsafe candidate with an explicit warning, because ranking on energy once picked the −166 m
path over one missing clearance by 0.4 m of pad interpolation noise. But the generator itself
produces unusable candidates on this corridor, and TERRAIN_FOLLOW — whose entire purpose is to
follow terrain — is the worst of them.

Figures: `reports/mission_profiles_2d.png` (altitude and AGL against ground distance) and
`reports/mission_profiles_3d.png` (the same tracks over the DEM). Regenerate with
`python -m hpraptor_mdao.run profiles`.

The figures currently show the three m1 candidates and the sizing MDO's cruise altitude. The
dymos trace is **not** among them: `plot_profiles_2d` accepts a `dymos` key but
`collect_profiles` does not populate one, because a converged trajectory costs ~1.5 h and is
not something to run inside a plotting command. Wiring `run mission`'s solution into that key
is the obvious next step.

---

# Part II — The code

## Layout

```
hpraptor/                 physics library
├── core/                 config, mission loader, atmosphere, flight path/segments
├── m1_mission/           NASADEM ingestion, terrain surrogate, path building
├── m2_geometry/          wing planform, fuselage, tail, rotors; AeroSandbox assembly
├── m3_structures/        spar sizing, mass buildup, stability
├── m4_aero/              parasite drag buildup, AeroSandbox/VLM interface
├── m5_propulsion/        motors, ICE, fuel cell, battery catalogue, architecture blending
├── m6_dynamics/          differentiable 3-DoF equations of motion
└── postprocessing/       dashboards, 3D renderers, mission-profile figures, local server

hpraptor_mdao/            the OpenMDAO layer
├── components/           each discipline as an ExplicitComponent
├── trajectory/           dymos phases and ODEs
├── groups.py             the coupled sizing group (NLBGS on the mass loop)
├── problem.py            design variables, constraints, objective, driver
├── coupled.py            sizing + trajectory in one problem
├── campaign.py           every architecture × both fidelity paths, tabulated
├── iteration_history.py  record / print / plot what the driver did
├── mission_context.py    the m1 → MDO bridge
├── xdsm.py               conceptual and introspected model diagrams
└── run.py                CLI entry point
```

## Running it

```bash
pip install -e ".[all]"
setx OPENTOPOGRAPHY_API_KEY <your-free-key>     # Windows; export on POSIX
```

The key is optional but strongly recommended: without it the DEM builder silently falls back
from NASADEM (~31 m) to keyless SRTM (~111 m) **and overwrites your cached DEM**.

```bash
python -m hpraptor_mdao.run all                 # everything (~6 min)
python -m hpraptor_mdao.run campaign            # every architecture × both fidelity paths
python -m hpraptor_mdao.run campaign --quick    # analytical only, ~2 s
python -m hpraptor_mdao.run sizing --arch turbo_electric
python -m hpraptor_mdao.run sizing --history    # + iteration trace and figure
python -m hpraptor_mdao.run profiles            # mission-profile comparison, 2D and 3D
python -m hpraptor_mdao.run mission             # five-phase dymos trajectory
python -m hpraptor_mdao.run coupled             # sizing + trajectory in one loop
python -m hpraptor_mdao.run xdsm                # model diagrams
python -m hpraptor_mdao.run sizing --dry-run    # print the formulation, solve nothing
```

Interactive 3D pages must be served over `http://localhost`, not opened from disk: Chromium
gives `file://` pages an opaque origin, which blocks what vtk.js needs.
`python -m hpraptor.postprocessing.serve` exists for that.

## Watching a solve

`--history` records the driver's iterates and renders both a terminal table and
`reports/iterations_<arch>.png`. A converged objective says where the optimizer stopped; the
trace says whether it walked there. For the `series` run it shows SLSQP reaching feasibility at
iteration 5, *losing* it from 15 to 32, and recovering only at 33 — the sort of thing a results
table cannot show.

## A different route

Change two coordinate pairs and one cache path in a copy of `configs/quito_mission.yaml`:

```bash
python -m hpraptor.m1_mission.srtm_downloader --mission configs/mine.yaml \
    --source auto --resolution native --output data/dem/mine.npz
python -m hpraptor_mdao.run all --mission configs/mine.yaml
```

The corridor bounding box is derived from the endpoints; `altitude_amsl` is only a fallback,
since pad elevations are anchored to the DEM itself.

## Adding things

- **A discipline** → an `ExplicitComponent` in `hpraptor_mdao/components/`, promoted in
  `groups.py`. Use `declare_partials(method="cs")` and keep every branch on `.real` (see the
  complex-step rule at the top of `architecture_np.py`).
- **An architecture** → extend `ARCH_NAMES`, add a branch to `arch_power_split` in **both**
  `architecture_np.py` and `architecture_index.py`, and add its column to the `_HAS_*` presence
  vectors in `components/propulsion.py`. The parity test will catch a drift between the two
  implementations; nothing will catch a missing `_HAS_*` column.
- **Outputs** → `results/` for data, `reports/` for diagrams and figures, `figures/` for the
  legacy dashboards.

---

# Part III — What is known to be wrong

Listed so nobody rediscovers them the hard way. Roughly in order of how much they affect a
quoted number.

### 1. The stated design range is discarded

`configs/quito_mission.yaml` asks for `design_range_km: 50.0`. When terrain is loaded — the
default — `build_problem` overwrites `range_m` with the corridor's real distance, so the
vehicle is sized for **13.58 km**. The override is deliberate and documented ("sized for the
mission it will actually fly"), but the YAML key is then dead config that silently means
nothing. Either honour `max(design_range, corridor)` or remove the key. Any number quoted
against "50 km design range" is wrong.

### 2. Two design variables are inert for `all_electric`

With the architecture pinned to `all_electric`, both `k_electric` and `m_fuel` have **exactly
zero** influence on every response — they are gated out of the energy balance by `fuel_capable`
and out of the mass balance by `m_fuel_carried`. The problem therefore carries a
two-dimensional null space, SLSQP's QP subproblem is singular in those directions, and the
result becomes path-dependent.

On the analytical path the same problem, differing only by one extra `run_model()` call before
the driver, converges to 57.18 Wh or 57.64 Wh — differing almost entirely in taper (0.356 vs
0.652), with the better point feasible in both cases. On the AeroSandbox path three separate
campaign runs of the *identical* problem gave **54.81, 54.97 and 54.68 Wh** (26, 26 and 48
iterations). Every other architecture reproduces to the printed digits across the same runs;
only `all_electric` moves.

**The `all_electric` rows are not reliable global optima, and the spread is the same order as
the margin by which all-electric wins the campaign.** Fix: drop `k_electric` and `m_fuel` from
the design vector when the architecture is pinned to all-electric.

### 3. `series_parallel` is not a distinct architecture

It shares `parallel`'s branch in `arch_power_split` and `series`' component vector in
`ArchitectureComp._HAS_*`. It is parallel's physics with series' mass and nothing of its own,
which is why the two rows agree to every printed digit. One of the six architectures is a
duplicate.

### 4. `P_elec_bus` is computed and discarded

`ArchitectureComp` blends a real bus power and nothing in the sizing group consumes it.
`EnergyComp` instead derives battery energy from `P_hover`/`P_cruise` with a flat motor
efficiency. Related, and still open: `P_hover` already divides by `eta_motor` and g₆ divides
again (overstating peak pack draw ~15%), while cruise/climb battery energy omits the motor loss
entirely.

### 5. Converter masses are sized on fuel power, not shaft power

`P_ice = P_cruise / (0.30 × 0.90)` in `ArchitectureComp`. Separately, the turbine's part-load
lapse is evaluated against the base manager's fixed `P_max_sl` (~8 kW at `m_tow_guess = 20`)
rather than that sized power — so an 8 kW turbine is asked for ~330 W and sits on the 5% clamp
floor. Turbo-electric's 0.033–0.047 efficiency is therefore directionally right but dominated
by a rated-power mismatch, and should not be quoted as a turbine characteristic.

### 6. The m1 strategy generator produces unusable candidates

See Part I §7: TERRAIN_FOLLOW reaches −149 m AGL and MINIMAL_ENERGY −166 m on this corridor.
Detected and handled downstream, but the generator should not be producing them, and
PathBuilder is disconnected from the MDAO in any case.

### 7. The MDAO layer imports from the legacy entry script

`mission_context.py` and `trajectory_3d.py` both do `from run_mission import ...` for
`_resolve_dem_path` and `_facility_nodes_from_mission`. The new pipeline depends on the old
one's entry script, so `run_mission.py` cannot be retired until those two helpers move into
`hpraptor/core`.

### 8. The coupled problem does not reach KKT

`run coupled` returns feasible designs but exits on iteration limit rather than satisfying its
KKT test. It needs a real NLP solver (IPOPT/SNOPT via pyoptsparse) and better scaling.

### 9. The discreteness penalty is absolute

`λ = 200 Wh` regardless of problem scale. At long range it yields a non-discrete blend; λ of
order the objective recovers a one-hot answer. Every one-hot vertex is a local minimum, so a
relaxed run needs `--multistart` to break the symmetry — the penalty has exactly zero gradient
at `z_arch = 0`.

### 10. Taper is multimodal

With taper in the design vector the relaxed run on the AeroSandbox path finds the right
architecture but a local planform optimum. The pinned runs are the ones to quote.

---

*Model state as of the 2026-09-23 audit. `python -m hpraptor_mdao.run campaign` reproduces the
results table; `python -m pytest` runs 299 tests.*
