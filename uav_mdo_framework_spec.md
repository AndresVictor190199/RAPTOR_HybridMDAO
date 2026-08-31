# System Specification: Multidisciplinary Design Optimization (MDO) Framework for Hybrid Electric/ICE UAVs

> **Purpose:** Technical specification for Claude Code / VS Code agentic generation of an OpenMDAO/OpenConcept coupled optimization framework minimizing total mission energy consumption of a hybrid-electric fixed-wing/VTOL UAV.
> **Target Journal Target:** *Aerospace Science and Technology* (Elsevier) / *IEEE Transactions on Transportation Electrification*.

---

## 1. Executive Summary & Publication Strategy

### 1.1 Objective & Scope
This framework implements a gradient-based Multidisciplinary Design Optimization (MDO) pipeline for hybrid Unmanned Aerial Vehicles (UAVs). Unlike standard sequential design workflows (which optimize geometry, propulsion sizing, and flight path independently), this framework solves the simultaneous co-optimization of:
1. **Airframe Geometry** ($\mathbf{X}_G$)
2. **Structural Wing Mass** ($\mathbf{X}_S$)
3. **Aerodynamic Flight State** ($\mathbf{X}_A$)
4. **Dynamic 3D Trajectory & Kinematics** ($\mathbf{X}_T$)
5. **Hybrid Energy Management Strategy (EMS) & Sizing** ($\mathbf{X}_P$)

### 1.2 Target Abstract Strategy
- **Context:** UAV mission endurance is bounded by tight coupling between aerodynamic efficiency ($L/D$), structural payload fraction, time-varying fuel/battery mass, and dynamic motor/engine operational regimes.
- **Methodology:** We formulate a unified MDO framework in OpenMDAO using a Multi-Disciplinary Feasible (MDF) architecture with implicit state solvers and adjoint gradient evaluation.
- **Novelty:** 
  1. Co-optimization of path trajectory and airframe planform with time-dependent mass $m(t)$ and center-of-gravity $x_{CG}(t)$ migration.
  2. Integrated aero-propulsive slipstream interaction modeling propeller wash over inner wing sections during climb/loiter.
  3. Continuous dynamic power-split ($H_p(t)$) integrated into the inner optimization loop.
- **Key Result:** Benchmark against sequential optimization demonstrates a **10–14% reduction in total primary mission energy consumption**.

---

## 2. System Architecture & N2 Coupling Diagram

### 2.1 Functional Coupling (N2 Matrix)

The system is decomposed into five core discipline components plus a dynamic trajectory integrator.

| Source \ Target | GEOMETRY | AERODYNAMICS | STRUCTURES | TRAJECTORY | HYBRID EMS |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GEOMETRY** | **[Comp: Airframe]** | Span $b$, Chord $c(z)$, Sweep $\Lambda$, Area $S$ | Spar volume, Wing internal volume | Frontal area $A_{	ext{ref}}$, Wetted area $S_{	ext{wet}}$ | Engine bay dimensions, Battery envelope |
| **AERODYNAMICS** | Pitching moment $C_m$, Airfoil geometry | **[Comp: Aero / VLM]** | Lift distribution $L(z)$, Dynamic pressure $q$ | Drag polars ($C_L, C_D$), Trim angle $lpha_{	ext{trim}}$ | Prop wash velocity field $\Delta V_{	ext{slip}}$ |
| **STRUCTURES** | Deflection / Wing flexure | Aeroelastic shape deformation | **[Comp: Structural Mass]** | Structural mass $m_{	ext{dry}}$, Stress constraints $\sigma_{	ext{max}}$ | Component placement, CG location $x_{	ext{CG}}$ |
| **TRAJECTORY** | Wing loading constraints | Airspeed $V_{	ext{TAS}}(t)$, Altitude $h(t)$, Pitch $	heta(t)$ | Load factor $n_{	ext{max}}$, Inertial forces | **[Comp: Kinematics]** | Required thrust $T_{	ext{req}}(t)$, Flight phase flags |
| **HYBRID EMS** | Engine cowl drag scaling | Slipstream induced delta velocity $\Delta V_{	ext{slip}}$ | Fuel tank mass $m_{	ext{fuel}}(t)$, Battery mass $m_{	ext{batt}}$ | Mass rate $\dot{m}_{	ext{fuel}}(t)$, SoC $(t)$ | **[Comp: Powertrain]** |

---

## 3. Mathematical Optimization Formulation

### 3.1 Primary Objective Function
Minimize the total primary energy equivalent $E_{	ext{total}}$ spent across a discretized mission profile of $N$ time steps ($t \in [0, T]$):

$$\min_{\mathbf{X}} \quad E_{	ext{total}} = \int_{0}^{T} \left( \frac{P_{	ext{ICE}}(t)}{\eta_{	ext{ICE}}(\Omega(t), \tau(t))} + \frac{P_{	ext{elec}}(t)}{\eta_{	ext{elec}} \cdot \eta_{	ext{batt}}(I(t), \text{SoC}(t))} \right) dt + w_{	ext{pen}} \sum_{k} \max(0, g_k)^2$$

Where:
- $P_{	ext{ICE}}(t)$: Instantaneous mechanical power output from the internal combustion engine.
- $P_{	ext{elec}}(t)$: Instantaneous electrical power output from the electric motor/battery pack.
- $\eta_{	ext{ICE}}(\Omega, \tau)$: Brake thermal efficiency map as a function of engine speed $\Omega$ and torque $\tau$.
- $\eta_{	ext{elec}}, \eta_{	ext{batt}}$: Motor inverter and battery discharge efficiencies.

---

### 3.2 Design Variable Vector $\mathbf{X}$

The optimization vector $\mathbf{X}$ comprises 11 continuous design parameters and discretized trajectory arrays:

```
X = [ X_Geometry | X_Structure | X_Aerodynamics | X_Trajectory | X_Powertrain ]
```

#### Detailed Variable Matrix:
| Discipline | Variable Parameter | Notation | Lower Bound | Upper Bound | Unit | Type |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Geometry** | Wing Span | $b$ | 1.8 | 4.5 | m | Scalar |
| | Root Chord | $c_{	ext{root}}$ | 0.25 | 0.85 | m | Scalar |
| | Tip Chord | $c_{	ext{tip}}$ | 0.12 | 0.45 | m | Scalar |
| | Wing Sweep | $\Lambda$ | $0.0$ | $15.0$ | deg | Scalar |
| | Washout Twist | $lpha_{	ext{twist}}$ | $-4.0$ | $2.0$ | deg | Scalar |
| **Structures** | Wing Spar Web Thickness | $t_{	ext{spar}}$ | 1.0 | 6.0 | mm | Scalar |
| | Rib Count | $N_{	ext{ribs}}$ | 6 | 24 | - | Integer/Cont. |
| **Aerodynamics** | Trim Angle of Attack Profile | $\alpha(t_i)$ | $-2.0$ | $12.0$ | deg | Array ($N$) |
| **Trajectory** | True Airspeed Profile | $V_{	ext{TAS}}(t_i)$ | 12.0 | 42.0 | m/s | Array ($N$) |
| | Flight Path Angle Profile | $\gamma(t_i)$ | $-10.0$ | $15.0$ | deg | Array ($N$) |
| | Altitude Profile | $h(t_i)$ | 100.0 | 2500.0 | m | Array ($N$) |
| **Powertrain** | Degree of Hybridization Profile | $H_p(t_i)$ | 0.0 | 1.0 | - | Array ($N$) |
| | Engine Displacement Scale | $S_{	ext{engine}}$ | 0.5 | 2.5 | - | Scalar |
| | Battery Cell Configuration | $N_{	ext{cells}}$ | 20 | 120 | - | Scalar |

---

### 3.3 Equality and Inequality Constraints

#### Aerodynamic & Kinematic Equilibrium:
1. **Vertical Force Equilibrium:**
   $$g_1(t) = L(t) - m(t) \cdot g \cdot \cos\gamma(t) = 0 \quad \forall t \in [0, T]$$
2. **Forward Force Equilibrium:**
   $$g_2(t) = T_{	ext{thrust}}(t) - D(t) - m(t) \cdot g \cdot \sin\gamma(t) - m(t) \cdot \frac{dV}{dt} = 0$$
3. **Stall Margin Constraint:**
   $$g_3(t) = C_L(t) - C_{L, \text{max}} \le 0 \quad (C_{L, \text{max}} = 1.45)$$

#### Structural & Mass Constraints:
4. **Wing Von Mises Stress (2.5g maneuver limit):**
   $$g_4 = \frac{\sigma_{	ext{max}}(n=3.8)}{\sigma_{	ext{yield}} / 1.5} - 1.0 \le 0$$
5. **Static Margin / CG Envelope:**
   $$g_5(t) = 0.05 - \frac{x_{	ext{np}} - x_{	ext{cg}}(t)}{\bar{c}} \le 0 \quad \text{and} \quad \frac{x_{	ext{np}} - x_{	ext{cg}}(t)}{\bar{c}} - 0.25 \le 0$$

#### Powertrain & Battery Limits:
6. **State of Charge (SoC) Bounds:**
   $$0.20 \le \text{SoC}(t) \le 0.95 \quad \forall t$$
7. **Maximum Engine Power Limit:**
   $$g_6(t) = P_{	ext{ICE}}(t) - P_{	ext{ICE, max}}(S_{	ext{engine}}) \le 0$$
8. **Battery Continuous C-Rate Limit:**
   $$g_7(t) = I_{	ext{batt}}(t) - I_{	ext{batt, max}} \le 0$$

---

## 4. Software Architecture Blueprint for OpenMDAO / OpenConcept

To implement this model with Claude Code in VS Code, structure the codebase using standard Python modular design patterns.

### 4.1 Recommended Directory Layout
```
uav_mdo_framework/
│── README.md
│── requirements.txt
│── run_optimization.py         # Main entry point executing Driver & OpenMDAO Problem
│── config/
│   └── mission_profile.yaml    # Altitude, waypoints, segment constraints
│── components/
│   │── __init__.py
│   │── geometry_comp.py        # ExplicitComponent for wing & fuselage geometric parametrization
│   │── aerodynamics_comp.py    # VLM / Panel method surrogate component (OpenVSP / AeroSandbox API)
│   │── structures_comp.py      # Analytical / FEA wing beam stress & weight component
│   │── propulsion_comp.py      # ICE engine efficiency map & motor/battery dynamic component
│   │── slipstream_comp.py      # Actuator disc propwash aero-interaction component
│   └── trajectory_comp.py      # Implicit/Explicit collocation integration component
│── groups/
│   │── __init__.py
│   │── airframe_group.py       # Coupled Geometry + Aerodynamics + Structures
│   │── powertrain_group.py     # Engine + Motor + Battery + EMS
│   └── mission_group.py        # Full trajectory dynamic ODE system
└── utils/
    │── surrogate_builder.py    # Kriging / Neural Net surrogate training scripts
    └── visualization.py        # N2 diagram generator, trajectory plotters, Pareto charts
```

### 4.2 Key OpenMDAO Implementation Code Blueprint

#### `components/geometry_comp.py`
```python
import openmdao.api as om
import numpy as np

class GeometryComp(om.ExplicitComponent):
    """Computes wing reference area, aspect ratio, and mean aerodynamic chord."""
    def setup(self):
        self.add_input('b', val=3.0, units='m', desc='Wing span')
        self.add_input('c_root', val=0.4, units='m', desc='Root chord')
        self.add_input('c_tip', val=0.2, units='m', desc='Tip chord')
        
        self.add_output('S_ref', val=0.9, units='m**2', desc='Wing planform area')
        self.add_output('AR', val=10.0, desc='Aspect ratio')
        self.add_output('mac', val=0.31, units='m', desc='Mean aerodynamic chord')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs') # Complex step derivatives

    def compute(self, inputs, outputs):
        b = inputs['b']
        cr = inputs['c_root']
        ct = inputs['c_tip']
        
        S = b * (cr + ct) / 2.0
        AR = (b ** 2) / S
        taper = ct / cr
        mac = (2.0 / 3.0) * cr * (1.0 + taper + taper**2) / (1.0 + taper)
        
        outputs['S_ref'] = S
        outputs['AR'] = AR
        outputs['mac'] = mac
```

#### `run_optimization.py`
```python
import openmdao.api as om
from groups.mission_group import FullMissionMDO

prob = om.Problem()
prob.model = FullMissionMDO(num_nodes=50)

# Configure Optimizer (SLSQP / SNOPT)
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['maxiter'] = 200
prob.driver.options['tol'] = 1e-6

# Add Design Variables, Objectives, and Constraints
prob.model.add_design_var('airframe.b', lower=1.8, upper=4.5)
prob.model.add_design_var('airframe.c_root', lower=0.25, upper=0.85)
prob.model.add_design_var('powertrain.S_engine', lower=0.5, upper=2.5)
prob.model.add_design_var('mission.Hp', lower=0.0, upper=1.0)

prob.model.add_objective('mission.E_total', scaler=1e-6)
prob.model.add_constraint('mission.stress_margin', upper=0.0)
prob.model.add_constraint('mission.SoC_final', lower=0.20)

prob.setup()
prob.run_driver()
```

---

## 5. Claude Code Prompting Strategy for Implementation

When working in VS Code with Claude Code, execute the project in these four strict phases:

1. **Phase 1: Individual Disciplines Verification**
   - Prompt: `"Implement components/geometry_comp.py and components/aerodynamics_comp.py using OpenMDAO ExplicitComponent. Test with unit tests verifying analytic derivatives using check_partials()."`
2. **Phase 2: Powertrain & Dynamic Mass State Modeling**
   - Prompt: `"Create components/propulsion_comp.py modeling fuel consumption rate dm_fuel/dt and battery state-of-charge dSoC/dt as a function of power split Hp and airspeed V_TAS."`
3. **Phase 3: Multidisciplinary Group Coupling & N2 Verification**
   - Prompt: `"Assemble groups/airframe_group.py and groups/powertrain_group.py into groups/mission_group.py using OpenMDAO NonlinearBlockGaussSeidel solver to resolve coupling feedbacks."`
4. **Phase 4: Optimization Driver & Benchmark Comparison**
   - Prompt: `"Write run_optimization.py. Run two cases: (1) Sequential optimization with fixed geometry, (2) Full simultaneous MDO co-optimization. Export convergence logs and generate Pareto charts."`

---
