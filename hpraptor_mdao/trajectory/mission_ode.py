"""
Mission-phase ODE for dymos.

One component, instantiated once per phase with `phase` fixed at setup,
so the phase branch is resolved before any node is evaluated rather than
inside the inner loop.

Wraps m6_dynamics.eom_np for the kinematics and m5_propulsion's
architecture blending for the power split, so the trajectory and the
static sizing model share one set of physics.
"""

from __future__ import annotations
import numpy as np
import openmdao.api as om

from hpraptor.m5_propulsion.architecture_index import ContinuousArchitectureManager
from hpraptor.m5_propulsion.architecture_np import NumpyArchitectureManager
from hpraptor.m6_dynamics import eom_np
from hpraptor_mdao.mission_context import TerrainModel, evaluate_terrain

N_ARCH = 6


class MissionPhaseODE(om.ExplicitComponent):
    """Flight dynamics plus blended propulsion for one mission phase."""

    def initialize(self):
        self.options.declare("num_nodes", types=int)
        self.options.declare("phase", values=eom_np.PHASES)
        self.options.declare("base_manager", types=ContinuousArchitectureManager)
        self.options.declare("eta_prop", default=0.75, types=float)
        self.options.declare("kappa_i", default=1.15, types=float)
        self.options.declare("terrain_model", default=None, types=TerrainModel,
                             allow_none=True,
                             desc="Differentiable terrain(x) from m1. When "
                                  "given, ground elevation follows the real "
                                  "profile instead of a per-phase constant.")

    def setup(self):
        nn = self.options["num_nodes"]
        self._npm = NumpyArchitectureManager(self.options["base_manager"])
        self._vtol = self.options["phase"] in eom_np.VTOL_PHASES

        # States consumed by the rates. `x` is needed whenever terrain
        # varies along the route: ground elevation is then a function of how
        # far along the corridor the vehicle is, not a constant.
        self.add_input("x", val=np.zeros(nn), units="m")
        self.add_input("h", val=np.full(nn, 2900.0), units="m")
        self.add_input("v", val=np.full(nn, 30.0), units="m/s")
        self.add_input("m", val=np.full(nn, 20.0), units="kg")

        # Controls
        self.add_input("T_vtol", val=np.zeros(nn), units="N")
        self.add_input("T_cruise", val=np.full(nn, 15.0), units="N")
        self.add_input("k_electric", val=np.full(nn, 0.5))
        self.add_input("alpha_deg", val=np.full(nn, 2.0), units="deg")

        # Parameters
        self.add_input("S_ref", val=np.full(nn, 0.7), units="m**2")
        self.add_input("AR", val=np.full(nn, 10.0))
        self.add_input("C_D0", val=np.full(nn, 0.02))
        self.add_input("A_rotor", val=np.full(nn, 0.65), units="m**2")
        self.add_input("E_batt_J", val=np.full(nn, 9.0e5), units="J")
        self.add_input("arch_weights", val=np.tile(np.ones(N_ARCH) / N_ARCH, (nn, 1)))
        self.add_input("ground_elevation", val=np.full(nn, 2850.0), units="m")
        self.add_input("path_heading_deg", val=np.zeros(nn), units="deg")
        self.add_input("wind_speed_ref", val=np.zeros(nn), units="m/s")
        self.add_input("wind_heading_deg", val=np.zeros(nn), units="deg")

        # Rates
        self.add_output("x_dot", val=np.full(nn, 30.0), units="m/s",
                        tags=["dymos.state_rate_source:x"])
        self.add_output("h_dot", val=np.zeros(nn), units="m/s",
                        tags=["dymos.state_rate_source:h"])
        self.add_output("v_dot", val=np.zeros(nn), units="m/s**2",
                        tags=["dymos.state_rate_source:v"])
        self.add_output("m_dot", val=np.zeros(nn), units="kg/s",
                        tags=["dymos.state_rate_source:m"])
        self.add_output("SOC_dot", val=np.zeros(nn), units="1/s",
                        tags=["dymos.state_rate_source:SOC"])

        # Diagnostics / constraint sources
        self.add_output("gamma", val=np.zeros(nn), units="rad")
        self.add_output("C_L", val=np.full(nn, 0.4))
        self.add_output("P_shaft", val=np.full(nn, 500.0), units="W")
        self.add_output("P_elec_bus", val=np.full(nn, 400.0), units="W")
        self.add_output("fuel_flow", val=np.zeros(nn), units="kg/s")
        self.add_output("agl", val=np.full(nn, 50.0), units="m",
                        desc="Height above ground, for terrain clearance")
        self.add_output("ground_elev_used", val=np.full(nn, 2850.0), units="m",
                        desc="Terrain height the clearance constraint used")

        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        phase = self.options["phase"]
        h, v, m = inputs["h"], inputs["v"], inputs["m"]
        w = inputs["arch_weights"]

        # Ground elevation under the vehicle, resolved once so the wind
        # model's AGL and the clearance constraint reference the same
        # terrain. With a terrain model this tracks the real profile —
        # including the ridge the route crosses — instead of the per-phase
        # constant that let a trajectory satisfy a 100 m AGL constraint
        # while flying through a mountain.
        terrain = self.options["terrain_model"]
        ground = (inputs["ground_elevation"] if terrain is None
                  else evaluate_terrain(terrain, inputs["x"]))

        P_shaft = eom_np.shaft_power(
            inputs["T_vtol"], inputs["T_cruise"], v, h, inputs["A_rotor"],
            eta_prop=self.options["eta_prop"], kappa_i=self.options["kappa_i"],
        )

        results = [self._npm.arch_power_split(a, P_shaft, inputs["k_electric"], h)
                   for a in self._npm.ARCH_NAMES]
        fuel_flow = sum(w[:, i] * r["fuel_flow_kg_s"] for i, r in enumerate(results))
        P_bus = sum(w[:, i] * r["P_elec_from_bus"] for i, r in enumerate(results))

        params = {
            "S_ref": inputs["S_ref"], "AR": inputs["AR"], "C_D0": inputs["C_D0"],
            "E_batt_J": inputs["E_batt_J"],
            "ground_elevation": ground,
            "path_heading_deg": inputs["path_heading_deg"],
            "wind_speed_ref": inputs["wind_speed_ref"],
            "wind_heading_deg": inputs["wind_heading_deg"],
            "z_0": 0.1, "h_ref": 10.0,
        }
        controls = {
            "T_vtol": inputs["T_vtol"], "T_cruise": inputs["T_cruise"],
            "alpha_deg": inputs["alpha_deg"],
        }
        rates = eom_np.state_rates(
            phase, {"h": h, "v": v, "m": m}, controls, params, fuel_flow, P_bus,
        )

        for key in ("x_dot", "h_dot", "v_dot", "m_dot", "SOC_dot"):
            outputs[key] = rates[key]
        outputs["gamma"] = rates["gamma"]

        if self._vtol:
            # No wing loading in hover; report zero rather than a C_L that
            # would be meaningless (and huge) at near-zero airspeed.
            outputs["C_L"] = 0.0 * v
        else:
            _, _, C_L = eom_np.aerodynamic_forces(
                v, inputs["alpha_deg"], h, inputs["S_ref"], inputs["AR"], inputs["C_D0"])
            outputs["C_L"] = C_L

        outputs["P_shaft"] = P_shaft
        outputs["P_elec_bus"] = P_bus
        outputs["fuel_flow"] = fuel_flow
        outputs["ground_elev_used"] = ground
        outputs["agl"] = h - ground
