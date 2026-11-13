"""
Mission-phase ODE for dymos.

One component, instantiated once per phase with `phase` fixed at setup.
Wraps m6_dynamics.eom_np for the dynamics, m5_propulsion.rotor_power for the
rotor/propeller/battery physics and architecture_np for the power split —
the same functions the sizing model uses, at the same component ratings, so
the trajectory flies the aircraft the sizer built rather than a nominal one.

Motor and pack limits are outputs here and PATH CONSTRAINTS in the
problems: a trajectory that asks the lift motors, the cruise motor or the
pack for more than their ratings is not a flight of this aircraft. The
previous ODE had no such limits and bounded rotor thrust at 3 W.
"""

from __future__ import annotations
import numpy as np
import openmdao.api as om

from hpraptor.m5_propulsion.architecture_index import ContinuousArchitectureManager
from hpraptor.m5_propulsion.architecture_np import NumpyArchitectureManager
from hpraptor.m5_propulsion import battery_catalogue as batcat
from hpraptor.m5_propulsion import rotor_power as rp
from hpraptor.m5_propulsion.technology import DEFAULT_TECHNOLOGY, PowertrainTechnology
from hpraptor.m6_dynamics import eom_np
from hpraptor_mdao.mission_context import TerrainModel, evaluate_terrain

N_ARCH = 6
G = 9.80665


class MissionPhaseODE(om.ExplicitComponent):
    """Flight dynamics plus blended propulsion for one mission phase."""

    def initialize(self):
        self.options.declare("num_nodes", types=int)
        self.options.declare("phase", values=eom_np.PHASES)
        self.options.declare("base_manager", types=ContinuousArchitectureManager)
        self.options.declare("technology", default=DEFAULT_TECHNOLOGY,
                             types=PowertrainTechnology)
        self.options.declare("cell", default=batcat.DEFAULT_CELL, types=str)
        self.options.declare("pack_overhead", default=batcat.DEFAULT_PACK_OVERHEAD,
                             types=float)
        self.options.declare("eta_prop", default=0.75, types=float)
        self.options.declare("kappa_i", default=1.15, types=float)
        self.options.declare("wind_z0", default=0.1, types=float)
        self.options.declare("wind_h_ref", default=10.0, types=float)
        self.options.declare("terrain_model", default=None, types=TerrainModel,
                             allow_none=True,
                             desc="Differentiable terrain(x) from m1. When "
                                  "given, ground elevation follows the real "
                                  "profile instead of a per-phase constant.")

    def setup(self):
        nn = self.options["num_nodes"]
        tech = self.options["technology"]
        self._npm = NumpyArchitectureManager(self.options["base_manager"], technology=tech)
        self._vtol = self.options["phase"] in eom_np.VTOL_PHASES

        # States
        self.add_input("x", val=np.zeros(nn), units="m")
        self.add_input("h", val=np.full(nn, 2900.0), units="m")
        self.add_input("v", val=np.full(nn, 30.0), units="m/s")
        self.add_input("m", val=np.full(nn, 10.0), units="kg")
        if not self._vtol:
            self.add_input("gam", val=np.zeros(nn), units="rad")

        # Controls
        self.add_input("T_vtol", val=np.zeros(nn), units="N")
        self.add_input("T_cruise", val=np.full(nn, 5.0), units="N")
        self.add_input("k_electric", val=np.full(nn, 0.5))
        self.add_input("alpha_deg", val=np.full(nn, 2.0), units="deg")

        # Vehicle parameters (from the sizing)
        self.add_input("S_ref", val=np.full(nn, 0.3), units="m**2")
        self.add_input("AR", val=np.full(nn, 15.0))
        self.add_input("e_oswald", val=np.full(nn, 0.8))
        self.add_input("C_D0", val=np.full(nn, 0.03))
        self.add_input("A_rotor", val=np.full(nn, 1.0), units="m**2")
        self.add_input("E_batt_J", val=np.full(nn, 5.0e5), units="J")
        self.add_input("m_battery", val=np.full(nn, 1.0), units="kg")
        self.add_input("P_lift_rated", val=np.full(nn, 1500.0), units="W")
        self.add_input("P_cm_rated", val=np.full(nn, 800.0), units="W")
        self.add_input("P_conv_rated", val=np.full((nn, N_ARCH), 400.0), units="W")
        self.add_input("W_ref", val=np.full(nn, 100.0), units="N",
                       desc="Reference weight for the rotor-running gate")
        self.add_input("arch_weights", val=np.tile(np.ones(N_ARCH) / N_ARCH, (nn, 1)))
        # Environment
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
        if not self._vtol:
            self.add_output("gam_dot", val=np.zeros(nn), units="rad/s",
                            tags=["dymos.state_rate_source:gam"])
        self.add_output("m_dot", val=np.zeros(nn), units="kg/s",
                        tags=["dymos.state_rate_source:m"])
        self.add_output("SOC_dot", val=np.zeros(nn), units="1/s",
                        tags=["dymos.state_rate_source:SOC"])

        # Diagnostics / path-constraint sources
        self.add_output("C_L", val=np.full(nn, 0.4))
        self.add_output("P_rotor", val=np.zeros(nn), units="W")
        self.add_output("P_prop", val=np.full(nn, 200.0), units="W")
        self.add_output("P_shaft", val=np.full(nn, 200.0), units="W")
        self.add_output("P_elec_bus", val=np.full(nn, 230.0), units="W",
                        desc="Battery terminal power")
        self.add_output("fuel_flow", val=np.zeros(nn), units="kg/s")
        self.add_output("lift_power_margin", val=np.full(nn, -0.5),
                        desc="P_rotor / lift-motor rating - 1  (<= 0)")
        self.add_output("cm_power_margin", val=np.full(nn, -0.5),
                        desc="P_prop / cruise-motor rating - 1  (<= 0)")
        self.add_output("pack_power_margin", val=np.full(nn, -0.5),
                        desc="Battery terminal power / pack limit - 1  (<= 0)")
        self.add_output("agl", val=np.full(nn, 50.0), units="m",
                        desc="Height above ground, for terrain clearance")
        self.add_output("ground_elev_used", val=np.full(nn, 2850.0), units="m")

        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        opt = self.options
        tech = opt["technology"]
        phase = opt["phase"]
        h, v, m = inputs["h"], inputs["v"], inputs["m"]
        w = inputs["arch_weights"]

        terrain = opt["terrain_model"]
        ground = (inputs["ground_elevation"] if terrain is None
                  else evaluate_terrain(terrain, inputs["x"]))

        P_rot, P_prop = eom_np.propulsor_powers(
            phase, inputs["T_vtol"], inputs["T_cruise"], v, h, inputs["A_rotor"],
            inputs["W_ref"], eta_prop=opt["eta_prop"], kappa_i=opt["kappa_i"])

        P_lift_r = inputs["P_lift_rated"]
        P_cm_r = inputs["P_cm_rated"]
        P_conv = inputs["P_conv_rated"]
        results = [self._npm.arch_power_split(
                       a, P_prop, inputs["k_electric"], h,
                       ratings={"motor": P_cm_r, "lift": P_lift_r, "conv": P_conv[:, i]},
                       P_lift=P_rot)
                   for i, a in enumerate(self._npm.ARCH_NAMES)]
        fuel_flow = sum(w[:, i] * r["fuel_flow_kg_s"] for i, r in enumerate(results))
        P_batt = sum(w[:, i] * r["P_elec_from_bus"] for i, r in enumerate(results))

        cell = batcat.get_cell(opt["cell"])
        n_cells = batcat.pack_cell_count(inputs["m_battery"], cell, opt["pack_overhead"])
        eta_dis = rp.battery_discharge_efficiency(P_batt, n_cells, cell.voltage_nom_v,
                                                  cell.r_internal_ohm)
        P_limit = batcat.pack_power_limit_w(inputs["m_battery"], cell, opt["pack_overhead"])

        params = {
            "S_ref": inputs["S_ref"], "AR": inputs["AR"], "C_D0": inputs["C_D0"],
            "e_oswald": inputs["e_oswald"], "E_batt_J": inputs["E_batt_J"],
            "ground_elevation": ground,
            "path_heading_deg": inputs["path_heading_deg"],
            "wind_speed_ref": inputs["wind_speed_ref"],
            "wind_heading_deg": inputs["wind_heading_deg"],
            "z_0": opt["wind_z0"], "h_ref": opt["wind_h_ref"],
        }
        controls = {"T_vtol": inputs["T_vtol"], "T_cruise": inputs["T_cruise"],
                    "alpha_deg": inputs["alpha_deg"]}
        states = {"h": h, "v": v, "m": m}
        if not self._vtol:
            states["gam"] = inputs["gam"]
        rates = eom_np.state_rates(phase, states, controls, params, fuel_flow,
                                   P_batt / eta_dis)

        for key in ("x_dot", "h_dot", "v_dot", "m_dot", "SOC_dot"):
            outputs[key] = rates[key]
        if not self._vtol:
            outputs["gam_dot"] = rates["gam_dot"]

        if self._vtol:
            outputs["C_L"] = 0.0 * v
        else:
            _, _, C_L = eom_np.aerodynamic_forces(
                v, inputs["alpha_deg"], h, inputs["S_ref"], inputs["AR"],
                inputs["C_D0"], e_oswald=inputs["e_oswald"])
            outputs["C_L"] = C_L

        outputs["P_rotor"] = P_rot
        outputs["P_prop"] = P_prop
        outputs["P_shaft"] = P_rot + P_prop
        outputs["P_elec_bus"] = P_batt
        outputs["fuel_flow"] = fuel_flow
        outputs["lift_power_margin"] = P_rot / P_lift_r - 1.0
        outputs["cm_power_margin"] = P_prop / P_cm_r - 1.0
        outputs["pack_power_margin"] = P_batt / P_limit - 1.0
        outputs["ground_elev_used"] = ground
        outputs["agl"] = h - ground
