"""
Continuous Architecture Relaxation — NumPy / Complex-Step-Safe Physics
=========================================================================

The power-split physics of the six propulsion architectures, written so it
stays exact under COMPLEX-STEP differentiation (OpenMDAO's
``declare_partials(method='cs')``), and the softmax blending that turns the
discrete architecture choice into a continuous design variable.

This is the single implementation the sizing MDO (``ArchitectureComp``,
``CruiseSplitComp``) and the dymos trajectory ODEs share. The CasADi twin in
``architecture_index.casadi_arch_power_split`` mirrors it for the parity
tests; when the two disagree the tests fail.

What changed, and why
---------------------
**Ratings are arguments.** Every part-load efficiency — motor, engine
Willans line, turbine SFC lapse, fuel-cell polarization — is evaluated
against the rating of the component the mass model actually bought. They
used to be evaluated against ratings frozen at a 20 kg first guess (a 6 kW
motor flying a 200 W cruise at 3 % load), which made every fuel-path
efficiency the model reported a property of that mismatch instead of the
architecture. Pass ``ratings=None`` to fall back to the base manager's
component values, which is what the parity tests do.

**One motor per drivetrain.** Series, turbo-electric and fuel-cell aircraft
drive the propulsor through ONE motor fed from a shared bus. The old code
evaluated that motor twice, at the battery share and at the fuel share
separately, as if there were two half-loaded motors — double-counting the
part-load penalty — and then drew the motor loss on the fuel share from the
battery. The bus demand is now formed once and split between battery and
converter.

**A saturated converter does not create energy.** Where the fuel path is
asked for more than it can deliver at altitude, the shortfall is drawn from
the battery. It used to be silently unserved: the engine was clamped at its
limit while the shaft received the full demand anyway.

**Series-parallel is its own architecture.** It shared parallel's power
split and series' component list, so its results duplicated series to every
printed digit. It is now a power-split hybrid: a fraction
``sp_mech_fraction`` of engine output drives the propeller mechanically and
the rest goes through the generator to the bus.

**A lift-rotor demand can ride on the same bus.** ``P_lift`` (shaft power
of the lift rotors, through their own motors and rating) is added to the bus
demand. Electric drivetrains can serve it from the converter; parallel and
series-parallel, whose engine is geared to the cruise propeller, cannot.

COMPLEX-STEP RULE
-----------------
Any comparison is taken on the real part only (``_re``). Returned values are
never truncated to it.

Author: Victor Berrazueta (LUAS-EPN)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from hpraptor.m5_propulsion.architecture_index import ContinuousArchitectureManager
from hpraptor.m5_propulsion import fuel_cell_polarization as fc_polarization
from hpraptor.m5_propulsion.smooth import (  # re-exported for existing imports
    _re, _smoothing_eps, smooth_max, smooth_min, smooth_switch,
)
from hpraptor.m5_propulsion.technology import DEFAULT_TECHNOLOGY, PowertrainTechnology
from hpraptor.m5_propulsion import rotor_power as rp

__all__ = [
    "smooth_max", "smooth_min", "smooth_switch", "softmax_weights",
    "discreteness_penalty", "NumpyArchitectureManager", "FC_RATED_CURRENT_DENSITY",
]

#: Current density at which a stack is "rated" [A/cm^2]. The polarization
#: model has no mass-transport term, so its power curve need not peak inside
#: i_max; rating at a fixed current density instead of at a fraction of the
#: peak keeps the rated point where real air-cooled stacks put it (~0.65-0.7
#: V/cell, ~50 % system efficiency).
FC_RATED_CURRENT_DENSITY = 0.8


# ═══════════════════════════════════════════════════════════════════════════
# BLENDING PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════

def softmax_weights(z_vec: np.ndarray, temp: float = 1.5) -> np.ndarray:
    """
    Architecture blending weights, w_i = exp(temp*z_i) / sum_j exp(temp*z_j).

    Shifted by max(Re(z)) for numerical stability. The shift cancels
    exactly in the ratio, so it does not perturb the complex step.
    """
    z_shift = z_vec - np.max(_re(z_vec))
    e = np.exp(temp * z_shift)
    return e / np.sum(e)


def discreteness_penalty(weights: np.ndarray) -> Any:
    """
    Projection penalty sum_i w_i*(1 - w_i).

    Zero exactly when the weights are one-hot (a buildable, committed
    architecture) and maximal (5/6) when they are spread evenly.
    """
    return np.sum(weights * (1.0 - weights))


def _clamp01(k: Any) -> Any:
    """
    Power split clamped to [0, 1] inside the physics.

    A collocation optimizer enforces control bounds only at its nodes, so a
    split can overshoot between them; k > 1 would make the fuel share
    negative and let the vehicle gain mass. The smoothing scale is small
    (offset ~5e-6) so a commanded k = 1 leaves no phantom idle fuel flow.
    """
    return smooth_min(smooth_max(k, 0.0, scale=0.01), 1.0, scale=0.01)


# ═══════════════════════════════════════════════════════════════════════════
# NUMPY ARCHITECTURE MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class NumpyArchitectureManager:
    """
    Complex-step-safe power-split physics for the six architectures.

    Borrows the component TECHNOLOGY objects (efficiency maps, Willans
    slope, polarization curve) from a ContinuousArchitectureManager. The
    component RATINGS come from the caller through ``ratings``; the base
    manager's ratings are only a fallback.
    """

    ARCH_NAMES = ContinuousArchitectureManager.ARCH_NAMES

    #: Lower heating value [J/kg] per architecture, ARCH_NAMES order. Index 5
    #: (fuel cell) burns hydrogen. all_electric carries no fuel, but its entry
    #: stays finite so it cannot put a 0/0 into a blended denominator.
    FUEL_LHV = np.array([43.0e6, 43.0e6, 43.0e6, 43.0e6, 43.0e6, 120.0e6])

    def __init__(self, base_manager: ContinuousArchitectureManager,
                 technology: Optional[PowertrainTechnology] = None):
        self.base = base_manager
        self.prop_systems = base_manager.prop_systems
        self.tech = (technology or getattr(base_manager, "technology", None)
                     or DEFAULT_TECHNOLOGY)
        t = self.tech
        self.FUEL_LHV = np.array([t.lhv_hydrocarbon_j_kg] * 5 + [t.lhv_h2_j_kg])

        fc = self.prop_systems["fuel_cell"].fuel_cell
        # Net power of the reference stack at its rated current density; a
        # stack rated r_fc delivers demand P at the efficiency the reference
        # stack has at P * P_ref / r_fc (net power scales with active area).
        self._fc_P_ref = float(np.real(
            fc_polarization.net_power(fc, FC_RATED_CURRENT_DENSITY)))

    # ── Ratings ──────────────────────────────────────────────────────────

    def default_ratings(self, arch: str) -> Dict[str, Any]:
        """The base manager's own component ratings (legacy behaviour)."""
        ps = self.prop_systems[arch]
        conv = None
        if arch in ("series", "parallel", "series_parallel"):
            conv = ps.ice.P_max_sl
        elif arch == "turbo_electric":
            conv = ps.gas_turbine.P_max_sl
        return {"motor": ps.motor.P_max, "lift": ps.motor.P_max, "conv": conv}

    def _resolve(self, arch: str, ratings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        r = self.default_ratings(arch)
        scaled = False
        if ratings is not None:
            for key in ("motor", "lift", "conv"):
                if ratings.get(key) is not None:
                    r[key] = ratings[key]
            scaled = ratings.get("conv") is not None
        r["_scaled"] = scaled
        return r

    # ── Electric drive ───────────────────────────────────────────────────

    def _motor_electrical_power(self, arch: str, P_mech: Any,
                                P_rated: Any = None) -> Any:
        """Bus power to deliver P_mech through motor + ESC [W]."""
        motor = self.prop_systems[arch].motor
        P_rated = motor.P_max if P_rated is None else P_rated
        return rp.drive_electrical_power(P_mech, P_rated, motor.eta_max,
                                         motor.k_loss, self.tech.esc_eta)

    # ── Atmosphere (inlined, differentiable) ─────────────────────────────

    @staticmethod
    def _density(altitude_m: Any) -> Any:
        """ISA density with a smoothed temperature floor."""
        return rp.isa_density_smooth(altitude_m)

    def _ice_available(self, arch: str, r_conv: Any, altitude_m: Any) -> Any:
        ice = self.prop_systems[arch].ice
        sigma = self._density(altitude_m) / 1.225
        return r_conv * sigma ** ice.altitude_derating_exp

    @staticmethod
    def _gt_available(r_conv: Any, altitude_m: Any) -> Any:
        T_loc = smooth_max(288.15 - 0.0065 * altitude_m, 200.0, scale=288.15)
        rho = 1.225 * (T_loc / 288.15) ** 4.255877
        return r_conv * (rho / 1.225) * smooth_min((288.15 / T_loc) ** 0.5, 1.1)

    def _willans(self, arch: str, P_shaft: Any, r_conv: Any) -> Any:
        """Engine fuel flow [kg/s], Willans line on the engine's own rating."""
        ice = self.prop_systems[arch].ice
        b = ice.BSFC_rated / 3.6e9                      # g/kWh -> kg/J
        a = ice.idle_fuel_frac * b * r_conv
        return a + b * P_shaft

    # ── Per-architecture power split ─────────────────────────────────────

    def arch_power_split(self, arch: str, P_demand: Any, k_electric: Any,
                         altitude_m: Any, ratings: Optional[Dict[str, Any]] = None,
                         P_lift: Any = 0.0) -> Dict[str, Any]:
        """
        Power split for ONE architecture.

        Parameters
        ----------
        P_demand : shaft power at the cruise propulsor [W]
        k_electric : battery share of the bus (electric drivetrains) or of
            the shaft (parallel / series-parallel)
        altitude_m : altitude for converter derating
        ratings : {'motor', 'lift', 'conv'} — cruise-motor shaft rating,
            lift-motor shaft rating, converter rating (sea-level shaft W for
            engines and turbines, net electrical W for a fuel cell)
        P_lift : lift-rotor shaft power on the same bus [W]

        Returns ``P_elec_from_bus`` (battery draw at the terminals),
        ``fuel_flow_kg_s``, ``heat_total``, ``P_conv_out`` (power the fuel
        converter delivered) and ``P_bus_motor`` (total drive demand).
        """
        ps = self.prop_systems[arch]
        t = self.tech
        r = self._resolve(arch, ratings)
        k = _clamp01(k_electric)
        zero = 0.0 * P_demand + 0.0 * k

        P_lift = P_lift + zero
        bus_lift = self._motor_electrical_power(arch, P_lift, r["lift"])
        fuel_flow = zero
        P_conv_out = zero
        conv_loss = zero

        if arch == "all_electric":
            # No fuel converter: the battery carries the whole demand,
            # whatever k says.
            P_bus = self._motor_electrical_power(arch, P_demand, r["motor"]) + bus_lift
            P_batt = P_bus
            P_motor_shaft = P_demand + P_lift

        elif arch in ("series", "turbo_electric", "fuel_cell"):
            P_bus = self._motor_electrical_power(arch, P_demand, r["motor"]) + bus_lift
            want = (1.0 - k) * P_bus
            switch = smooth_switch(want)
            r_c = r["conv"]

            if arch == "series":
                eta_gen = ps.generator.eta_rated if ps.generator else t.gen_eta
                avail = eta_gen * self._ice_available(arch, r_c, altitude_m)
                P_gen = smooth_min(want, avail, scale=r_c)
                P_engine = P_gen / eta_gen
                fuel_flow = switch * self._willans(arch, P_engine, r_c)
                lhv = ps.ice.fuel_lhv

            elif arch == "turbo_electric":
                gt = ps.gas_turbine
                eta_gen = ps.generator.eta_rated if ps.generator else t.gen_eta
                P_av = self._gt_available(r_c, altitude_m)
                P_gen = smooth_min(want, eta_gen * P_av, scale=r_c)
                P_engine = P_gen / eta_gen
                x = smooth_min(smooth_max(P_engine / (P_av + 1e-10), 0.05), 1.0)
                sfc_d = gt.SFC_design
                if r["_scaled"]:
                    # Scale effect: small turboshafts burn more per kWh.
                    sfc_d = t.gt_sfc_ref_g_kwh * (t.gt_P_ref_w / r_c) ** t.gt_sfc_scale_exp
                sfc = sfc_d * (gt.sfc_c0 + gt.sfc_c1 * x + gt.sfc_c2 * x ** 2) / x
                fuel_flow = switch * sfc * P_engine / 3.6e9
                lhv = gt.fuel_lhv

            else:  # fuel_cell
                fc = ps.fuel_cell
                if r["_scaled"]:
                    P_gen = smooth_min(want, r_c, scale=r_c)
                    P_eff = P_gen * (self._fc_P_ref / r_c)
                else:
                    # Legacy: demand applied to the reference stack directly.
                    P_gen = want
                    P_eff = P_gen
                eta_fc = fc_polarization.system_efficiency(fc, P_eff)
                fuel_flow = switch * P_gen / (eta_fc * fc.h2_lhv)
                P_engine = P_gen
                lhv = fc.h2_lhv

            P_conv_out = P_gen
            P_batt = P_bus - P_gen
            conv_loss = fuel_flow * lhv - P_gen
            P_motor_shaft = P_demand + P_lift

        elif arch == "parallel":
            r_c = r["conv"]
            want = (1.0 - k) * P_demand
            P_engine = smooth_min(want, self._ice_available(arch, r_c, altitude_m),
                                  scale=r_c)
            fuel_flow = smooth_switch(want) * self._willans(arch, P_engine, r_c)
            P_bus = self._motor_electrical_power(arch, P_demand - P_engine, r["motor"]) + bus_lift
            P_batt = P_bus
            P_conv_out = P_engine
            conv_loss = fuel_flow * ps.ice.fuel_lhv - P_engine
            P_motor_shaft = P_demand - P_engine + P_lift

        elif arch == "series_parallel":
            r_c = r["conv"]
            s = t.sp_mech_fraction
            eta_gen = ps.generator.eta_rated if ps.generator else t.gen_eta
            eta_ref = ps.motor.eta_max * t.esc_eta
            # Engine output that delivers the fuel share (1-k) of the shaft
            # demand through the mechanical path (s) and the electrical one
            # (1-s, via generator, bus and motor at nominal efficiency).
            want = (1.0 - k) * P_demand / (s + (1.0 - s) * eta_gen * eta_ref)
            P_engine = smooth_min(want, self._ice_available(arch, r_c, altitude_m),
                                  scale=r_c)
            fuel_flow = smooth_switch(want) * self._willans(arch, P_engine, r_c)
            P_direct = s * P_engine
            P_gen = (1.0 - s) * eta_gen * P_engine
            P_bus = self._motor_electrical_power(arch, P_demand - P_direct, r["motor"]) + bus_lift
            # Generator output beyond what the motor draws is throttled back,
            # not stored: regeneration into the pack is not modelled.
            P_batt = smooth_max(P_bus - P_gen, 0.0, scale=r["motor"])
            P_conv_out = P_engine
            conv_loss = fuel_flow * ps.ice.fuel_lhv - P_direct - P_gen
            P_motor_shaft = P_demand - P_direct + P_lift

        else:
            raise ValueError(f"Unknown architecture: {arch!r}")

        heat_drive = P_bus - P_motor_shaft
        return {
            "P_elec_from_bus": P_batt,
            "fuel_flow_kg_s": fuel_flow,
            "heat_total": heat_drive + conv_loss,
            "P_conv_out": P_conv_out,
            "P_bus_motor": P_bus,
        }

    # ── Blending across architectures ────────────────────────────────────

    def blend_power_split(self, weights: np.ndarray, P_demand: Any,
                          k_electric: Any, altitude_m: Any,
                          ratings: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
                          P_lift: Any = 0.0) -> Dict[str, Any]:
        """Evaluate all six architectures and blend the outputs by ``weights``."""
        ratings = ratings or [None] * len(self.ARCH_NAMES)
        results = [self.arch_power_split(a, P_demand, k_electric, altitude_m,
                                         ratings=ratings[i], P_lift=P_lift)
                   for i, a in enumerate(self.ARCH_NAMES)]
        out = {"P_demand": P_demand}
        for key in ("P_elec_from_bus", "fuel_flow_kg_s", "heat_total", "P_conv_out"):
            out[key] = sum(w * res[key] for w, res in zip(weights, results))
        return out

    def arch_fuel_efficiency(self, arch: str, P_shaft_ref: Any, altitude_m: Any,
                             eta_fallback: float = 0.30,
                             ratings: Optional[Dict[str, Any]] = None) -> Any:
        """
        Shaft energy per unit of fuel chemical energy, fuel path alone
        (k = 0), at the reference shaft power [-].

        Now a DIAGNOSTIC. The sizing model no longer converts a tank into
        shaft energy through this number; it integrates the architecture's
        own fuel flow over the mission. Where the converter saturates the
        battery tops up the shortfall, so this then overstates the fuel
        path; the ratings rule keeps cruise below saturation.
        """
        if arch == "all_electric":
            return eta_fallback + 0.0 * P_shaft_ref
        split = self.arch_power_split(arch, P_shaft_ref, 0.0, altitude_m, ratings=ratings)
        P_chem = split["fuel_flow_kg_s"] * self.FUEL_LHV[self.ARCH_NAMES.index(arch)]
        return P_shaft_ref / (P_chem + 1e-12)

    def blend_fuel_efficiency(self, weights: np.ndarray, P_shaft_ref: Any,
                              altitude_m: Any, eta_fallback: float = 0.30,
                              ratings: Optional[Sequence[Optional[Dict[str, Any]]]] = None) -> Any:
        """
        Blended fuel-path efficiency, weighted by w_i*LHV_i so that
        ``m * blend_fuel_lhv * blend_fuel_efficiency == m * sum w_i LHV_i eta_i``.
        """
        ratings = ratings or [None] * len(self.ARCH_NAMES)
        etas = [self.arch_fuel_efficiency(a, P_shaft_ref, altitude_m, eta_fallback,
                                          ratings=ratings[i])
                for i, a in enumerate(self.ARCH_NAMES)]
        wl = weights * self.FUEL_LHV
        return sum(x * e for x, e in zip(wl, etas)) / np.sum(wl)

    def blend_propulsion_mass(self, weights: np.ndarray) -> Any:
        """Blended preset propulsion dry mass [kg] (legacy presets)."""
        masses = np.array([self.prop_systems[a].total_mass for a in self.ARCH_NAMES])
        return np.sum(weights * masses)

    def blend_fuel_lhv(self, weights: np.ndarray) -> Any:
        """Blended fuel lower heating value [J/kg]."""
        return np.sum(weights * self.FUEL_LHV)
