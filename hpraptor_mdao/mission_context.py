"""
The m1 → MDO bridge.

Until now the sizing problem was disconnected from the mission it was
supposedly sizing for: `build_problem` took a hardcoded 2900 m altitude, a
95 s VTOL allowance and a 50 km range, none of which came from the corridor
in the YAML or the terrain under it. The optimizer therefore could not know
that the route crosses a 3026 m ridge.

TerrainContext closes that gap. It reads the mission's DEM once and hands
the optimizer the quantities that actually constrain the design:

  * how far it has to fly,
  * how high the terrain gets along the way,
  * how much clearance is required above it,
  * how much vertical travel the VTOL phases cost at each pad.

With those in hand, cruise altitude becomes a real design variable traded
against energy, and terrain clearance becomes a real constraint rather than
an assumption baked into a default.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np

from hpraptor.core.mission_loader import MissionDefinition


@dataclass
class TerrainContext:
    """Terrain-derived quantities the sizing problem needs from m1."""

    range_m: float                 # ground distance origin -> destination
    h_origin_m: float              # pad elevation AMSL
    h_dest_m: float
    h_terrain_max_m: float         # highest terrain on the direct route
    clearance_cruise_m: float      # required AGL in cruise
    clearance_min_m: float         # absolute minimum AGL anywhere
    vtol_climb_m: float            # vertical travel at the departure pad
    vtol_descent_m: float          # vertical travel at the arrival pad
    t_vtol_total_s: float          # total time in powered-lift flight
    dem_source: str                # provenance, so results stay traceable

    @property
    def h_cruise_min_m(self) -> float:
        """
        Lowest cruise altitude that clears the route's highest terrain.

        This is the terrain-clearance constraint expressed as a bound; the
        optimizer sees it as g5 so the margin is reported, not just enforced.
        """
        return self.h_terrain_max_m + self.clearance_cruise_m

    def to_dict(self) -> Dict[str, float]:
        d = asdict(self)
        d["h_cruise_min_m"] = self.h_cruise_min_m
        return d

    def summary(self) -> str:
        return "\n".join([
            f"  Terrain context (DEM source: {self.dem_source})",
            f"    range                {self.range_m / 1000:8.2f} km",
            f"    origin pad           {self.h_origin_m:8.1f} m AMSL",
            f"    destination pad      {self.h_dest_m:8.1f} m AMSL",
            f"    highest terrain      {self.h_terrain_max_m:8.1f} m AMSL",
            f"    cruise clearance     {self.clearance_cruise_m:8.1f} m AGL",
            f"    -> min cruise alt    {self.h_cruise_min_m:8.1f} m AMSL",
            f"    VTOL climb/descent   {self.vtol_climb_m:8.1f} / "
            f"{self.vtol_descent_m:.1f} m  ({self.t_vtol_total_s:.0f} s)",
        ])


def build_terrain_context(
    mission: MissionDefinition,
    dem_path: Optional[str] = None,
    dem_source: str = "auto",
    dem_resolution=None,
    n_profile: int = 400,
    verbose: bool = True,
) -> TerrainContext:
    """
    Derive the terrain context for a mission, building its DEM if needed.

    Pad elevations come from the DEM rather than the YAML's declared
    values, for the same reason run_mission anchors them: clearance checks
    and pad altitudes must reference one consistent terrain model.
    """
    from hpraptor.m1_mission.dem import DEMInterface

    if dem_path is None:
        from run_mission import _resolve_dem_path
        dem_path = _resolve_dem_path(mission, dem_source=dem_source,
                                     dem_resolution=dem_resolution)
    if dem_path is None:
        raise RuntimeError(
            "No DEM available for this mission, so terrain clearance cannot be "
            "constrained. Check the mission's dem_path and the DEM source."
        )

    dem = DEMInterface(dem_path)
    origin = (mission.origin.lat, mission.origin.lon)
    dest = (mission.destination.lat, mission.destination.lon)

    profile = dem.terrain_profile(origin, dest, n=n_profile)
    h_origin = float(dem.elevation(*origin))
    h_dest = float(dem.elevation(*dest))
    if not (np.isfinite(h_origin) and np.isfinite(h_dest)):
        raise RuntimeError("DEM has no data at one of the facility coordinates.")

    req = mission.requirements
    cons = mission.constraints

    # VTOL legs: climb high enough at the pad that the wingborne transition
    # STARTS with clearance margin in hand. Climbing only to the transition's
    # AGL floor leaves exactly zero margin at the first node, so any terrain
    # rise under the transition violates immediately and the trajectory is
    # infeasible from its own initial guess. Using the cruise clearance here
    # gives the transition twice its floor to work with.
    vtol_climb = float(cons.min_cruise_terrain_clearance)
    vtol_descent = float(cons.min_cruise_terrain_clearance)
    t_vtol = (vtol_climb / max(req.vtol_climb_rate_ms, 0.1)
              + vtol_descent / max(req.vtol_descent_rate_ms, 0.1))

    ctx = TerrainContext(
        range_m=float(profile["total_distance"]),
        h_origin_m=h_origin,
        h_dest_m=h_dest,
        h_terrain_max_m=float(np.nanmax(profile["elevations"])),
        clearance_cruise_m=float(cons.min_cruise_terrain_clearance),
        clearance_min_m=float(cons.min_terrain_clearance),
        vtol_climb_m=vtol_climb,
        vtol_descent_m=vtol_descent,
        t_vtol_total_s=float(t_vtol),
        dem_source=str(dem.metadata.source),
    )

    if verbose:
        print(ctx.summary())
    return ctx


#: Gaussian centres used to represent terrain along the route, and the
#: kernel width as a multiple of the centre spacing. 60 centres at 2x spacing
#: track the Quito->Cumbaya ridge to ~16 m while staying smooth; a polynomial
#: basis needed 175 m of slack to bound the same profile from above.
TERRAIN_N_CENTERS = 60
TERRAIN_SIGMA_FACTOR = 2.0


@dataclass
class TerrainModel:
    """
    A differentiable terrain(x) model the trajectory ODE can evaluate.

    Ground elevation as a sum of Gaussian bumps in ground distance:

        z(x) = sum_j  w_j * exp(-((x - c_j) / sigma)^2)   +   lift

    Complex-step safe (exp and sums pass complex through natively), smooth
    enough for SLSQP, and — after the lift — never below the real terrain.
    """

    weights: np.ndarray
    centers: np.ndarray
    sigma: float
    lift_m: float
    range_m: float
    terrain_max_m: float
    rms_conservatism_m: float
    max_conservatism_m: float
    is_upper_bound: bool

    def __call__(self, x):
        return evaluate_terrain(self, x)

    def summary(self) -> str:
        return "\n".join([
            f"  Terrain model: {len(self.centers)} Gaussian centres, "
            f"sigma {self.sigma:.0f} m",
            f"    lifted {self.lift_m:.1f} m to bound terrain from above "
            f"({'verified' if self.is_upper_bound else 'NOT VERIFIED'})",
            f"    conservative by {self.rms_conservatism_m:.1f} m RMS, "
            f"{self.max_conservatism_m:.1f} m worst case",
        ])


def evaluate_terrain(model: "TerrainModel", x):
    """
    Ground elevation at distance x along the route.

    Written to accept complex x so OpenMDAO's complex step can differentiate
    the clearance constraint through it.
    """
    x = np.asarray(x)
    d = (x[..., None] - model.centers) / model.sigma
    return np.sum(model.weights * np.exp(-d * d), axis=-1) + model.lift_m


def build_terrain_model(
    mission: MissionDefinition,
    dem_path: str,
    n_centers: int = TERRAIN_N_CENTERS,
    sigma_factor: float = TERRAIN_SIGMA_FACTOR,
    n_samples: int = 400,
    verbose: bool = True,
) -> TerrainModel:
    """
    Fit the terrain(x) model the trajectory's clearance constraint uses.

    The mission trajectory previously carried a single CONSTANT ground
    elevation per phase — for cruise, the mean of the two pads. On a route
    crossing a ridge 400 m above that mean, a 100 m AGL path constraint was
    satisfied by a trajectory flying straight through the mountain.

    The fitted model is LIFTED so it never sits below real terrain. A
    least-squares fit dips below a sharp ridge by construction, and
    under-predicting terrain is the one error mode that matters here: it
    would licence a trajectory to fly into rock. Raising the whole surface
    by the worst under-prediction makes it a conservative upper bound, so
    the constraint can only ever be too cautious, never too permissive.
    """
    from hpraptor.m1_mission.dem import DEMInterface

    dem = DEMInterface(dem_path)
    profile = dem.terrain_profile(
        (mission.origin.lat, mission.origin.lon),
        (mission.destination.lat, mission.destination.lon),
        n=n_samples,
    )
    x = np.asarray(profile["distances"], dtype=float)
    z = np.asarray(profile["elevations"], dtype=float)

    length = float(x[-1])
    centers = np.linspace(0.0, length, n_centers)
    sigma = sigma_factor * length / n_centers

    basis = np.exp(-((x[:, None] - centers[None, :]) / sigma) ** 2)
    weights, *_ = np.linalg.lstsq(basis, z, rcond=None)

    fitted = basis @ weights
    lift = max(float(np.max(z - fitted)), 0.0)
    residual = (fitted + lift) - z

    model = TerrainModel(
        weights=weights,
        centers=centers,
        sigma=sigma,
        lift_m=lift,
        range_m=length,
        terrain_max_m=float(np.nanmax(z)),
        rms_conservatism_m=float(np.sqrt(np.mean(residual ** 2))),
        max_conservatism_m=float(np.max(residual)),
        is_upper_bound=bool(np.all(residual >= -1e-6)),
    )
    if not model.is_upper_bound:
        raise RuntimeError(
            "Terrain model is not an upper bound on the real profile; the "
            "clearance constraint would be unsafe. Increase n_centers."
        )
    if verbose:
        print(model.summary())
    return model
