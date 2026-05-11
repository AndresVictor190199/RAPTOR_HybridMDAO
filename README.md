# HybridPropulsion_Raptor v0.1.0

**Energy Optimization for Hybrid Propulsion Systems in Transition VTOL UAVs**

Built on the [RAPTOR](https://github.com/VAero-Lab/RAPTOR) path-planning framework, this package implements multi-source energy optimization for hybrid-electric VTOL aircraft across 5 propulsion architectures.

## Supported Architectures

| Architecture | Power Path | Best For |
|---|---|---|
| **Series** | ICE → Generator → Battery → Motor → Prop | Decoupled ICE sizing |
| **Parallel** | ICE + Motor → shared shaft | Mechanical efficiency |
| **Series-Parallel** | ICE split: mechanical + electrical | Flexibility |
| **Turbo-Electric** | Turbine → Generator → Motor → Prop | High power density |
| **Fuel Cell** | H₂ FC → Battery → Motor → Prop | Zero emissions |

## Installation

```bash
pip install -e .                # Development (editable)
pip install -e ".[mdao]"        # With OpenMDAO support
pip install -e ".[all]"         # All dependencies
```

**Requirements:** Python >= 3.9, numpy, scipy, matplotlib

## Quick Start

```python
from hpraptor import *

# Create a series hybrid vehicle (50 kg MTOW)
vehicle = series_hybrid_config(m_tow=50.0)

# Build a flight path
path = FlightPath(0.0, 0.0, 0.0, 0.01, 0.0, 0.0)
path.add_segment(VTOLAscend(altitude_gain=100, climb_rate=3.0))
path.add_segment(Transition())
path.add_segment(FWClimb(altitude_gain=200, climb_angle_deg=8, airspeed=25))
path.add_segment(FWCruise(ground_distance=5000, airspeed=30))
path.add_segment(FWDescend(altitude_loss=250, descent_angle_deg=6, airspeed=28))
path.add_segment(Transition())
path.add_segment(VTOLDescend(altitude_loss=80, descent_rate=2.5))

# Analyze hybrid energy
manager = HybridEnergyManager(vehicle)
result = manager.analyze_path(path)

print(f"Total fuel consumed: {result.total_fuel_consumed_kg:.2f} kg")
print(f"Battery SOC final:  {result.SOC_final*100:.1f}%")
print(f"Flight time:        {result.total_time/60:.1f} min")
print(f"Mass reduction:     {result.mass_initial_kg - result.mass_final_kg:.2f} kg")
```

## Package Structure

```
HybridPropulsion_Raptor/
├── hpraptor/                    # Core package
│   ├── atmosphere.py            # ISA standard atmosphere
│   ├── config.py                # UAV config + propulsion modes
│   ├── segments.py              # 6 flight segment types
│   ├── path.py                  # FlightPath construction
│   ├── propulsion_system.py     # Motor, ICE, Generator, FC, Turbine, Propeller
│   ├── battery_model.py         # Multi-chemistry battery model
│   ├── fuel_model.py            # Fuel consumption & tank model
│   ├── vehicles.py              # 5 vehicle configurations
│   ├── hybrid_energy.py         # Multi-source energy manager
│   ├── terrain.py               # Terrain clearance (optional)
│   ├── dem.py                   # DEM interface (optional)
│   └── builder.py               # Path builder
├── hpraptor_mdao/               # OpenMDAO MDAO (Phase 3)
├── examples/                    # Demo scripts
├── scripts/                     # Analysis scripts
├── data/                        # Vehicle configs, propulsion maps
└── tests/                       # Unit tests
```

## Propulsion Component Models

All models use generic parametric forms from published literature:

- **Electric Motor**: Parabolic efficiency η(P/P_rated) — Finger et al. (2020)
- **ICE**: Willans line BSFC model with altitude derating — Bowman et al. (2018)
- **Generator**: Constant-efficiency with off-design correction
- **Fuel Cell**: PEM polarization curve — Larminie & Dicks (2003)
- **Gas Turbine**: Polynomial SFC model — Kurzke (2015)
- **Propeller**: Parametric CT/CP from BEM — McCrink & Gregory (2017)
- **Battery**: Multi-chemistry with C-rate and temperature effects

## Roadmap

- [x] Phase 1: Foundation (atmosphere, segments, path, config)
- [x] Phase 2: Hybrid propulsion physics (5 architectures)
- [ ] Phase 3: OpenMDAO MDAO integration
- [ ] Phase 4: Visualization & analysis tools

## License

MIT — See LICENSE file.
