"""
Battery Cell Catalogue — What You Can Actually Buy
====================================================

Battery mass is a design variable, but it is not a continuous one in
reality: a pack is an integer number of cells of some standard format, in
series and parallel. This module supplies the bridge between the two.

During optimization the mass stays continuous, because SLSQP needs a
smooth design space. Afterwards `snap_to_pack` finds the nearest pack that
can actually be built from a real cell format and reports how far the
continuous optimum was from it. A design that cannot be realised within a
few percent is telling you the optimizer found something the catalogue
cannot supply.

The cell entries are **format archetypes, not part numbers**: representative
specifications for the 18650, 21700 and LiPo pouch formats at the cell
level. They carry the specific energy and C-rate that distinguish those
formats from one another, which is what the sizing decision turns on. Swap
in a vendor datasheet when you have one — that is the point of keeping this
in a table rather than as constants inside a component.

Two properties matter to the optimizer:

  * specific ENERGY  [Wh/kg] — how much mission the pack can carry
  * specific POWER   [W/kg]  — whether it can sustain the hover peak

A pack sized purely on energy can be physically incapable of delivering
hover power, which is why `power_margin` exists and why the sizing problem
carries it as a constraint rather than assuming it away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from hpraptor.m5_propulsion.battery_model import CellChemistry

#: Fraction of pack mass that is structure, wiring, BMS and cooling rather
#: than cells. 15% is a common figure for small UAV packs.
DEFAULT_PACK_OVERHEAD = 0.15


@dataclass(frozen=True)
class CellSpec:
    """
    One cell format at the cell level (before pack overhead).

    Attributes
    ----------
    capacity_ah, voltage_nom_v, mass_kg
        The three numbers a datasheet leads with.
    max_c_discharge
        Continuous discharge rate the cell tolerates, in multiples of
        capacity. This is what limits a pack's power, and it is where the
        high-energy and high-power formats genuinely differ.
    """

    name: str
    chemistry: CellChemistry
    capacity_ah: float
    voltage_nom_v: float
    mass_kg: float
    max_c_discharge: float
    note: str = ""

    @property
    def energy_wh(self) -> float:
        """Usable energy of one cell."""
        return self.capacity_ah * self.voltage_nom_v

    @property
    def specific_energy_wh_kg(self) -> float:
        """Cell-level specific energy, before pack overhead."""
        return self.energy_wh / self.mass_kg

    @property
    def max_power_w(self) -> float:
        """Continuous power one cell can deliver at its C-rate limit."""
        return self.capacity_ah * self.max_c_discharge * self.voltage_nom_v

    @property
    def specific_power_w_kg(self) -> float:
        """Cell-level specific power, before pack overhead."""
        return self.max_power_w / self.mass_kg


#: Representative cell formats. Energy-dense cylindricals cannot deliver
#: the C-rate a multirotor hover peak demands; pouch cells can, at a third
#: less energy per kilogram. That trade is the reason to keep a catalogue.
CATALOGUE: Dict[str, CellSpec] = {
    "li_ion_18650_high_energy": CellSpec(
        name="18650 cylindrical, high-energy Li-ion",
        chemistry=CellChemistry.LI_ION_NMC,
        capacity_ah=3.5, voltage_nom_v=3.6, mass_kg=0.048, max_c_discharge=2.0,
        note="Best Wh/kg, poorest C-rate. Energy-limited missions only.",
    ),
    "li_ion_21700_balanced": CellSpec(
        name="21700 cylindrical, balanced Li-ion",
        chemistry=CellChemistry.LI_ION_NMC,
        capacity_ah=5.0, voltage_nom_v=3.6, mass_kg=0.070, max_c_discharge=3.0,
        note="The usual compromise for long-endurance electric UAVs.",
    ),
    "lipo_pouch_high_power": CellSpec(
        name="LiPo pouch, high-power",
        chemistry=CellChemistry.LIPO,
        capacity_ah=5.0, voltage_nom_v=3.7, mass_kg=0.105, max_c_discharge=25.0,
        note="Hover-capable C-rate at a large energy penalty.",
    ),
    "lipo_pouch_balanced": CellSpec(
        name="LiPo pouch, balanced",
        chemistry=CellChemistry.LIPO,
        capacity_ah=6.0, voltage_nom_v=3.7, mass_kg=0.108, max_c_discharge=10.0,
        note="Default for VTOL: enough C-rate for hover, decent Wh/kg.",
    ),
}

DEFAULT_CELL = "lipo_pouch_balanced"


def get_cell(key: str = DEFAULT_CELL) -> CellSpec:
    if key not in CATALOGUE:
        raise KeyError(f"Unknown cell {key!r}; catalogue has {sorted(CATALOGUE)}")
    return CATALOGUE[key]


# ═══════════════════════════════════════════════════════════════════════════
# CONTINUOUS PACK PROPERTIES  (what the optimizer sees)
# ═══════════════════════════════════════════════════════════════════════════

def pack_specific_energy(cell: CellSpec,
                         overhead: float = DEFAULT_PACK_OVERHEAD) -> float:
    """Pack-level Wh/kg — cell specific energy derated by pack overhead."""
    return cell.specific_energy_wh_kg * (1.0 - overhead)


def pack_specific_power(cell: CellSpec,
                        overhead: float = DEFAULT_PACK_OVERHEAD) -> float:
    """Pack-level W/kg at the cell's continuous C-rate limit."""
    return cell.specific_power_w_kg * (1.0 - overhead)


def pack_energy_wh(mass_kg, cell: CellSpec,
                   overhead: float = DEFAULT_PACK_OVERHEAD):
    """
    Energy stored in a pack of the given mass.

    Deliberately linear in mass and free of any branch, so it is safe to
    differentiate through with complex step.
    """
    return mass_kg * pack_specific_energy(cell, overhead)


def pack_power_limit_w(mass_kg, cell: CellSpec,
                       overhead: float = DEFAULT_PACK_OVERHEAD):
    """Continuous electrical power a pack of the given mass can deliver."""
    return mass_kg * pack_specific_power(cell, overhead)


# ═══════════════════════════════════════════════════════════════════════════
# DISCRETE REALISATION  (what you can actually build)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PackBuild:
    """A buildable pack, and how far it sits from the continuous optimum."""

    cell: CellSpec
    n_series: int
    n_parallel: int
    mass_kg: float
    energy_wh: float
    power_limit_w: float
    voltage_nom_v: float
    requested_mass_kg: float

    @property
    def n_cells(self) -> int:
        return self.n_series * self.n_parallel

    @property
    def mass_error_frac(self) -> float:
        """Signed fractional mass error against the continuous request."""
        return self.mass_kg / self.requested_mass_kg - 1.0

    def summary(self) -> str:
        return (
            f"  {self.cell.name}\n"
            f"    {self.n_series}S{self.n_parallel}P = {self.n_cells} cells, "
            f"{self.voltage_nom_v:.1f} V nominal\n"
            f"    mass {self.mass_kg:.3f} kg vs {self.requested_mass_kg:.3f} kg "
            f"requested ({self.mass_error_frac:+.1%})\n"
            f"    energy {self.energy_wh:.1f} Wh, "
            f"continuous power limit {self.power_limit_w:.0f} W"
        )


def snap_to_pack(
    mass_kg: float,
    cell: CellSpec,
    bus_voltage_v: float = 44.4,
    overhead: float = DEFAULT_PACK_OVERHEAD,
) -> PackBuild:
    """
    Nearest buildable pack to a continuous mass target.

    Series count is set by the bus voltage the powertrain expects — that is
    an electrical requirement, not something to optimize against mass — and
    the parallel count then absorbs the mass target. Both are integers, so
    the realised mass steps in increments of one series-string.
    """
    n_series = max(1, int(round(bus_voltage_v / cell.voltage_nom_v)))
    cell_mass_with_overhead = cell.mass_kg / (1.0 - overhead)
    string_mass = n_series * cell_mass_with_overhead

    n_parallel = max(1, int(round(mass_kg / string_mass)))
    realised_mass = n_parallel * string_mass
    n_cells = n_series * n_parallel

    return PackBuild(
        cell=cell,
        n_series=n_series,
        n_parallel=n_parallel,
        mass_kg=realised_mass,
        energy_wh=n_cells * cell.energy_wh,
        power_limit_w=n_cells * cell.max_power_w,
        voltage_nom_v=n_series * cell.voltage_nom_v,
        requested_mass_kg=mass_kg,
    )


def best_cell_for(
    energy_wh: float,
    peak_power_w: float,
    overhead: float = DEFAULT_PACK_OVERHEAD,
) -> Tuple[str, float]:
    """
    Lightest catalogue cell that meets both an energy and a power demand.

    A pack must satisfy BOTH: sized on energy alone it may not sustain the
    hover peak, and sized on power alone it may not finish the mission. The
    governing mass is the larger of the two, and the best cell is whichever
    format minimises that.

    Returns (catalogue key, required pack mass in kg).
    """
    best_key, best_mass = None, np.inf
    for key, cell in CATALOGUE.items():
        m_energy = energy_wh / pack_specific_energy(cell, overhead)
        m_power = peak_power_w / pack_specific_power(cell, overhead)
        m = max(m_energy, m_power)
        if m < best_mass:
            best_key, best_mass = key, m
    return best_key, float(best_mass)


def catalogue_table() -> str:
    """Human-readable catalogue, for reports and CLI output."""
    lines = [
        f"{'key':<26s}{'Wh/kg':>8s}{'W/kg':>9s}{'C-max':>7s}  note",
        "-" * 96,
    ]
    for key, c in CATALOGUE.items():
        lines.append(
            f"{key:<26s}{pack_specific_energy(c):8.0f}{pack_specific_power(c):9.0f}"
            f"{c.max_c_discharge:7.0f}  {c.note}"
        )
    return "\n".join(lines)
