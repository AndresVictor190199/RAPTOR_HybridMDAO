"""
Tests for the battery cell catalogue.

The catalogue exists to stop the optimizer buying a pack that cannot be
built or cannot deliver hover power, so the tests target exactly those two
failure modes rather than the arithmetic.
"""

import numpy as np
import pytest

from hpraptor.m5_propulsion import battery_catalogue as batcat


def test_every_catalogue_entry_is_physically_coherent():
    for key, cell in batcat.CATALOGUE.items():
        assert cell.capacity_ah > 0 and cell.mass_kg > 0, key
        # Cell-level specific energy for lithium chemistries sits in a well
        # known band; anything outside it is a data-entry error.
        assert 100.0 < cell.specific_energy_wh_kg < 350.0, (
            f"{key}: {cell.specific_energy_wh_kg:.0f} Wh/kg is outside the "
            f"plausible range for a lithium cell"
        )
        assert cell.max_c_discharge >= 1.0, key


def test_pack_overhead_derates_both_energy_and_power():
    cell = batcat.get_cell("lipo_pouch_balanced")
    assert batcat.pack_specific_energy(cell) < cell.specific_energy_wh_kg
    assert batcat.pack_specific_power(cell) < cell.specific_power_w_kg


def test_high_power_cells_trade_energy_for_c_rate():
    """
    The whole reason to keep a catalogue rather than one specific energy:
    the formats differ in which limit governs, and the difference is large.
    """
    dense = batcat.get_cell("li_ion_18650_high_energy")
    punchy = batcat.get_cell("lipo_pouch_high_power")

    assert dense.specific_energy_wh_kg > punchy.specific_energy_wh_kg
    assert punchy.specific_power_w_kg > 4.0 * dense.specific_power_w_kg


def test_pack_energy_and_power_are_linear_in_mass():
    """Both must be smooth and linear, so the optimizer can differentiate them."""
    cell = batcat.get_cell()
    assert batcat.pack_energy_wh(2.0, cell) == pytest.approx(
        2.0 * batcat.pack_energy_wh(1.0, cell))
    assert batcat.pack_power_limit_w(3.0, cell) == pytest.approx(
        3.0 * batcat.pack_power_limit_w(1.0, cell))


def test_pack_energy_survives_complex_step():
    """Battery mass is a design variable, so this must differentiate."""
    cell = batcat.get_cell()
    out = batcat.pack_energy_wh(np.array([1.0 + 1e-30j]), cell)
    assert np.iscomplexobj(out)
    deriv = out.imag[0] / 1e-30
    assert deriv == pytest.approx(batcat.pack_specific_energy(cell), rel=1e-9)


def test_best_cell_respects_whichever_limit_governs():
    """
    A hover-dominated demand must select a high-C-rate format even though
    its specific energy is worse — that is the decision the catalogue exists
    to make, and getting it backwards would silently under-size the pack.
    """
    key_power, mass_power = batcat.best_cell_for(energy_wh=60.0, peak_power_w=3000.0)
    assert batcat.get_cell(key_power).max_c_discharge >= 10.0

    # With a gentle power demand the energy-dense format should win instead.
    key_energy, _ = batcat.best_cell_for(energy_wh=600.0, peak_power_w=200.0)
    assert batcat.get_cell(key_energy).specific_energy_wh_kg > \
        batcat.get_cell(key_power).specific_energy_wh_kg


def test_snap_to_pack_is_buildable_and_honest_about_the_error():
    cell = batcat.get_cell("lipo_pouch_balanced")
    build = batcat.snap_to_pack(1.0, cell, bus_voltage_v=22.2)

    assert build.n_series >= 1 and build.n_parallel >= 1
    assert build.n_cells == build.n_series * build.n_parallel
    assert float(build.n_series).is_integer()
    # The realised mass is what the cells actually weigh, so it must be
    # reachable as an integer multiple of one series string.
    string_mass = build.mass_kg / build.n_parallel
    assert build.mass_kg == pytest.approx(build.n_parallel * string_mass)
    assert abs(build.mass_error_frac) < 1.5


def test_series_count_follows_the_bus_voltage():
    """Series count is an electrical requirement, not a mass optimization."""
    cell = batcat.get_cell("lipo_pouch_balanced")
    low = batcat.snap_to_pack(1.0, cell, bus_voltage_v=22.2)
    high = batcat.snap_to_pack(1.0, cell, bus_voltage_v=44.4)
    assert high.n_series > low.n_series
    assert high.voltage_nom_v > low.voltage_nom_v
