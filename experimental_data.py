"""Experimental data for the genuine resolution manuscript.

The numerical routines live in ``Resolution.py``. This file only contains the
manuscript data so users can replace these constants with their own intensities,
click-probability table, observed guessing probabilities, and certification
cases.
"""

from __future__ import annotations


# Coherent-state mean photon numbers used in the experiment.
INTENSITIES = (0.0, 1.1574, 2.3386, 3.4774, 4.6532, 6.9896, 7.9741)


# Rounded click-probability table reported in the manuscript.
# Rows follow INTENSITIES. Columns are p(0|mu), ..., p(7|mu), p(>=8|mu).
CLICK_PROBABILITIES = (
    (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.3620, 0.3740, 0.1880, 0.0599, 0.0141, 0.00232, 0.0005, 0.0, 0.0),
    (0.1380, 0.2810, 0.2800, 0.1780, 0.0829, 0.0291, 0.0081, 0.0019, 0.0005),
    (0.0533, 0.1650, 0.2450, 0.2350, 0.1610, 0.0860, 0.0370, 0.0130, 0.0047),
    (0.0120, 0.0820, 0.1700, 0.2196, 0.2070, 0.1510, 0.0870, 0.0420, 0.0236),
    (0.0030, 0.0205, 0.0648, 0.1260, 0.1790, 0.1940, 0.1670, 0.1200, 0.1250),
    (0.0015, 0.0110, 0.0390, 0.0890, 0.1420, 0.1800, 0.1800, 0.1500, 0.2070),
)


# Exact observed guessing probabilities used in the certification and summary
# tables. These are kept separate because recomputing from the rounded
# CLICK_PROBABILITIES changes the last printed digits.
OBSERVED_GUESSING_PROBABILITIES = {
    2: 0.8190,
    3: 0.6516072975,
    4: 0.5475440031,
    5: 0.4797908742,
    6: 0.4503133164,
    7: 0.404025137,
}


# Certification rows as (num_inputs, max_photons, resolution, intensity_cap).
# Use intensity_cap=None to take the largest intensity among the selected inputs,
# which must satisfy the subspace-witness condition I <= max_photons + 1.
CERTIFICATION_CASES = (
    (2, 1, 1, None),
    (3, 2, 2, None),
    (3, 3, 2, None),
    (4, 4, 3, None),
    (5, 5, 3, None),
    (6, 6, 3, None),
    (7, 7, 3, None),
    (7, 8, 3, None),
)
