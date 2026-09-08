#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024-2025, QUEENS contributors.
#
# This file is part of QUEENS.
#
# QUEENS is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version. QUEENS is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details. You
# should have received a copy of the GNU Lesser General Public License along with QUEENS. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Test acquisition functions for Bayesian optimization."""

import numpy as np

from queens.utils.acquisition_functions import (
    expected_improvement,
    probability_of_improvement,
    upper_confidence_bound,
)


def test_expected_improvement_known_value():
    """Test expected improvement against a known value."""
    value = expected_improvement(mean=0.8, standard_deviation=0.2, best_objective=1.0)
    expected = 0.2166630941172537

    np.testing.assert_allclose(value, expected, rtol=1e-10)


def test_expected_improvement_at_zero_standardized_improvement():
    """Test expected improvement when the standardized improvement is zero."""
    value = expected_improvement(mean=1.0, standard_deviation=0.3, best_objective=1.0)
    expected = 0.3 / np.sqrt(2.0 * np.pi)

    np.testing.assert_allclose(value, expected, rtol=1e-10)


def test_expected_improvement_reduced_by_exploration_parameter():
    """Test the effect of the expected improvement exploration threshold."""
    value_without_exploration = expected_improvement(
        mean=0.8,
        standard_deviation=0.2,
        best_objective=1.0,
        exploration_parameter=0.0,
    )
    value_with_exploration = expected_improvement(
        mean=0.8,
        standard_deviation=0.2,
        best_objective=1.0,
        exploration_parameter=0.1,
    )

    assert value_with_exploration < value_without_exploration


def test_probability_of_improvement_known_value():
    """Test probability of improvement against a known value."""
    value = probability_of_improvement(mean=0.8, standard_deviation=0.2, best_objective=1.0)
    expected = 0.8413447460685429

    np.testing.assert_allclose(value, expected, rtol=1e-10)


def test_probability_of_improvement_at_zero_standardized_improvement():
    """Test expected improvement when the standardized improvement is zero."""
    value = probability_of_improvement(mean=1.0, standard_deviation=0.3, best_objective=1.0)
    expected = 0.5

    np.testing.assert_allclose(value, expected, rtol=1e-10)


def test_probability_of_improvement_reduced_by_exploration_parameter():
    """Test the probability of improvement exploration threshold."""
    value_without_exploration = probability_of_improvement(
        mean=0.8,
        standard_deviation=0.2,
        best_objective=1.0,
        exploration_parameter=0.0,
    )
    value_with_exploration = probability_of_improvement(
        mean=0.8,
        standard_deviation=0.2,
        best_objective=1.0,
        exploration_parameter=0.1,
    )

    assert value_with_exploration < value_without_exploration


def test_upper_confidence_bound_known_value():
    """Test the upper confidence bound against a known value."""
    value = upper_confidence_bound(mean=1.0, standard_deviation=0.3, exploration_parameter=2.0)
    expected = -0.4

    np.testing.assert_allclose(value, expected, rtol=1e-10)
