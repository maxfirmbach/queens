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
"""Acquisition functions for Bayesian optimization."""

from scipy.stats import norm


def expected_improvement(
    mean: float,
    standard_deviation: float,
    best_objective: float,
    exploration_parameter: float = 0.0,
) -> float:
    r"""Compute the expected improvement for a minimization problem.

    Expected improvement measures the improvement over the best observed
    objective value that can be expected when evaluating a candidate point.
    It accounts for both the predicted objective value and the uncertainty of
    the surrogate model.

    The expected improvement is defined as

    .. math::
        \operatorname{EI}(x) = \Delta(x)\Phi(z(x)) + \sigma(x)\phi(z(x)),
    where
    .. math::
        \Delta(x) = f_{\min} - \mu(x) - \xi, \qquad z(x) = \frac{\Delta(x)}{\sigma(x)}.

    Here, :math:`\mu(x)` and :math:`\sigma(x)` are the predictive mean and
    standard deviation, :math:`f_{\min}` is the best objective value observed
    so far, and :math:`\xi` is the exploration parameter. The functions
    :math:`\Phi` and :math:`\phi` denote the cumulative distribution function
    and probability density function of the standard normal distribution.

    References:
        .. [1] Jones, D. R., Schonlau, M., and Welch, W. J. (1998).
           "Efficient Global Optimization of Expensive Black-Box Functions."
           Journal of Global Optimization, 13, 455–492.
           https://doi.org/10.1023/A:1008306431147

    Args:
        mean:
            Predictive mean of the objective at the candidate point.
        standard_deviation:
            Predictive standard deviation at the candidate point.
        best_objective:
            Smallest objective value observed so far.
        exploration_parameter:
            Nonnegative parameter controlling exploration. Larger values make
            improvement more difficult to achieve and generally encourage
            exploration. Defaults to ``0.0``.

    Returns:
        float:
            Expected improvement at the candidate point. Larger values indicate
            more promising candidates.

    Note:
        If the predictive standard deviation is effectively zero, the
        acquisition value is the deterministic improvement

        .. math::
            \max(f_{\min} - \mu(x) - \xi, 0).

        Expected improvement is normally maximized. If it is passed to an
        optimizer that minimizes its objective, its negative should be used.
    """
    improvement = best_objective - mean - exploration_parameter
    standardized_improvement = improvement / standard_deviation

    return float(
        improvement * norm.cdf(standardized_improvement)
        + standard_deviation * norm.pdf(standardized_improvement)
    )


def probability_of_improvement(
    mean: float,
    standard_deviation: float,
    best_objective: float,
    exploration_parameter: float = 0.0,
) -> float:
    r"""Compute the probability of improvement for a minimization problem.

    Probability of improvement measures the probability that evaluating a
    candidate point produces an objective value smaller than the best value
    observed so far by at least the exploration parameter.

    It is defined as

    .. math::
        \operatorname{PI}(x) = \Phi\left(\frac{f_{\min} - \mu(x) - \xi}{\sigma(x)}\right),

    where :math:`\mu(x)` and :math:`\sigma(x)` are the predictive mean and
    standard deviation, :math:`f_{\min}` is the best objective value observed
    so far, and :math:`\xi` is the exploration parameter.

    Unlike expected improvement, probability of improvement considers only the
    probability of obtaining an improvement, not its possible magnitude.

    References:
        .. [1] Kushner, H. J. (1964).
           "A New Method of Locating the Maximum Point of an Arbitrary Multipeak
           Curve in the Presence of Noise."
           Journal of Basic Engineering, 86(1), 97–106.
           https://doi.org/10.1115/1.3653121

    Args:
        mean:
            Predictive mean of the objective at the candidate point.
        standard_deviation:
            Predictive standard deviation at the candidate point.
        best_objective:
            Smallest objective value observed so far.
        exploration_parameter:
            Nonnegative parameter controlling exploration. Larger values
            require a candidate to outperform the current best value by a
            larger margin. Defaults to ``0.0``.

    Returns:
        float:
            Probability of improvement at the candidate point, between zero and
            one. Larger values indicate more promising candidates.

    Note:
        If the predictive standard deviation is effectively zero, this function
        returns ``1.0`` if the deterministic prediction satisfies the required
        improvement and ``0.0`` otherwise.

        Probability of improvement is normally maximized. If it is passed to an
        optimizer that minimizes its objective, its negative should be used.
    """
    improvement = best_objective - mean - exploration_parameter
    standardized_improvement = improvement / standard_deviation

    return float(norm.cdf(standardized_improvement))


def upper_confidence_bound(
    mean: float,
    standard_deviation: float,
    exploration_parameter: float = 2.0,
) -> float:
    r"""Compute a confidence-bound acquisition value for maximization.

    This function returns the upper confidence bound:

    .. math::
        \operatorname{UCB}_{\min}(x)
        = -\mu(x) + \kappa \sigma(x).

    Maximizing this acquisition function favors candidates with either a small
    predicted objective value or large predictive uncertainty.

    Args:
        mean:
            Predictive mean of the objective at the candidate point.
        standard_deviation:
            Predictive standard deviation at the candidate point.
        exploration_parameter:
            Nonnegative parameter controlling exploration. Larger values give
            more weight to uncertain candidates. Defaults to ``2.0``.

    Returns:
        float:
            Acquisition value. Larger values indicate more promising candidates.
    """
    return float(-mean + exploration_parameter * standard_deviation)
