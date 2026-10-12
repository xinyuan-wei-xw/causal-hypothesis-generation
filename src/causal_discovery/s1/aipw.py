"""Score-only prototype for binary-outcome AIPW.

This module does not fit nuisance models, perform cross-fitting, integrate
imputation, select a population, or calculate clustered or pooled uncertainty.
Callers must supply predictions aligned by row. Valid numerical inputs alone
do not establish causal identification, overlap adequacy, or honest evaluation.
"""

import numpy as np


def aipw_scores(y, t, e, mu0, mu1):
    """Return binary-outcome AIPW scores for the treated-minus-control contrast.

    All inputs must be nonempty, finite, one-dimensional arrays of equal length.
    ``y`` and ``t`` are binary; ``e`` is the treatment probability strictly
    between zero and one; ``mu0`` and ``mu1`` are outcome probabilities in
    [0, 1]. No clipping, missing-value deletion, weighting, or index alignment
    is performed. In a cross-fitted analysis the caller must construct the
    appropriate out-of-fold nuisance predictions before calling this function.
    """
    arrays = {}
    for name, value in (("y", y), ("t", t), ("e", e), ("mu0", mu0), ("mu1", mu1)):
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if array.ndim != 1 or array.size == 0:
            raise ValueError(f"{name} must be a nonempty one-dimensional array")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")
        arrays[name] = array
    if len({array.size for array in arrays.values()}) != 1:
        raise ValueError("All inputs must have the same length")
    y, t, e, mu0, mu1 = (arrays[name] for name in ("y", "t", "e", "mu0", "mu1"))
    for name, array in (("y", y), ("t", t)):
        if not np.isin(array, [0, 1]).all():
            raise ValueError(f"{name} must be binary (0 or 1)")
    if not ((e > 0) & (e < 1)).all():
        raise ValueError("e must be strictly between 0 and 1; clipping is not automatic")
    for name, array in (("mu0", mu0), ("mu1", mu1)):
        if not ((array >= 0) & (array <= 1)).all():
            raise ValueError(f"{name} must be a probability in [0, 1]")
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        try:
            return mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
        except FloatingPointError as exc:
            raise ValueError("AIPW arithmetic overflow; inspect propensity support") from exc


def aipw_mean(y, t, e, mu0, mu1):
    """Return the unweighted mean score, without an inference procedure."""
    scores = aipw_scores(y, t, e, mu0, mu1)
    # Scale first so the summation does not overflow for finite large scores.
    return float(np.sum(scores / scores.size))
