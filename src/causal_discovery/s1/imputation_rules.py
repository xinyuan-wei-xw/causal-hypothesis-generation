"""Logical constraints for approved MI candidates; no imputation model is fitted here."""
import numpy as np
import pandas as pd


def additional_count_bounds(sub2, sub3):
    """Bounds on the two recorded additional slots; missing is never explicit none.

    Accept harmonized source codes 1..19 and missing values, not raw -9 sentinels.
    Source series are never mutated.
    """
    if not sub2.index.equals(sub3.index) or not sub2.index.is_unique:
        raise ValueError('Slot indices must be identical and unique.')
    slots = pd.concat([sub2, sub3], axis=1)
    for s in (sub2, sub3):
        if not s.dropna().isin(range(1, 20)).all():
            raise ValueError('Expected harmonized codes 1..19 or missing.')
    known = slots.gt(1).sum(axis=1).astype('int64')
    unknown = slots.isna().sum(axis=1).astype('int64')
    return pd.DataFrame({'lower': known, 'upper': known + unknown}, index=sub2.index)


def draw_constrained_counts(probabilities, bounds, *, rng):
    """Draw from supplied category probabilities restricted to feasible totals.

    probabilities: indexed DataFrame with columns [0, 1, 2]. This function neither
    estimates probabilities nor approves a model. Zero feasible mass for an
    uncertain count is an error, not a fallback to uniform or a constant.
    Fully known counts are deterministic regardless of model probabilities.
    """
    if (not probabilities.index.is_unique or not probabilities.index.equals(bounds.index)
            or list(probabilities.columns) != [0, 1, 2]):
        raise ValueError('Aligned unique indices and probability columns [0,1,2] required.')
    p = probabilities.to_numpy(dtype=float, copy=True)
    b = bounds[['lower', 'upper']].to_numpy(dtype=float)
    if (not np.isfinite(p).all() or (p < 0).any()
            or not np.allclose(p.sum(axis=1), 1)):
        raise ValueError('Finite nonnegative probabilities summing to one required.')
    if (not np.isfinite(b).all() or not np.equal(b, np.floor(b)).all()
            or (b < 0).any() or (b > 2).any() or (b[:, 0] > b[:, 1]).any()):
        raise ValueError('Bounds must be ordered integers in 0..2.')
    fixed = b[:, 0] == b[:, 1]
    allowed = (np.arange(3)[None, :] >= b[:, :1]) & (np.arange(3)[None, :] <= b[:, 1:])
    p *= allowed
    mass = p.sum(axis=1)
    if (mass[~fixed] <= 0).any():
        raise ValueError('No model probability on feasible uncertain totals; review support.')
    result = b[:, 0].astype('int64')
    if (~fixed).any():
        q = p[~fixed] / mass[~fixed, None]
        u = rng.random(len(q))
        result[~fixed] = (u[:, None] >= np.cumsum(q, axis=1)).sum(axis=1)
    return pd.Series(result, index=bounds.index, name='additional_substances_total')


def conditional_detail_plan(parent, detail, completed_parent, *, applicable_code,
                            parent_codes, detail_codes):
    """Plan within-draw detail handling, stopping on source cases requiring review.

    The parent completion is supplied by the caller. Returns statuses and source
    details, but does not fit/draw details or conflate structural N/A with unknown.
    """
    if (not parent.index.is_unique or not parent.index.equals(detail.index)
            or not parent.index.equals(completed_parent.index)):
        raise ValueError('Aligned unique indices required.')
    if (not parent.dropna().isin(parent_codes).all()
            or not detail.dropna().isin(detail_codes).all()
            or completed_parent.isna().any()
            or not completed_parent.isin(parent_codes).all()
            or applicable_code not in parent_codes):
        raise ValueError('Invalid source/completed domains or incomplete parent.')
    observed = parent.notna()
    if not completed_parent.loc[observed].eq(parent.loc[observed]).all():
        raise ValueError('Observed parent values must be preserved.')
    review = detail.notna() & (parent.isna() | parent.ne(applicable_code))
    if review.any():
        raise ValueError(f'{int(review.sum())} observed-detail records need separate source review.')
    applies = completed_parent.eq(applicable_code)
    status = pd.Series('not_applicable', index=parent.index, dtype='string')
    status.loc[applies & detail.notna()] = 'observed_detail'
    status.loc[applies & detail.isna()] = 'impute_detail'
    return pd.DataFrame({'status': status, 'observed_detail': detail.copy()}, index=parent.index)
