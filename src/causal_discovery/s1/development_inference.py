"""Conditional score diagnostics, NOT validated cluster-DML or MI inference."""
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def state_score_uncertainty(scores, states, level=0.95):
    """CR1 intercept-only score variance with t(G-1) reference.

    Treats supplied scores as fixed estimated observations. Does not propagate
    nuisance fitting, imputation, selection, or sensitivity-parameter uncertainty.
    Episode folds overlapping states do not establish cluster-DML validity.
    """
    if not isinstance(scores, pd.Series) or not isinstance(states, pd.Series):
        raise ValueError('Aligned Series required')
    if not scores.index.is_unique or not scores.index.equals(states.index):
        raise ValueError('Unique exactly aligned IDs required')
    a = scores.to_numpy(dtype=float)
    if len(a) < 2 or not np.isfinite(a).all() or states.isna().any():
        raise ValueError('Finite complete scores and state labels required')
    if not 0 < level < 1:
        raise ValueError('level must lie in (0,1)')
    n = len(a); mean = float(a.mean())
    sums = pd.Series(a-mean,index=scores.index).groupby(states).sum()
    g = len(sums)
    if g < 2:
        raise ValueError('At least two clusters required')
    se = float(np.sqrt(g/(g-1)*np.square(sums).sum()/n**2))
    q = float(student_t.ppf((1+level)/2,g-1))
    return {'estimate':mean,'n':n,'clusters':g,'state_score_se':se,
            'conditional_interval_lower':mean-q*se,'conditional_interval_upper':mean+q*se,
            'episode_score_se':float(np.std(a,ddof=1)/np.sqrt(n)),
            'largest_state_fraction':float(states.value_counts().max()/n),
            'inference_status':'conditional score diagnostic only; not validated final inference'}


def ovb_point_bound(estimate, sigma2, nu2, cf_y, cf_d, rho=1):
    """Chernozhukov et al. OVB point bound for supplied sensitivity moments.

    No CI. cf_d is gain in Riesz precision for ATE, NOT treatment partial R2.
    Positive moments must be externally estimated/validated, preferably using
    out-of-fold predictions. Nonpositive orthogonal nu2 fails; no fallback.
    """
    vals=np.asarray([estimate,sigma2,nu2,cf_y,cf_d,rho],dtype=float)
    if not np.isfinite(vals).all() or sigma2 < 0 or nu2 <= 0:
        raise ValueError('Finite inputs with nonnegative sigma2 and positive nu2 required')
    if not 0 <= cf_y <= 1 or not 0 <= cf_d < 1 or not -1 <= rho <= 1:
        raise ValueError('Invalid sensitivity parameters')
    bias=float(abs(rho)*np.sqrt(sigma2*nu2*cf_y*cf_d/(1-cf_d)))
    return {'bias_bound':bias,'point_bound_lower':estimate-bias,'point_bound_upper':estimate+bias}
