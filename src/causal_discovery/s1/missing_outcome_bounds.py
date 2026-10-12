"""Worst-case bounds for descriptive binary-outcome contrasts, not causal effects."""
import numpy as np
import pandas as pd


def binary_contrast_bounds(outcome, exposure):
    y=pd.Series(outcome).reset_index(drop=True)
    t=pd.Series(exposure).reset_index(drop=True)
    if len(y)!=len(t) or t.isna().any() or set(t.unique())!={0,1}:
        raise ValueError('Two observed binary exposure groups are required.')
    if not (y.isna()|y.isin([0,1])).all():
        raise ValueError('Outcome must be binary or missing.')
    rows=[]
    for arm in [0,1]:
        a=y[t.eq(arm)];n=len(a);s=int(a.eq(1).sum());u=int(a.isna().sum())
        rows.append({'exposure':arm,'n':n,'observed_n':n-u,'missing_n':u,'positive_n':s,
           'observed_risk':s/(n-u) if n>u else np.nan,
           'lower_risk':s/n,'upper_risk':(s+u)/n})
    arms=pd.DataFrame(rows).set_index('exposure')
    contrast={'observed_risk_difference':arms.loc[1,'observed_risk']-arms.loc[0,'observed_risk'],
              'lower_risk_difference':arms.loc[1,'lower_risk']-arms.loc[0,'upper_risk'],
              'upper_risk_difference':arms.loc[1,'upper_risk']-arms.loc[0,'lower_risk']}
    return arms.reset_index(),contrast
