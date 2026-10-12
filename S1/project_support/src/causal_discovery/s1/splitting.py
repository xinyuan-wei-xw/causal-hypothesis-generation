"""Configurable episode partitions; no final project ratios are selected here."""
import numpy as np
import pandas as pd

ID='episode_source_id'

def _validate(frame, year):
    if frame.empty or not frame.DISYR.eq(year).all():
        raise ValueError(f'Expected a nonempty {year} discharge cohort only.')
    if frame[ID].isna().any() or not frame[ID].is_unique:
        raise ValueError('Episode IDs must be observed and unique.')
    return frame.sort_values(ID).reset_index(drop=True)

def split_development(frame, *, ratios, seed, strata, development_exposed_ids=(), development_year=2022):
    """Reserve known development-exposed episodes for train; floor held-out counts per stratum.

    Ratios and seed are mandatory decisions. Requested counts use each full stratum;
    all rounding remainder goes to training. Small strata may lack held-out records:
    the returned audit exposes that fact. IDs identify episodes, not unique people.
    """
    if set(ratios)!={'train','validation','test'} or any(v is None for v in ratios.values()):
        raise ValueError('Explicit train/validation/test ratios are required.')
    if any(not np.isfinite(v) or v<0 for v in ratios.values()) or not np.isclose(sum(ratios.values()),1):
        raise ValueError('Ratios must be finite, nonnegative, and sum to one.')
    if seed is None or not strata or any(x not in frame for x in strata):
        raise ValueError('Explicit seed and valid stratification columns are required.')
    d=_validate(frame,development_year);rng=np.random.default_rng(seed)
    forced=d[ID].isin(set(development_exposed_ids));labels=np.full(len(d),'train',dtype=object);audit=[]
    for key,indices in d.groupby(list(strata),dropna=False,observed=True,sort=True).indices.items():
        n=len(indices);nt=int(np.floor(n*ratios['test']));nv=int(np.floor(n*ratios['validation']))
        free=indices[~forced.iloc[indices].to_numpy()]
        if nt+nv>len(free):
            raise ValueError(f'Development-exposed records prevent requested held-out allocation in stratum {key}.')
        shuffled=rng.permutation(free);labels[shuffled[:nt]]='test';labels[shuffled[nt:nt+nv]]='validation'
        key=key if isinstance(key,tuple) else (key,)
        audit.append(dict(zip(strata,key))|{'n':n,'train':n-nt-nv,'validation':nv,'test':nt,'forced_train':int(forced.iloc[indices].sum())})
    assignments=d[[ID,'DISYR']].copy();assignments['partition']=labels
    assert assignments.loc[forced,'partition'].eq('train').all()
    return assignments,pd.DataFrame(audit)

def select_temporal_validation(frame, *, mode='all', seed=None, fraction=None, ratio_to_training=None, training_n=None):
    """Select 2023 IDs uniformly without outcomes; reject impossible sizes explicitly."""
    d=_validate(frame,2023);n=len(d)
    if mode=='all':
        if fraction is not None or ratio_to_training is not None:
            raise ValueError('Do not supply size parameters in all mode.')
        k=n
    elif mode=='fraction':
        if fraction is None or not np.isfinite(fraction) or not 0<fraction<=1 or ratio_to_training is not None:
            raise ValueError('Fraction mode requires a fraction in (0,1] only.')
        k=int(np.floor(n*fraction))
    elif mode=='ratio_to_training':
        if ratio_to_training is None or not np.isfinite(ratio_to_training) or ratio_to_training<=0 or training_n is None or training_n<=0 or fraction is not None:
            raise ValueError('Ratio mode requires a positive ratio and training sample size.')
        k=int(np.floor(training_n*ratio_to_training))
    else:
        raise ValueError('Unknown temporal-validation mode.')
    if not 1<=k<=n:
        raise ValueError(f'Requested {k} temporal episodes, but {n} are available; change the configuration explicitly.')
    if k<n and seed is None:
        raise ValueError('Sampling requires an explicit seed.')
    selected=np.arange(n) if k==n else np.sort(np.random.default_rng(seed).choice(n,k,replace=False))
    result=d.iloc[selected][[ID,'DISYR']].copy();result['partition']='temporal_validation'
    return result.reset_index(drop=True)
