"""Complete-input episode-fold nuisance prediction prototype.

Not an MI adapter or cluster-DML estimator. Callers supply fixed fold labels and
an approved feature matrix; this module neither selects rows nor generates folds.
"""
import time
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def crossfit_predictions(x, y, treatment, folds, *, categorical, estimator, n_folds=5, unseen_category_policy="error"):
    """Return aligned e, mu0, mu1 and a fold audit; fit only on other folds.

    Unknown categorical levels, constant outcomes within a fit, missing values,
    model nonconvergence warnings and probability failures are explicit errors.
    No clipping or full-data category vocabulary is learned.
    Explicit zero_indicator policy encodes unseen categories as an all-zero
    indicator block and logs them. It assumes no learned category-specific
    contribution; it does not establish overlap and is not the strict default.
    """
    if unseen_category_policy not in ["error", "zero_indicator"]:
        raise ValueError("Unknown unseen-category policy")
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    if not isinstance(x, pd.DataFrame) or x.empty or not x.index.is_unique or x.columns.duplicated().any():
        raise ValueError('A nonempty feature DataFrame with unique IDs and columns is required.')
    for name, values in [('outcome',y),('treatment',treatment),('folds',folds)]:
        if not isinstance(values,pd.Series) or not values.index.equals(x.index):
            raise ValueError(f'{name} must have exactly aligned row IDs.')
    if x.isna().any().any() or not y.isin([0,1]).all() or not treatment.isin([0,1]).all():
        raise ValueError('Complete features and binary observed outcome/exposure required; no automatic deletion.')
    if not set(categorical)<=set(x.columns) or len(categorical)!=len(set(categorical)):
        raise ValueError('Categorical columns must be unique and present.')
    numeric=[v for v in x.columns if v not in categorical]
    if numeric and not np.isfinite(x[numeric].to_numpy(dtype=float)).all():
        raise ValueError('Numeric features must be finite.')
    if n_folds!=5 or set(folds.unique())!=set(range(n_folds)):
        raise ValueError('This approved prototype requires five fixed folds labeled 0..4.')
    result=pd.DataFrame(np.nan,index=x.index,columns=['e','mu0','mu1']);audit=[]
    for fold in range(n_folds):
        train=folds.ne(fold);score=folds.eq(fold)
        for name,target,fitmask in [('e',treatment,train),('mu0',y,train&treatment.eq(0)),('mu1',y,train&treatment.eq(1))]:
            if target.loc[fitmask].nunique()!=2:
                raise ValueError(f'Fold {fold} {name}: insufficient observed binary support.')
            unsupported = {}
            for col in categorical:
                unseen=set(x.loc[score,col])-set(x.loc[fitmask,col])
                if unseen:
                    unsupported[col] = sorted(str(v) for v in unseen)
                    if unseen_category_policy == 'error':
                        raise ValueError(f'Fold {fold} {name}: unseen {col} categories; review support.')
            assert not (fitmask&score).any()
            preprocess=ColumnTransformer([
                ('categorical',OneHotEncoder(handle_unknown='error' if unseen_category_policy == 'error' else 'ignore',sparse_output=False),list(categorical)),
                ('numeric',StandardScaler(),numeric)],remainder='drop')
            model=make_pipeline(preprocess,clone(estimator));start=time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                model.fit(x.loc[fitmask],target.loc[fitmask])
            probabilities=model.predict_proba(x.loc[score])
            class_index=list(model.classes_).index(1)
            pred=probabilities[:,class_index]
            if not np.isfinite(pred).all() or ((pred<0)|(pred>1)).any():
                raise ValueError('Invalid predicted probabilities.')
            if name=='e' and ((pred<=0)|(pred>=1)).any():
                raise ValueError('Propensity reached boundary; inspect support; no clipping.')
            result.loc[score,name]=pred
            audit.append({'fold':fold,'model':name,'fit_n':int(fitmask.sum()),'score_n':int(score.sum()),'shared_fit_score_ids':0,'seconds':time.perf_counter()-start,'unseen_category_policy':unseen_category_policy,'unsupported_categories':json.dumps(unsupported)})
    assert result.notna().all().all()
    return result,pd.DataFrame(audit)
