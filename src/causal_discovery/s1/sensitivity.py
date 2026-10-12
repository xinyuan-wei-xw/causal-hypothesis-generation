"""Same-episode adjustment sensitivity on complete inputs.

Development component only: does not integrate MI or provide valid clustered
inference. Missing values are errors, never instructions to select complete cases.
Scientific thresholds and alternative estimators are not chosen by this module.
"""
import hashlib
import json
import time

import numpy as np
import pandas as pd

from .aipw import aipw_scores
from .crossfit import crossfit_predictions


def frame_hash(frame):
    """Hash ordered values, IDs, column labels and dtypes for run traceability."""
    digest = hashlib.sha256()
    digest.update(json.dumps([(str(c), str(d)) for c, d in frame.dtypes.items()]).encode())
    digest.update(frame.to_csv(index=True, float_format='%.17g').encode())
    return digest.hexdigest()


def adjustment_sets(primary, alternatives):
    """Resolve explicit remove/add comparisons without altering primary fields."""
    if not primary or len(primary) != len(set(primary)):
        raise ValueError('Primary features must be nonempty and unique.')
    result = {'primary': list(primary)}
    for spec in alternatives:
        name = spec['id']
        if name in result:
            raise ValueError('Comparison IDs must be unique; primary is reserved.')
        remove, add = spec.get('remove', []), spec.get('add', [])
        if len(remove) != len(set(remove)) or len(add) != len(set(add)):
            raise ValueError('Duplicate remove/add fields.')
        if not set(remove) <= set(primary) or set(add) & set(primary):
            raise ValueError('Remove must reference primary fields; add must be new.')
        selected = [c for c in primary if c not in remove] + list(add)
        if not selected or selected == list(primary):
            raise ValueError('An alternative must make a nonempty feature change.')
        result[name] = selected
    return result


def run_adjustment_sensitivity(x, y, treatment, folds, *, feature_sets,
                               categorical, estimator, unseen_category_policy="error"):
    """Refit all nuisance models for each feature set on exactly the same rows.

    Inputs are a completed development matrix and fixed episode-fold assignments.
    Multiple completed datasets are not analyzed together. All requested
    fields must be complete, including added covariates. A separate approved
    missing-data workflow must prepare them; there is no implicit row deletion.

    Returns point estimates, row scores, fit audit, and provenance. No standard
    errors, confidence intervals, p-values, or automatic robustness pass/fail.
    """
    if not isinstance(x, pd.DataFrame) or x.empty or not x.index.is_unique:
        raise ValueError('Nonempty feature frame with unique episode IDs required.')
    if x.columns.duplicated().any() or 'primary' not in feature_sets:
        raise ValueError('Unique columns and a primary feature set required.')
    for name, values in [('y', y), ('treatment', treatment), ('folds', folds)]:
        if not isinstance(values, pd.Series) or not values.index.equals(x.index):
            raise ValueError(f'{name}: exact episode ID and order alignment required.')
    if len(categorical) != len(set(categorical)) or not set(categorical) <= set(x):
        raise ValueError('Categorical fields must be unique and present.')
    for name, fields in feature_sets.items():
        if not fields or len(fields) != len(set(fields)) or not set(fields) <= set(x):
            raise ValueError(f'{name}: invalid or absent features.')
    requested = list(dict.fromkeys(c for fields in feature_sets.values() for c in fields))
    if x[requested].isna().any().any():
        raise ValueError('Missing comparison features: prepare inputs explicitly; no automatic deletion.')
    if not y.isin([0, 1]).all() or not treatment.isin([0, 1]).all():
        raise ValueError('Observed binary outcome and exposure required.')
    identity = pd.DataFrame({'outcome': y, 'treatment': treatment, 'fold': folds})
    frozen_hash = frame_hash(identity)
    estimates, scores, audits = [], {}, []
    for name, fields in feature_sets.items():
        start = time.perf_counter()
        predictions, audit = crossfit_predictions(
            x[fields], y, treatment, folds,
            categorical=[c for c in categorical if c in fields], estimator=estimator, unseen_category_policy=unseen_category_policy)
        score = aipw_scores(y, treatment, predictions.e, predictions.mu0, predictions.mu1)
        estimates.append({'comparison': name, 'n': len(x), 'n_treated': int(treatment.sum()),
                          'estimate': float(np.mean(score)), 'seconds': time.perf_counter()-start,
                          'propensity_min': float(predictions.e.min()),
                          'propensity_max': float(predictions.e.max()),
                          'same_episode_outcome_exposure_fold_sha256': frozen_hash,
                          'inference_status': 'not_implemented'})
        scores[name] = predictions.assign(score=score)
        audits.append(audit.assign(comparison=name))
    results = pd.DataFrame(estimates)
    baseline = results.loc[results.comparison.eq('primary'), 'estimate'].iloc[0]
    results['difference_from_primary'] = results.estimate - baseline
    manifest = {'scope': 'complete-input development point estimates only',
                'input_sha256': frame_hash(x[requested]), 'fixed_design_sha256': frozen_hash,
                'feature_sets': feature_sets, 'categorical': list(categorical),
                'estimator_class': type(estimator).__module__ + '.' + type(estimator).__name__,
                'estimator_parameters': estimator.get_params(deep=True),
                'unseen_category_policy': unseen_category_policy,
                'automatic_row_deletion': False, 'inference_status': 'not_implemented',
                'limitations': ['No multiple-imputation integration or clustered uncertainty.',
                                'No causal identification established by stability.',
                                'No effect-size-based model or specification selection.']}
    return results, scores, pd.concat(audits, ignore_index=True), manifest
