"""Frozen-policy propensity and support checks for the two primary completion trees."""

import csv
import gc
import hashlib
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parent.parent
PRIMARY = BASE.parent / 'outputs/S1_02_policy_trees_20260926_195240_351369'
BRANCHES = ('residential_eligible', 'outpatient_eligible')
THRESHOLDS = (.01, .025, .05)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def values(y, t, e, mu0, mu1, action, weight):
    score0 = mu0 + (t == 0) * (y - mu0) / (1 - e)
    score1 = mu1 + (t == 1) * (y - mu1) / e
    chosen = np.where(action == 1, score1, score0)
    return [float(np.average(score, weights=weight)) for score in (chosen, score0, score1)]


def branch_rows(branch, sources):
    root = PRIMARY / branch
    manifest = json.loads((root / 'manifest.json').read_text())
    for name in ('frozen_policy.joblib', 'holdout_assignments.csv.gz', 'holdout_policy_value.csv'):
        p = root / name
        assert sha(p) == manifest['files'][name], f'Changed primary output: {p}'
        sources[str(p.relative_to(PROJECT))] = manifest['files'][name]
    source = PROJECT / 'S1/fixed_inputs' / Path(manifest['source_file']).parent.name / Path(manifest['source_file']).name
    assert sha(source) == manifest['source_file_sha256']
    sources[str(source.relative_to(PROJECT))] = manifest['source_file_sha256']
    saved = pd.read_csv(root / 'holdout_assignments.csv.gz', dtype={'episode_source_id': str})
    bundle = joblib.load(root / 'frozen_policy.joblib')
    fields = bundle['x_fields']
    columns = ['episode_source_id', 'STFIPS', 'DISYR', 'outcome', 'treatment', *fields]
    data = pd.read_csv(source, usecols=list(dict.fromkeys(columns)), dtype={'episode_source_id': str})
    assert data.episode_source_id.is_unique and saved.episode_source_id.is_unique
    data = data.set_index('episode_source_id').loc[saved.episode_source_id]
    assert len(data) == len(saved)
    assert np.array_equal(data.STFIPS.to_numpy(), saved.STFIPS.to_numpy())
    assert np.array_equal(data.DISYR.to_numpy(), saved.DISYR.to_numpy())
    x = bundle['x_encoder'].transform(data[fields].astype(str))
    state_year = (data.STFIPS.astype(str) + '_' + data.DISYR.astype(str)).to_frame('state_year')
    w = bundle['w_encoder'].transform(state_year)
    features = np.hstack((x, w))
    propensity = bundle['evaluation_propensity'].predict_proba(features)[:, 1]
    assert np.allclose(propensity, saved.propensity.to_numpy(), rtol=0, atol=2e-7)
    mu0 = bundle['mu0'].predict(features)
    mu1 = bundle['mu1'].predict(features)
    y = data.outcome.to_numpy(int)
    t = data.treatment.to_numpy(int)
    a = saved.policy.to_numpy(int)
    assert np.isin(y, (0, 1)).all() and np.isin(t, (0, 1)).all()
    h = propensity * (1 - propensity)
    original = pd.read_csv(root / 'holdout_policy_value.csv').set_index('policy')
    baseline = values(y, t, propensity, mu0, mu1, a, h)
    for value, policy in zip(baseline, ('tree', 'always_0', 'always_1')):
        assert abs(value - float(original.loc[policy, 'overlap_value'])) < 1e-6

    rows = []
    for scenario, threshold in [('reference', 0),
                                *[('clip_score_denominator', x) for x in THRESHOLDS],
                                *[('restrict_propensity_support', x) for x in THRESHOLDS]]:
        keep = np.ones(len(y), bool) if scenario != 'restrict_propensity_support' else ((propensity >= threshold) & (propensity <= 1 - threshold))
        e_used = np.clip(propensity[keep], threshold, 1 - threshold) if scenario == 'clip_score_denominator' else propensity[keep]
        weight = h[keep]  # fixed original overlap weights; restricted support defines a new population
        tree, zero, one = values(y[keep], t[keep], e_used, mu0[keep], mu1[keep], a[keep], weight)
        rows.append({'branch': branch, 'scenario': scenario, 'threshold': threshold,
                     'n': int(keep.sum()), 'n_excluded': int((~keep).sum()),
                     'states': int(data.STFIPS.iloc[np.flatnonzero(keep)].nunique()),
                     'overlap_ess': float(weight.sum() ** 2 / (weight @ weight)),
                     'tree_treatment_1_share': float(a[keep].mean()),
                     'tree_value': tree, 'always_0_value': zero, 'always_1_value': one,
                     'tree_minus_best_constant': tree - max(zero, one),
                     'tree_gain_change_from_reference': tree - baseline[0],
                     'policy_is_frozen': True, 'nuisance_models_are_frozen': True})
    del data, bundle, features, x, w, mu0, mu1
    gc.collect()
    return rows


def run():
    out = BASE / 'outputs' / f'S1_overlap_sensitivity_{datetime.now().astimezone():%Y%m%d_%H%M%S_%f}'
    out.mkdir(parents=True)
    sources = {}
    rows = []
    for branch in BRANCHES:
        rows.extend(branch_rows(branch, sources))
        print(branch, 'completed', flush=True)
    with (out / 'overlap_policy_sensitivity.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / 'README.md').write_text(
        '# Frozen completion-policy overlap sensitivity\n\n'
        'The residential and outpatient policy trees, fitted outcome models, held-out episodes, '
        'and original propensities are verified against the main S1_02 run. The reference policy '
        'values are reproduced numerically before each sensitivity calculation.\n\n'
        '`clip_score_denominator` changes only the inverse-probability denominators in the held-out '
        'doubly robust scores; the original overlap weights and episodes stay fixed. This changes '
        'the estimating score and may introduce bias. `restrict_propensity_support` excludes held-out '
        'episodes outside the stated interval, keeps nuisance predictions fixed, and changes the '
        'target population. Thresholds 0.01, 0.025, and 0.05 are reported together; none is chosen '
        'based on its effect estimate. The results are descriptive observational diagnostics, not '
        'validated causal confidence intervals or policy recommendations.\n')
    files = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    (out / 'manifest.json').write_text(json.dumps({'status': 'complete_frozen_policy_overlap_sensitivity',
        'created_at': datetime.now().astimezone().isoformat(), 'primary_s1_run': str(PRIMARY),
        'source_file_sha256': sources, 'output_file_sha256': files,
        'thresholds': THRESHOLDS, 'models_refitted': False}, indent=2) + '\n')
    print(out)
    return out


if __name__ == '__main__':
    run()
