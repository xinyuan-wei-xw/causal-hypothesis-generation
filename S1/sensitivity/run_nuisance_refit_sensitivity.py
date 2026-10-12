"""Alternative nuisance learners for evaluating frozen completion policies."""

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from run_unmeasured_confounding_sensitivity import sha, rewards, write_csv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BASE = HERE.parent / 'outputs/S1_02_policy_trees_20260926_195240_351369'


def values(reward, action, weight):
    scores = {'tree': reward[np.arange(len(action)), action],
              'always_0': reward[:, 0], 'always_1': reward[:, 1]}
    result = {name: float(np.average(score, weights=weight)) for name, score in scores.items()}
    result['tree_minus_best_constant'] = result['tree'] - max(result['always_0'], result['always_1'])
    result['always_1_minus_0'] = result['always_1'] - result['always_0']
    return result


def run():
    out = HERE / 'outputs' / f'S1_nuisance_refit_{datetime.now().astimezone():%Y%m%d_%H%M%S_%f}'
    out.mkdir(parents=True)
    rows, diagnostics, sources = [], [], {}
    for branch in ('residential_eligible', 'outpatient_eligible'):
        folder = BASE / branch
        manifest = json.loads((folder / 'manifest.json').read_text())
        source = ROOT / 'S1/fixed_inputs' / Path(manifest['source_file']).parent.name / Path(manifest['source_file']).name
        assert sha(source) == manifest['source_file_sha256']
        assert sha(folder / 'frozen_policy.joblib') == manifest['frozen_model_sha256']
        data = pd.read_csv(source)
        split = pd.read_csv(BASE / f'{branch}_split.csv.gz')
        assert data.episode_source_id.astype(str).equals(split.episode_source_id.astype(str))
        train = split.role.eq('train').to_numpy()
        test = split.role.eq('test').to_numpy()
        assert train.sum() and test.sum() and not np.any(train & test)
        fitted = joblib.load(folder / 'frozen_policy.joblib')
        fields = fitted['x_fields']
        assert 'PSYPROB' in fields

        def encode(frame):
            x = fitted['x_encoder'].transform(frame[fields].astype(str))
            w = fitted['w_encoder'].transform(
                (frame.STFIPS.astype(str) + '_' + frame.DISYR.astype(str)).to_frame('state_year'))
            return np.hstack((x, w))

        d_train, d_test = data.loc[train], data.loc[test]
        z_train, z_test = encode(d_train), encode(d_test)
        y_train = d_train.outcome.to_numpy(int)
        treatment_train = d_train.treatment.to_numpy(int)
        y_test = d_test.outcome.to_numpy(int)
        treatment_test = d_test.treatment.to_numpy(int)
        assignment = pd.read_csv(folder / 'holdout_assignments.csv.gz')
        assert d_test.episode_source_id.astype(str).reset_index(drop=True).equals(
            assignment.episode_source_id.astype(str))
        action = assignment.policy.to_numpy(int)
        original_e = fitted['evaluation_propensity'].predict_proba(z_test)[:, 1]
        original_m0 = fitted['mu0'].predict(z_test)
        original_m1 = fitted['mu1'].predict(z_test)
        assert np.allclose(original_e, assignment.propensity)
        weight = original_e * (1 - original_e)

        # Outcome and propensity learners change separately; the original target weights remain fixed.
        new_m = []
        for arm in (0, 1):
            mask = treatment_train == arm
            learner = LogisticRegression(C=1, solver='lbfgs', max_iter=2000)
            learner.fit(z_train[mask], y_train[mask])
            new_m.append(learner.predict_proba(z_test)[:, 1])
        propensity = RandomForestClassifier(n_estimators=100, max_depth=8,
                                            min_samples_leaf=100, n_jobs=2,
                                            random_state=61)
        propensity.fit(z_train, treatment_train)
        raw_e = propensity.predict_proba(z_test)[:, 1]
        new_e = np.clip(raw_e, .01, .99)
        configurations = {
            'original_nuisance': (original_e, original_m0, original_m1),
            'logistic_outcome_only': (original_e, new_m[0], new_m[1]),
            'forest_propensity_only': (new_e, original_m0, original_m1),
            'both_alternative': (new_e, new_m[0], new_m[1]),
        }
        original = None
        for label, (e, m0, m1) in configurations.items():
            score = rewards(y_test, treatment_test, e, m0, m1)
            value = values(score, action, weight)
            if label == 'original_nuisance':
                recorded = pd.read_csv(folder / 'holdout_policy_value.csv').set_index('policy')
                assert all(abs(value[name] - recorded.loc[name, 'overlap_value']) < 1e-6
                           for name in ('tree', 'always_0', 'always_1'))
                original = value
            rows.append({'branch': branch, 'n_test': len(d_test), 'n_train': len(d_train),
                         'nuisance_configuration': label, **value,
                         'gain_change_from_original': value['tree_minus_best_constant'] -
                         original['tree_minus_best_constant'],
                         'mean_original_overlap_weight': float(weight.mean())})
        diagnostics.append({'branch': branch, 'alternative_propensity_min': float(raw_e.min()),
                            'alternative_propensity_max': float(raw_e.max()),
                            'alternative_propensity_clipped_n': int(np.sum(new_e != raw_e)),
                            'mean_absolute_propensity_change': float(np.mean(np.abs(new_e - original_e))),
                            'mean_absolute_mu0_change': float(np.mean(np.abs(new_m[0] - original_m0))),
                            'mean_absolute_mu1_change': float(np.mean(np.abs(new_m[1] - original_m1)))})
        sources[branch] = {'source_sha256': sha(source),
                           'split_sha256': sha(BASE / f'{branch}_split.csv.gz'),
                           'model_sha256': sha(folder / 'frozen_policy.joblib'),
                           'manifest_sha256': sha(folder / 'manifest.json')}
    write_csv(out / 'nuisance_refit_policy_values.csv', rows)
    write_csv(out / 'nuisance_prediction_diagnostics.csv', diagnostics)
    assert len(rows) == 8 and len(diagnostics) == 2
    (out / 'README.md').write_text(
        '# Alternative nuisance-learner refit\n\n'
        'The primary tree and held-out episodes are fixed. Original overlap weights from the '
        'primary fitted propensity define one common evaluation target across all four configurations. '
        'Alternative outcome regressions are arm-specific logistic models; alternative treatment '
        'propensity is a random forest. Both use the original training episodes and encoded covariates. '
        'Alternative propensity predictions are clipped to [.01,.99] for score stability; the clipping '
        'count is reported. Baseline scores are reproduced before alternatives are reported. '
        'This checks nuisance-learner dependence of point policy values, not policy-learning '
        'stability, statistical significance, or unmeasured-confounding robustness. It is an '
        'exploratory post-result sensitivity analysis without interval estimates.\n')
    files = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    (out / 'manifest.json').write_text(json.dumps({
        'status': 'complete_exploratory_nuisance_refit',
        'created_at': datetime.now().astimezone().isoformat(),
        'source_hashes': sources,
        'outcome_alternative': 'LogisticRegression C=1 max_iter=2000 by arm',
        'propensity_alternative': 'RandomForestClassifier 100 trees depth=8 min_leaf=100',
        'propensity_clip': [.01, .99],
        'output_file_sha256': files}, indent=2) + '\n')
    print(out)
    return out


if __name__ == '__main__':
    run()
