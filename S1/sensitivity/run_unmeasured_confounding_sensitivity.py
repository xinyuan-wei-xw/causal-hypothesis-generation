"""Fixed-policy confounding scenarios and a measured PSYPROB omission check."""

import csv
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BASE = HERE.parent / 'outputs/S1_02_policy_trees_20260926_195240_351369'
sys.path.insert(0, str(HERE.parent / 'project_support/src'))
from causal_discovery.s1.development_inference import ovb_point_bound

STRENGTHS = (0, .01, .03, .05, .10, .20)


def adjustment_run():
    configured = os.environ.get('REPLICATION_SA01_RUN_DIR')
    candidates = ([Path(configured).expanduser().resolve()] if configured else
                  sorted((HERE / 'outputs').glob('S1_adjustment_without_PSYPROB_*'), reverse=True))
    for folder in candidates:
        manifest = folder / 'manifest.json'
        if manifest.is_file() and json.loads(manifest.read_text()).get('status', '').startswith('complete'):
            return folder
    raise FileNotFoundError('Run SA01 first or set REPLICATION_SA01_RUN_DIR to its completed output folder')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def predictions(model, data):
    x = model['x_encoder'].transform(data[model['x_fields']].astype(str))
    w = model['w_encoder'].transform(
        (data.STFIPS.astype(str) + '_' + data.DISYR.astype(str)).to_frame('state_year'))
    z = np.hstack((x, w))
    return (model['evaluation_propensity'].predict_proba(z)[:, 1],
            model['mu0'].predict(z), model['mu1'].predict(z))


def rewards(y, treatment, e, mu0, mu1):
    assert np.isfinite(e).all() and ((e > 0) & (e < 1)).all()
    return np.column_stack((
        mu0 + (treatment == 0) * (y - mu0) / (1 - e),
        mu1 + (treatment == 1) * (y - mu1) / e))


def verify_saved_values(folder, reward, e, action):
    h = e * (1 - e)
    recorded = pd.read_csv(folder / 'holdout_policy_value.csv').set_index('policy')
    values = {'tree': np.average(reward[np.arange(len(action)), action], weights=h),
              'always_0': np.average(reward[:, 0], weights=h),
              'always_1': np.average(reward[:, 1], weights=h)}
    assert all(abs(value - recorded.loc[name, 'overlap_value']) < 1e-6
               for name, value in values.items())


def run():
    alt_run = adjustment_run()
    out = HERE / 'outputs' / f'S1_unmeasured_confounding_{datetime.now().astimezone():%Y%m%d_%H%M%S_%f}'
    out.mkdir(parents=True)
    observed_rows, diagnostic_rows, grid_rows, source_hashes = [], [], [], {}
    for branch in ('residential_eligible', 'outpatient_eligible'):
        base, alt = BASE / branch, alt_run / branch
        main_manifest = json.loads((base / 'manifest.json').read_text())
        omitted_manifest = json.loads((alt / 'manifest.json').read_text())
        source = ROOT / 'S1/fixed_inputs' / Path(main_manifest['source_file']).parent.name / Path(main_manifest['source_file']).name
        assert source.is_file() and sha(source) == main_manifest['source_file_sha256']
        assert omitted_manifest['source_file_sha256'] == main_manifest['source_file_sha256']
        for folder, manifest in ((base, main_manifest), (alt, omitted_manifest)):
            assert sha(folder / 'frozen_policy.joblib') == manifest['frozen_model_sha256']
        split = BASE / f'{branch}_split.csv.gz'
        data = pd.read_csv(source)
        roles = pd.read_csv(split)
        assert data.episode_source_id.astype(str).equals(roles.episode_source_id.astype(str))
        data = data.loc[roles.role.eq('test')].reset_index(drop=True)
        baseline_assignment = pd.read_csv(base / 'holdout_assignments.csv.gz')
        omitted_assignment = pd.read_csv(alt / 'holdout_assignments.csv.gz')
        assert data.episode_source_id.astype(str).equals(baseline_assignment.episode_source_id.astype(str))
        assert data.episode_source_id.astype(str).equals(omitted_assignment.episode_source_id.astype(str))
        action = baseline_assignment.policy.to_numpy(int)
        assert np.array_equal(action, omitted_assignment.policy.to_numpy(int))
        full_model, short_model = joblib.load(base / 'frozen_policy.joblib'), joblib.load(alt / 'frozen_policy.joblib')
        assert 'PSYPROB' in full_model['x_fields'] and 'PSYPROB' not in short_model['x_fields']
        e_full, f0, f1 = predictions(full_model, data)
        e_short, s0, s1 = predictions(short_model, data)
        assert np.allclose(e_full, baseline_assignment.propensity)
        assert np.allclose(e_short, omitted_assignment.propensity)
        y, treatment = data.outcome.to_numpy(int), data.treatment.to_numpy(int)
        full_reward = rewards(y, treatment, e_full, f0, f1)
        short_reward = rewards(y, treatment, e_short, s0, s1)
        verify_saved_values(base, full_reward, e_full, action)
        verify_saved_values(alt, short_reward, e_short, action)
        h_full, h_short = e_full * (1 - e_full), e_short * (1 - e_short)
        sigma2_full = float(np.mean((y - np.where(treatment == 1, f1, f0)) ** 2))
        sigma2_short = float(np.mean((y - np.where(treatment == 1, s1, s0)) ** 2))
        cf_y_proxy = float(np.mean(np.where(treatment == 1, f1 - s1, f0 - s0) ** 2) / sigma2_short)
        contrasts = {'always_1_vs_0': np.ones(len(data)),
                     'tree_vs_always_0': action.astype(float),
                     'tree_vs_always_1': action.astype(float) - 1}
        source_hashes[branch] = {'source': sha(source), 'split': sha(split),
                                 'full_manifest': sha(base / 'manifest.json'),
                                 'omit_manifest': sha(alt / 'manifest.json')}
        for name, coefficient in contrasts.items():
            theta_full = float(np.average(coefficient * (full_reward[:, 1] - full_reward[:, 0]), weights=h_full))
            theta_short_fixed_target = float(np.average(
                coefficient * (short_reward[:, 1] - short_reward[:, 0]), weights=h_full))
            observed_rows.append({'branch': branch, 'contrast': name, 'n_heldout': len(data),
                                  'full_estimate_fixed_target': theta_full,
                                  'omit_PSYPROB_estimate_fixed_target': theta_short_fixed_target,
                                  'omit_minus_full': theta_short_fixed_target - theta_full,
                                  'mean_absolute_propensity_change': float(np.mean(np.abs(e_short - e_full))),
                                  'same_policy_assignments': True})
            if not np.any(coefficient):
                continue
            alpha_full = (h_full * coefficient / np.mean(h_full) *
                          (treatment / e_full - (1 - treatment) / (1 - e_full)))
            nu2_full = float(np.mean(alpha_full ** 2))
            assert sigma2_full >= 0 and nu2_full > 0
            for cf_y in STRENGTHS:
                for cf_d in STRENGTHS:
                    bound = ovb_point_bound(theta_full, sigma2_full, nu2_full, cf_y, cf_d, rho=1)
                    grid_rows.append({'branch': branch, 'contrast': name, 'cf_y_assumed': cf_y,
                                      'cf_d_assumed': cf_d, 'rho_assumed': 1,
                                      'estimate': theta_full, 'sigma2': sigma2_full, 'nu2': nu2_full,
                                      **bound, 'bound_includes_zero': bound['point_bound_lower'] <= 0 <= bound['point_bound_upper']})
            # This is a diagnostic of two fitted models, not a valid calibration of the grid.
            alpha_short = (h_short * coefficient / np.mean(h_short) *
                           (treatment / e_short - (1 - treatment) / (1 - e_short)))
            alpha_long = (h_short * coefficient / np.mean(h_short) *
                          (treatment / e_full - (1 - treatment) / (1 - e_full)))
            v_short, v_long = float(np.mean(alpha_short ** 2)), float(np.mean(alpha_long ** 2))
            cf_d_proxy = (v_long - v_short) / v_long
            short_theta = float(np.average(coefficient * (short_reward[:, 1] - short_reward[:, 0]), weights=h_short))
            long_theta = float(np.average(coefficient * (full_reward[:, 1] - full_reward[:, 0]), weights=h_short))
            proxy_bound = (ovb_point_bound(short_theta, sigma2_short, v_short, cf_y_proxy, cf_d_proxy)['bias_bound']
                           if 0 <= cf_y_proxy <= 1 and 0 <= cf_d_proxy < 1 else float('nan'))
            diagnostic_rows.append({'branch': branch, 'contrast': name,
                                    'outcome_strength_proxy': cf_y_proxy,
                                    'riesz_strength_proxy': cf_d_proxy,
                                    'observed_short_long_difference_fixed_target': long_theta - short_theta,
                                    'plug_in_proxy_bound': proxy_bound,
                                    'proxy_check_pass': bool(np.isfinite(proxy_bound) and
                                                             abs(long_theta - short_theta) <= proxy_bound + 1e-12),
                                    'formal_calibration_validated': False})
    write_csv(out / 'observed_PSYPROB_omission.csv', observed_rows)
    write_csv(out / 'PSYPROB_calibration_diagnostic.csv', diagnostic_rows)
    write_csv(out / 'hypothetical_confounding_grid.csv', grid_rows)
    assert len(observed_rows) == 6 and len(diagnostic_rows) == 5 and len(grid_rows) == 5 * len(STRENGTHS) ** 2
    assert any(not row['proxy_check_pass'] for row in diagnostic_rows)
    (out / 'README.md').write_text(
        '# Unmeasured-confounding sensitivity: point scenarios\n\n'
        'The primary held-out completion policies, outcomes, treatment definitions, and original estimated '
        'overlap target are fixed. The hypothetical grid uses the existing Chernozhukov-style point-bound '
        'formula with rho=1 and assumed cf_y/cf_d in {0, .01, .03, .05, .10, .20}. '
        'cf_d describes incremental Riesz precision, not ordinary treatment partial R-squared. '
        'The fitted propensity and outcome models supply sigma2 and nu2; they and the overlap target are '
        'treated as fixed. These are point scenarios, not confidence bounds or validated causal inference. '
        'The original data and models cannot identify the strength of an actual omitted confounder.\n\n'
        'Separately, observed_PSYPROB_omission.csv reports the measured change after refitting without '
        'PSYPROB on exactly the same held-out episodes, policies, and original overlap weights. This is '
        'an observed-variable stress test; it is not an estimate of the bias from an unmeasured variable. '
        'PSYPROB_calibration_diagnostic.csv tests whether naive model-difference proxies even enclose '
        'the observed short-versus-long estimate change. They fail for at least one contrast, so these '
        'proxies are NOT used to calibrate the hypothetical grid. The measured omission also changes '
        'fitted nuisance models, so its difference is not solely an omitted-variable effect.\n\n'
        'The tree-versus-constant contrasts treat each fitted policy as fixed. Residential tree-versus-'
        'always-0 is exactly zero and therefore has no nonzero Riesz representer or scenario grid. '
        'For outpatient care, a tree is guaranteed to beat both constant policies under a chosen '
        'point-bound scenario only if BOTH tree-versus-constant lower bounds exceed zero. '
        'No confidence intervals, policy-selection uncertainty, imputation uncertainty, or state-cluster '
        'uncertainty are propagated.\n\n'
        'Method context: Chernozhukov et al., Long Story Short: Omitted Variable Bias in Causal Machine '
        'Learning, Review of Economics and Statistics (2026), https://doi.org/10.1162/REST.a.1705 ; '
        'Cinelli and Hazlett, Making Sense of Sensitivity, '
        'JRSS B (2020), https://doi.org/10.1111/rssb.12348 . The latter cautions against informal '
        'observed-covariate benchmarking; this run does not claim a formal observed-covariate calibration.\n')
    files = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    (out / 'manifest.json').write_text(json.dumps({
        'status': 'complete_exploratory_point_scenarios_not_calibrated',
        'created_at': datetime.now().astimezone().isoformat(),
        'adjustment_run': str(alt_run),
        'source_hashes': source_hashes, 'assumed_strengths': STRENGTHS,
        'rho': 1, 'PSYPROB_used_as_formal_calibration': False,
        'output_file_sha256': files}, indent=2) + '\n')
    print(out)
    return out


if __name__ == '__main__':
    run()
