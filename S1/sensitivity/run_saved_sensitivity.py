"""Recompute descriptive sensitivity summaries from frozen S1 and S4 outputs."""

import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, median

import pandas as pd

BASE = Path(__file__).resolve().parent
NOTEBOOKS = BASE.parent.parent
S1_RUN = BASE.parent / 'outputs/S1_02_policy_trees_20260926_195240_351369'
MI_RUN = BASE.parent / 'outputs/S1_03_completion_inference_20260926_202737_859797'
S4_BASE = NOTEBOOKS / 'S4'
sys.path.insert(0, str(S4_BASE))
from export_evaluator_comparison import ELIGIBLE, FIELDS, load_rows  # noqa: E402

BRANCHES = ['residential_eligible', 'residential_complete_case',
            'outpatient_eligible', 'outpatient_complete_case',
            'outpatient_temporal_2023', 'outpatient_temporal_mode_filled_2023']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_csv(path, rows):
    assert rows
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def s1_checks(folder, sources):
    summaries = []
    for branch in BRANCHES:
        root = S1_RUN / branch
        manifest = json.loads((root / 'manifest.json').read_text())
        source = root / 'manifest.json'
        sources[str(source.relative_to(NOTEBOOKS))] = digest(source)
        inputs = ['holdout_policy_value.csv', 'leave_one_state_out.csv',
                  'policy_gain_by_year.csv', 'holdout_assignments.csv.gz']
        for name in inputs:
            p = root / name
            assert digest(p) == manifest['files'][name], f'Changed S1 input: {p}'
            sources[str(p.relative_to(NOTEBOOKS))] = manifest['files'][name]
        policy = pd.read_csv(root / inputs[0]).set_index('policy')
        state = pd.read_csv(root / inputs[1])
        year = pd.read_csv(root / inputs[2])
        tree = float(policy.loc['tree', 'overlap_value'])
        zero = float(policy.loc['always_0', 'overlap_value'])
        one = float(policy.loc['always_1', 'overlap_value'])
        # Recheck saved aggregate and state-omission summaries before reporting them.
        assert abs(tree - max(zero, one) - float(manifest['tree_minus_best_constant'])) < 1e-10
        assert len(state) == manifest['states_in_test']
        summaries.append({'branch': branch, 'treatment_comparison': 'long versus short residential' if branch.startswith('residential') else 'intensive versus non-intensive outpatient',
                          'outcome': 'recorded administrative completion', 'evaluation_n': int(policy.loc['tree', 'n']),
                          'test_states': manifest['states_in_test'], 'tree_value': tree,
                          'always_0_value': zero, 'always_1_value': one,
                          'tree_minus_best_constant': tree - max(zero, one),
                          'tree_treatment_1_share': float(policy.loc['tree', 'intensive_or_long_share']),
                          'overlap_ess': float(policy.loc['tree', 'overlap_ess']),
                          'omitted_state_min_gain_vs_always_0': float(state.tree_minus_always_0.min()),
                          'omitted_state_max_gain_vs_always_0': float(state.tree_minus_always_0.max()),
                          'omitted_states_negative_gain_count': int(state.tree_minus_always_0.lt(0).sum()),
                          'year_min_gain_vs_always_0': float(year.tree_minus_always_0.min()),
                          'year_max_gain_vs_always_0': float(year.tree_minus_always_0.max()),
                          'leaf_count': len(manifest['leaf_inference']),
                          'holm_p_below_0_10_leaf_count': sum(x['p_holm'] < .1 for x in manifest['leaf_inference'] if x['p_holm'] is not None)})
    save_csv(folder / 'policy_branch_summary.csv', summaries)

    pairs = [('residential_eligible', 'residential_complete_case'),
             ('outpatient_eligible', 'outpatient_complete_case'),
             ('outpatient_temporal_2023', 'outpatient_temporal_mode_filled_2023')]
    comparisons = []
    for a, b in pairs:
        left = pd.read_csv(S1_RUN / a / 'holdout_assignments.csv.gz').set_index('episode_source_id', verify_integrity=True)
        right = pd.read_csv(S1_RUN / b / 'holdout_assignments.csv.gz').set_index('episode_source_id', verify_integrity=True)
        shared = left.index.intersection(right.index)
        assert len(shared) > 0
        l, r = left.loc[shared], right.loc[shared]
        assert (l[['DISYR', 'STFIPS']].to_numpy() == r[['DISYR', 'STFIPS']].to_numpy()).all()
        comparisons.append({'reference_branch': a, 'alternative_branch': b,
                            'reference_evaluation_n': len(left), 'alternative_evaluation_n': len(right),
                            'shared_episode_n': len(shared),
                            'share_of_reference_evaluation_matched': len(shared) / len(left),
                            'policy_disagreement_on_matched_episodes': float((l.policy != r.policy).mean()),
                            'mean_cate_alternative_minus_reference_on_matched_episodes': float((r.cate - l.cate).mean()),
                            'mean_abs_cate_difference_on_matched_episodes': float((r.cate - l.cate).abs().mean()),
                            'same_evaluation_population': len(left) == len(right) == len(shared)})
    save_csv(folder / 'paired_policy_comparisons.csv', comparisons)

    mi_manifest = MI_RUN / 'manifest.json'
    sources[str(mi_manifest.relative_to(NOTEBOOKS))] = digest(mi_manifest)
    mi = pd.read_csv(MI_RUN / 'draw_estimates.csv')
    source = MI_RUN / 'draw_estimates.csv'
    sources[str(source.relative_to(NOTEBOOKS))] = digest(source)
    assert len(mi) == 40 and set(mi.branch) == {'residential', 'outpatient'}
    rows = []
    for branch, d in mi.groupby('branch', sort=True):
        assert len(d) == 20 and d[['chain', 'iteration']].drop_duplicates().shape[0] == 20
        rows.append({'branch': branch, 'n_imputation_draws': len(d),
                     'mean_augmented_estimate': float(d.estimate.mean()),
                     'min_augmented_estimate': float(d.estimate.min()),
                     'max_augmented_estimate': float(d.estimate.max()),
                     'draw_estimate_sd_descriptive': float(d.estimate.std()),
                     'mean_unaugmented_ipw_contrast': float(d.ipw_contrast.mean()),
                     'mean_augmented_minus_ipw': float((d.estimate - d.ipw_contrast).mean()),
                     'min_control_ess': float(d.control_ess.min()),
                     'min_treated_ess': float(d.treated_ess.min())})
    save_csv(folder / 'imputation_draw_sensitivity.csv', rows)
    return summaries, comparisons, rows


def s4_checks(folder, run, sources):
    models, indexed = load_rows(run)  # validates fixed corpus, records, scores and prompts
    for name in ['run_config.json', 'literature_corpus.json', 'evaluation_input_lock.json']:
        p = run / name
        sources[str(p.relative_to(NOTEBOOKS))] = digest(p)
    for model in models:
        slug = ''.join(c if c.isalnum() else '_' for c in model.lower()).strip('_')
        p = run / slug / f'{slug}_literature_grounded_hypothesis_priority.csv'
        sources[str(p.relative_to(NOTEBOOKS))] = digest(p)

    strata = {}
    for model, candidates in indexed.items():
        for key, row in candidates.items():
            for generator in row['source_models'].split(' | '):
                strata.setdefault((row['analysis'], generator, model), []).append(row)
    matrix = []
    for (analysis, generator, evaluator), rows in sorted(strata.items()):
        matrix.append({'analysis': analysis, 'generator_model': generator,
                       'evaluator_model': evaluator, 'unique_hypotheses': len(rows),
                       'same_model_generator_evaluator': generator == evaluator,
                       'eligible_for_expert_review_count': sum(r['automated_status'] == ELIGIBLE for r in rows),
                       'possible_prior_test_count': sum(r['automated_status'] == 'potentially_already_tested_verify_full_text' for r in rows),
                       'mean_coverage_adjusted_novelty_descriptive': mean(float(r['confidence_adjusted_novelty_1_5']) for r in rows)})
    save_csv(folder / 'generator_by_evaluator.csv', matrix)

    cases = []
    for key in sorted(indexed[models[0]]):
        full = [indexed[m][key] for m in models]
        scenarios = [('all_evaluators', '', full),
                     ('exclude_generator_evaluator_overlap', '', [r for r in full if str(r['evaluator_source_model_overlap']).lower() == 'false'])]
        scenarios += [('omit_one_evaluator', m, [r for r in full if r['evaluator_model'] != m]) for m in models]
        for scenario, omitted, rows in scenarios:
            base = {'candidate_ids': key, 'analysis': full[0]['analysis'],
                    'source_models': full[0]['source_models'], 'hypothesis': full[0]['hypothesis'],
                    'scenario': scenario, 'omitted_evaluator': omitted,
                    'retained_evaluators': ' | '.join(r['evaluator_model'] for r in rows),
                    'retained_count': len(rows),
                    'eligible_fraction': mean(r['automated_status'] == ELIGIBLE for r in rows) if rows else '',
                    'status_unanimous': len({r['automated_status'] for r in rows}) == 1 if rows else '',
                    'statuses': ' | '.join(r['automated_status'] for r in rows)}
            base.update({field + '_median': median(float(r[field]) for r in rows) if rows else '' for field in FIELDS})
            cases.append(base)
    save_csv(folder / 'hypothesis_scenarios.csv', cases)
    baseline = {r['candidate_ids']: r for r in cases if r['scenario'] == 'all_evaluators'}
    summary = []
    for (scenario, omitted), group in pd.DataFrame(cases).groupby(['scenario', 'omitted_evaluator'], sort=False):
        rows = group.to_dict('records')
        summary.append({'scenario': scenario, 'omitted_evaluator': omitted,
                        'hypotheses': len(rows),
                        'hypotheses_with_no_remaining_evaluator': sum(r['retained_count'] == 0 for r in rows),
                        'hypotheses_with_changed_eligible_fraction': sum(r['eligible_fraction'] != baseline[r['candidate_ids']]['eligible_fraction'] for r in rows if r['retained_count']),
                        'hypotheses_with_changed_novelty_median': sum(r['confidence_adjusted_novelty_1_5_median'] != baseline[r['candidate_ids']]['confidence_adjusted_novelty_1_5_median'] for r in rows if r['retained_count']),
                        'hypotheses_with_unanimous_status': sum(r['status_unanimous'] is True for r in rows)})
    save_csv(folder / 'scenario_summary.csv', summary)
    return matrix, cases, summary


def run():
    pointer = json.loads((S4_BASE / 'FINAL_RERUN_CURRENT.json').read_text())
    s4 = Path(pointer['run_directory'])
    if not s4.is_absolute():
        s4 = NOTEBOOKS / s4
    assert json.loads((s4 / 'monitor_status.json').read_text())['status'] == 'complete_verified'
    dest = BASE / 'outputs' / f'S1_S4_saved_sensitivity_{datetime.now().astimezone():%Y%m%d_%H%M%S_%f}'
    s1, s4folder = dest / 'S1_completion', dest / 'S4_generator_evaluator'
    s1.mkdir(parents=True)
    s4folder.mkdir()
    sources = {}
    branches, comparisons, draws = s1_checks(s1, sources)
    matrix, cases, scenarios = s4_checks(s4folder, s4, sources)
    lines = ['# Saved result sensitivity checks', '',
             f'Run: {dest.name}', '',
             '## Scope', '',
             'S1 uses the fixed residential and outpatient completion comparisons. The policy summaries are held-out observational estimates. State omission leaves the fitted policy fixed. Complete-case comparisons change the analysis population and fitting inputs. The 2023 mode-filled policy was added after the original 2023 results were inspected.', '',
             'S1_03 imputation draw ranges and augmented versus unaugmented estimates are descriptive. They do not validate interval coverage or remove confounding. Frequency-reduction sensitivity and new tree refits are separate pending analyses.', '',
             'S4 uses identical frozen literature and prompts across all three evaluators. Multi-source hypotheses appear under each generating model; generator rows therefore overlap. Removing a self-evaluation or one evaluator changes the number of ratings, and an empty set is reported as unavailable. Scores remain separate by dimension. Agreement does not establish accuracy, novelty, or causal validity.', '',
             '## Files', '',
             '- `S1_completion/policy_branch_summary.csv`: six fitted policy branches, state and year omission ranges.',
             '- `S1_completion/paired_policy_comparisons.csv`: common-episode policy and predicted-effect differences.',
             '- `S1_completion/imputation_draw_sensitivity.csv`: 20 saved draws for each treatment contrast.',
             '- `S4_generator_evaluator/generator_by_evaluator.csv`: generator × evaluator × residential/outpatient matrix.',
             '- `S4_generator_evaluator/hypothesis_scenarios.csv`: all 27 hypotheses under five evaluator sets.',
             '- `S4_generator_evaluator/scenario_summary.csv`: changes from the full three-evaluator profile.', '',
             '## Counts', '',
             f'- S1 policy branches: {len(branches)}; paired comparisons: {len(comparisons)}; imputation branches: {len(draws)}.',
             f'- S4 hypotheses: {len(cases)//5}; generator/evaluator strata: {len(matrix)}; evaluator scenarios: {len(scenarios)}.',
             '', 'Outputs were calculated by code from saved records; no LLM calls or manual scoring occurred.']
    (dest / 'README.md').write_text('\n'.join(lines) + '\n')
    files = {str(p.relative_to(dest)): digest(p) for p in dest.rglob('*') if p.is_file()}
    (dest / 'manifest.json').write_text(json.dumps({'status': 'complete_descriptive_saved_result_sensitivity',
        'created_at': datetime.now().astimezone().isoformat(), 's1_source': str(S1_RUN),
        'imputation_source': str(MI_RUN), 's4_source': str(s4),
        'source_file_sha256': sources, 'output_file_sha256': files,
        'limits': ['No new model/tree refits', 'No validated causal uncertainty',
                   'No frequency reduction outcomes', 'No pooled quality score']}, indent=2) + '\n')
    print(dest)
    print('S1 branches:', len(branches), 'S4 hypotheses:', len(cases)//5)
    return dest


if __name__ == '__main__':
    run()
