"""Deterministic comparisons of saved evaluator outputs; never calls an LLM."""
import argparse
import csv
import hashlib
import itertools
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median

BASE = Path(__file__).resolve().parent
FIELDS = {
    'scientific_validity_1_5': 'Scientific validity',
    'theoretical_coherence_1_5': 'Theoretical coherence',
    'significance_1_5': 'Scientific significance',
    'feasibility_1_5': 'Feasibility',
    'testability_1_5': 'Testability',
    'clarity_1_5': 'Clarity',
    'clinical_relevance_1_5': 'Clinical relevance',
    'confidence_adjusted_novelty_1_5': 'Coverage-adjusted novelty',
}
ELIGIBLE = 'eligible_for_expert_priority_review'
PRIOR = 'potentially_already_tested_verify_full_text'


def slug(model):
    return re.sub(r'[^a-z0-9]+', '_', model.lower()).strip('_')


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(headers, rows):
    def clean(value):
        return str(value).replace('|', '/').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(map(clean, headers)) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(clean, row)) + ' |' for row in rows])


def status_label(value):
    return {ELIGIBLE: 'Eligible for expert review', PRIOR: 'Possible prior test',
            'revise_before_priority_review': 'Revise before review'}.get(value, value)


def load_rows(run):
    config = json.loads((run / 'run_config.json').read_text())
    models = config['evaluator_models']
    if len(models) < 2:
        raise ValueError('Select at least two evaluators for comparison.')
    corpus_hash = hashlib.sha256((run / 'literature_corpus.json').read_bytes()).hexdigest()
    assert corpus_hash == config['literature_corpus_sha256'], 'Changed literature corpus'
    lock = json.loads((run / 'literature_corpus_lock.json').read_text())
    prompt_lock = json.loads((run / 'evaluation_input_lock.json').read_text())
    expected = set(lock['hypothesis_ids'])
    rows_by_model, prompt_hashes, descriptions = {}, {}, {}
    for model in models:
        folder = run / slug(model)
        records = json.loads((folder / f'{slug(model)}_evaluation_records.json').read_text())
        assert len(records) == len(expected), f'Incomplete evaluator: {model}'
        assert {r['candidate_ids'][0] for r in records} == expected
        with (folder / f'{slug(model)}_literature_grounded_hypothesis_priority.csv').open() as f:
            rows = list(csv.DictReader(f))
        indexed = {row['candidate_ids']: row for row in rows}
        assert len(rows) == len(indexed) == len(expected)
        for record in records:
            key = ' | '.join(record['candidate_ids'])
            row = indexed[key]
            assert row['evaluator_model'] == record['evaluator_model'] == model
            assert record['literature_corpus_sha256'] == corpus_hash
            assert record['system_prompt_sha256'] == prompt_lock['system_prompt_sha256']
            prompt_hashes.setdefault(key, set()).add(record['base_prompt_sha256'])
            descriptions.setdefault(key, set()).add((row['hypothesis'], row['analysis']))
            ratings = {r['dimension'] + '_1_5': r['rating'] for r in record['result']['ratings']}
            ratings['confidence_adjusted_novelty_1_5'] = record['confidence_adjusted_novelty_1_5']
            for field in FIELDS:
                assert 1 <= float(row[field]) <= 5 and float(row[field]) == ratings[field]
            assert row['hypothesis'] == record['hypothesis']
        rows_by_model[model] = indexed
    assert all(len(v) == 1 for v in prompt_hashes.values()), 'Different base prompts/evidence'
    assert all(len(v) == 1 for v in descriptions.values()), 'Different hypothesis texts'
    assert all(set(rows) == set(next(iter(rows_by_model.values()))) for rows in rows_by_model.values())
    return models, rows_by_model


def calculate(rows_by_model):
    models = list(rows_by_model)
    keys = sorted(rows_by_model[models[0]])
    summaries = []
    for model, indexed in rows_by_model.items():
        rows = list(indexed.values())
        item = {'evaluator_model': model, 'hypotheses': len(rows),
                'eligible_for_expert_review': sum(r['automated_status'] == ELIGIBLE for r in rows),
                'potentially_already_tested': sum(r['automated_status'] == PRIOR for r in rows),
                'revise_before_review': sum(r['automated_status'] == 'revise_before_priority_review' for r in rows),
                'source_model_overlap_count': sum(str(r['evaluator_source_model_overlap']).lower() == 'true' for r in rows)}
        for field in FIELDS:
            values = [float(r[field]) for r in rows]
            item[field + '_mean'] = round(mean(values), 4)
            item[field + '_median'] = median(values)
        summaries.append(item)
    comparisons = []
    for key in keys:
        rows = [rows_by_model[m][key] for m in models]
        statuses = [r['automated_status'] for r in rows]
        eligible = [s == ELIGIBLE for s in statuses]
        item = {'candidate_ids': key, 'hypothesis': rows[0]['hypothesis'], 'analysis': rows[0]['analysis'],
                'eligible_fraction': mean(eligible), 'eligibility_unanimous': len(set(eligible)) == 1,
                'status_unanimous': len(set(statuses)) == 1}
        for model, row in zip(models, rows):
            item[slug(model) + '_status'] = row['automated_status']
            item[slug(model) + '_pareto_tier'] = row['pareto_tier']
        for field in FIELDS:
            values = [float(row[field]) for row in rows]
            for model, value in zip(models, values):
                item[slug(model) + '_' + field] = value
            item[field + '_median'] = median(values)
            item[field + '_range'] = max(values) - min(values)
        item['largest_dimension_range'] = max(item[f + '_range'] for f in FIELDS)
        comparisons.append(item)
    agreements, status_agreements = [], []
    for a, b in itertools.combinations(models, 2):
        for field, label in FIELDS.items():
            differences = [float(rows_by_model[a][k][field]) - float(rows_by_model[b][k][field]) for k in keys]
            agreements.append({'evaluator_a': a, 'evaluator_b': b, 'dimension': label,
                               'hypotheses': len(keys), 'exact_agreement_percent': 100 * mean(d == 0 for d in differences),
                               'within_one_point_percent': 100 * mean(abs(d) <= 1 for d in differences),
                               'mean_absolute_difference': mean(abs(d) for d in differences),
                               'mean_signed_difference_a_minus_b': mean(differences)})
        pairs = [(rows_by_model[a][k]['automated_status'], rows_by_model[b][k]['automated_status']) for k in keys]
        status_agreements.append({'evaluator_a': a, 'evaluator_b': b, 'hypotheses': len(keys),
                                  'status_exact_agreement_percent': 100 * mean(x == y for x, y in pairs),
                                  'eligibility_agreement_percent': 100 * mean((x == ELIGIBLE) == (y == ELIGIBLE) for x, y in pairs)})
    return summaries, comparisons, agreements, status_agreements


def export_comparison(run):
    run = Path(run).resolve()
    models, rows_by_model = load_rows(run)
    summaries, comparisons, agreements, statuses = calculate(rows_by_model)
    outputs = []
    datasets = {'all_models_evaluation_scores.csv': [r for rows in rows_by_model.values() for r in rows.values()],
                'all_models_evaluator_summary.csv': summaries, 'all_models_sensitivity.csv': comparisons,
                'all_models_pairwise_rating_agreement.csv': agreements,
                'all_models_pairwise_status_agreement.csv': statuses}
    for name, rows in datasets.items():
        path = run / name
        write_csv(path, rows)
        outputs.append(path)
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/replication_plot_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def heatmap(matrix, labels, title, filename, maximum, minimum):
        fig, ax = plt.subplots(figsize=(12.5, 4.8))
        im = ax.imshow(matrix, vmin=minimum, vmax=maximum, cmap='Blues', aspect='auto')
        ax.set_xticks(range(len(FIELDS)), [textwrap.fill(s, 16) for s in FIELDS.values()], fontsize=9)
        ax.set_yticks(range(len(labels)), labels, fontsize=10)
        ax.set_title(title, pad=16)
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                        color='white' if value > minimum + .6 * (maximum - minimum) else 'black')
        fig.colorbar(im, ax=ax, fraction=.035, pad=.03)
        fig.tight_layout()
        path = run / filename
        fig.savefig(path, dpi=180, bbox_inches='tight')
        plt.close(fig)
        outputs.append(path)
    heatmap([[s[f + '_mean'] for f in FIELDS] for s in summaries], models,
            'Mean rating by dimension and evaluator (1–5)', 'all_models_dimension_means.png', 5, 1)
    pair_names = list(itertools.combinations(models, 2))
    heatmap([[next(r['mean_absolute_difference'] for r in agreements
                   if (r['evaluator_a'], r['evaluator_b'], r['dimension']) == (a, b, label))
              for label in FIELDS.values()] for a, b in pair_names],
            [a + '\nvs ' + b for a, b in pair_names], 'Mean absolute rating difference (0 = identical)',
            'all_models_rating_differences.png', 4, 0)
    parts = ['# Comparison of hypothesis evaluations',
             f'{len(comparisons)} hypotheses evaluated by {len(models)} models; {len(comparisons) * len(models)} saved evaluations.',
             'All evaluators used the same frozen literature and base prompts, verified by checksums. '
             'This report is generated directly from saved records and model CSV exports. '
             'It does not use an LLM to write or interpret the comparison.',
             '## How to read the comparison',
             'Scores remain separate dimensions; no composite weights or overall score are applied. '
             'Mean and median ratings describe scoring patterns, not which evaluator is correct. '
             'Means and absolute differences summarize the ordinal 1–5 scores numerically. '
             'Exact agreement is the percentage of hypotheses with the same rating; within-one-point agreement '
             'allows a difference of one. Signed differences above zero mean evaluator A gives higher ratings. '
             'Eligibility fraction is the fraction recommending expert priority review, not an agreement rate. '
             'Unanimity checks whether all evaluators make the same decision. Pareto tiers are specific to each '
             'evaluator; no cross-model tier average or consensus ranking is calculated. Agreement does not '
             'establish correctness, novelty, or causal validity. Source-model overlap is disclosed and expert review is required.',
             '## Evaluator summaries',
             table(['Evaluator', 'Hypotheses', 'Eligible', 'Potential prior test', 'Revise', 'Source overlap'],
                   [[s['evaluator_model'], s['hypotheses'], s['eligible_for_expert_review'],
                     s['potentially_already_tested'], s['revise_before_review'], s['source_model_overlap_count']] for s in summaries]),
             '![Dimension means](all_models_dimension_means.png)',
             '## Pairwise agreement',
             table(['Evaluator A', 'Evaluator B', 'Status agreement (%)', 'Eligibility agreement (%)'],
                   [[s['evaluator_a'], s['evaluator_b'], f"{s['status_exact_agreement_percent']:.1f}",
                     f"{s['eligibility_agreement_percent']:.1f}"] for s in statuses]),
             '![Rating differences](all_models_rating_differences.png)',
             '## Hypothesis-by-hypothesis scores',
             'Hypotheses are listed by candidate ID for navigation. This order is not a quality ranking.']
    for number, item in enumerate(comparisons, 1):
        key = item['candidate_ids']
        parts.extend([f"### {number}. {key}", item['hypothesis'],
                      table(['Evaluator', 'Automated status', 'Pareto tier'],
                            [[m, status_label(rows_by_model[m][key]['automated_status']), rows_by_model[m][key]['pareto_tier'] or '—'] for m in models]),
                      table(['Dimension'] + models + ['Range'],
                            [[label] + [f"{float(rows_by_model[m][key][field]):g}" for m in models] +
                             [f"{item[field + '_range']:g}"] for field, label in FIELDS.items()])])
    md = run / 'all_models_evaluation_comparison.md'
    md.write_text('\n\n'.join(parts) + '\n')
    word = run / 'all_models_evaluation_comparison.docx'
    subprocess.run(['pandoc', str(md), '--resource-path', str(run), '-o', str(word)], check=True)
    outputs.extend([md, word])
    summary = {'hypotheses': len(comparisons), 'evaluators': models,
               'evaluations': len(comparisons) * len(models),
               'hypotheses_with_unanimous_status': sum(r['status_unanimous'] for r in comparisons),
               'output_files': [p.name for p in outputs], 'all_outputs_generated_by_code': True}
    manifest_path = run / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['open_model_sensitivity'] = dict(summary, interpretation='descriptive evaluator agreement; not ground truth', composite_weights=None)
    for path in outputs:
        manifest['files'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return summary


CELL_SOURCE = '''# Export-only: all tables, text and figures are computed from saved evaluator outputs.
from pathlib import Path
import json, os, sys
_comparison_base=next(p for a in [Path.cwd(), *Path.cwd().parents]
    for p in [a, a/'S4']
    if (p/'export_evaluator_comparison.py').is_file())
if str(_comparison_base) not in sys.path:sys.path.insert(0,str(_comparison_base))
from export_evaluator_comparison import export_comparison
_comparison_run=globals().get('out') or os.environ.get('REPLICATION_S4_RESUME_DIR')
if not _comparison_run:_comparison_run=json.loads((_comparison_base/'FINAL_RERUN_CURRENT.json').read_text())['run_directory']
comparison_summary=export_comparison(Path(_comparison_run))
'''


def sync_notebook(summary=None, started=None):
    import nbformat
    from nbconvert import HTMLExporter
    path = BASE / 'S4_01_hypothesis_evaluation.ipynb'
    nb = nbformat.read(path, 4)
    cell = nbformat.v4.new_code_cell(CELL_SOURCE)
    cell.metadata['tags'] = ['evaluator-comparison-export']
    if summary is not None:
        cell.execution_count = max(c.get('execution_count') or 0 for c in nb.cells) + 1
        cell.outputs = [nbformat.v4.new_output('stream', name='stdout', text=json.dumps(summary, indent=2) + '\n')]
        elapsed = time.monotonic() - started
        nb.metadata['comparison_export_review'] = {'status': 'completed', 'elapsed_seconds': elapsed,
            'exported_at': datetime.now().astimezone().isoformat(), 'inference_repeated': False}
    nb.cells[-1] = cell
    nbformat.write(nb, path)
    if summary is not None:
        review = nbformat.from_dict(nb)
        execution = nb.metadata.get('execution_review', {})
        review.cells.insert(0, nbformat.v4.new_markdown_cell(
            '# Execution review\n\nNotebook execution: ' + json.dumps(execution) +
            '\n\nComparison export: ' + json.dumps(nb.metadata['comparison_export_review']) +
            '\n\nComparison regenerated from saved results only; no model inference repeated.'))
        report = BASE / 'reports' / f'{path.stem}_{datetime.now():%Y%m%d_%H%M%S}_full.html'
        report.parent.mkdir(exist_ok=True)
        report.write_text(HTMLExporter().from_notebook_node(review)[0])
        print('Executed comparison review:', report)


def wait_for_completion(run):
    while True:
        state_file = run / 'monitor_status.json'
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text())
            except json.JSONDecodeError:
                time.sleep(1)
                continue
            if state.get('status') == 'complete_verified':
                return
            if state.get('status', '').startswith('needs_attention'):
                raise RuntimeError('Evaluation monitor requires attention; comparison was not generated.')
        time.sleep(60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, nargs='?')
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--install-cell', action='store_true')
    parser.add_argument('--sync-notebook', action='store_true')
    args = parser.parse_args()
    if args.install_cell:
        sync_notebook()
    else:
        if args.run is None:
            parser.error('run directory is required')
        if args.wait:
            wait_for_completion(args.run)
        started = time.monotonic()
        summary = export_comparison(args.run)
        if args.sync_notebook:
            sync_notebook(summary, started)
