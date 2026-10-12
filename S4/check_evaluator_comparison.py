"""Synthetic, temporary-file checks; produces no research results or LLM calls."""
import csv
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

from export_evaluator_comparison import FIELDS, calculate, export_comparison, load_rows, slug


def main():
    with tempfile.TemporaryDirectory() as directory:
        run = Path(directory)
        corpus = run / 'literature_corpus.json'
        corpus.write_text('{}')
        digest = hashlib.sha256(corpus.read_bytes()).hexdigest()
        models = ['test-a', 'test-b', 'test-c']
        (run / 'run_config.json').write_text(json.dumps({'evaluator_models': models, 'literature_corpus_sha256': digest}))
        (run / 'literature_corpus_lock.json').write_text(json.dumps({'hypothesis_ids': ['H1', 'H2']}))
        (run / 'evaluation_input_lock.json').write_text(json.dumps({'system_prompt_sha256': 'test-system'}))
        (run / 'manifest.json').write_text(json.dumps({'status': 'synthetic_test', 'files': {}}))
        for number, model in enumerate(models):
            folder = run / slug(model)
            folder.mkdir()
            records, rows = [], []
            for key in ['H1', 'H2']:
                score = 3 + number
                prior = key == 'H2' and number > 0
                novelty = 1 if prior else score
                values = {field: score for field in FIELDS}
                values['confidence_adjusted_novelty_1_5'] = novelty
                hypothesis = 'Synthetic test hypothesis ' + key
                records.append({'candidate_ids': [key], 'hypothesis': hypothesis,
                                'evaluator_model': model, 'literature_corpus_sha256': digest,
                                'system_prompt_sha256': 'test-system', 'base_prompt_sha256': 'test-prompt-' + key,
                                'confidence_adjusted_novelty_1_5': novelty,
                                'result': {'ratings': [{'dimension': f.removesuffix('_1_5'), 'rating': score}
                                                       for f in FIELDS if not f.startswith('confidence_adjusted')]}})
                rows.append({'candidate_ids': key, 'hypothesis': hypothesis, 'evaluator_model': model,
                             'analysis': 'synthetic test', 'pareto_tier': '' if prior else '1',
                             'evaluator_source_model_overlap': False,
                             'automated_status': 'potentially_already_tested_verify_full_text' if prior else 'eligible_for_expert_priority_review',
                             **values})
            (folder / f'{slug(model)}_evaluation_records.json').write_text(json.dumps(records))
            with (folder / f'{slug(model)}_literature_grounded_hypothesis_priority.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        _, rows = load_rows(run)
        summaries, comparisons, ratings, statuses = calculate(rows)
        assert [s['eligible_for_expert_review'] for s in summaries] == [2, 1, 1]
        assert comparisons[1]['eligible_fraction'] == 1 / 3
        assert not comparisons[1]['eligibility_unanimous']
        ac = next(r for r in ratings if r['evaluator_a'] == 'test-a' and r['evaluator_b'] == 'test-c'
                  and r['dimension'] == 'Scientific validity')
        assert ac['mean_absolute_difference'] == 2 and ac['within_one_point_percent'] == 0
        assert ac['mean_signed_difference_a_minus_b'] == -2
        assert statuses[0]['status_exact_agreement_percent'] == 50
        summary = export_comparison(run)
        assert summary['evaluations'] == 6 and summary['hypotheses_with_unanimous_status'] == 1
        assert len(list(csv.DictReader((run / 'all_models_pairwise_rating_agreement.csv').open()))) == 24
        with zipfile.ZipFile(run / 'all_models_evaluation_comparison.docx') as archive:
            assert archive.testzip() is None
            assert b'Synthetic test hypothesis H1' in archive.read('word/document.xml')
        manifest = json.loads((run / 'manifest.json').read_text())
        assert all(hashlib.sha256((run / name).read_bytes()).hexdigest() == sha for name, sha in manifest['files'].items())
        corpus.write_text('{"changed": true}')
        try:
            load_rows(run)
        except AssertionError:
            pass
        else:
            raise AssertionError('Changed corpus was accepted')
    print('PASS: known disagreement metrics, descriptive report/plots, Word archive, checksums and corpus mismatch rejection.')


if __name__ == '__main__':
    main()
