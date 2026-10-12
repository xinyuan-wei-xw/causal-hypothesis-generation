"""Single entry point for the existing sensitivity notebooks."""

import csv
import hashlib
import html
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
HELPER = HERE.parent / 'helper/run_notebook.py'
STEPS = (
    ('saved_S1_S4', 'run_saved_sensitivity.py', 'S1_S4_saved_sensitivity_'),
    ('SA01_adjustment', 'S1_SA01_service_setting_adjustment.ipynb', 'S1_adjustment_without_PSYPROB_'),
    ('SA02_missingness', 'S1_SA02_service_setting_missingness.ipynb', 'S1_covariate_missingness_'),
    ('SA03_overlap', 'S1_SA03_service_setting_overlap.ipynb', 'S1_overlap_sensitivity_'),
    ('SA04_frequency', 'S1_SA04_outpatient_frequency_observation.ipynb', 'S1_outcome_observation_'),
    ('SA05_confounding', 'S1_SA05_service_setting_unmeasured_confounding.ipynb', 'S1_unmeasured_confounding_'),
    ('SA06_estimators', 'S1_SA06_service_setting_estimators.ipynb', 'S1_estimator_sensitivity_'),
    ('SA06b_nuisance', 'S1_SA06b_service_setting_nuisance_refit.ipynb', 'S1_nuisance_refit_'),
    ('SA07_sample_size', 'S1_SA07_service_setting_sample_size.ipynb', 'S1_sample_size_stability_'),
)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(folder):
    manifest = folder / 'manifest.json'
    info = json.loads(manifest.read_text())
    assert info['status'].startswith('complete'), folder
    for name, digest in info.get('output_file_sha256', {}).items():
        assert sha(folder / name) == digest, (folder, name)
    for name, key in (('sample_size_stability.csv', 'results_sha256'),
                      ('sample_size_summary.csv', 'summary_sha256'),
                      ('paired_refit_comparison.csv', 'paired_comparison_sha256')):
        if key in info:
            assert sha(folder / name) == info[key], (folder, name)
    return sha(manifest), info['status']


def latest_complete(prefix):
    for folder in sorted((HERE / 'outputs').glob(prefix + '*'), reverse=True):
        try:
            verify(folder)
            return folder
        except (OSError, ValueError, KeyError, AssertionError):
            continue
    raise FileNotFoundError(f'No verified completed run for {prefix}')


def save_progress(path, progress):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(progress, indent=2) + '\n')
    temp.replace(path)


def run(mode='rerun_all', resume_dir=None):
    if mode not in ('collect_latest', 'rerun_all'):
        raise ValueError('mode must be collect_latest or rerun_all')
    if mode == 'rerun_all':
        try:
            import econml  # noqa: F401 -- required by the two causal-forest refit notebooks
        except ImportError as exc:
            raise RuntimeError('Select a Python environment with econml installed before rerun_all') from exc
    output_root = HERE / 'outputs'
    output_root.mkdir(exist_ok=True)
    folder = (Path(resume_dir).resolve() if resume_dir else
              output_root / f'S1_04_sensitivity_{datetime.now().astimezone():%Y%m%d_%H%M%S_%f}')
    if resume_dir:
        progress = json.loads((folder / 'progress.json').read_text())
        assert folder.parent == output_root.resolve() and progress['mode'] == mode
    else:
        folder.mkdir()
        progress = {'mode': mode, 'started_at': datetime.now().astimezone().isoformat(),
                    'steps': {}}
        save_progress(folder / 'progress.json', progress)
    rows = []
    for label, notebook, prefix in STEPS:
        prior = progress['steps'].get(label, {})
        started = time.perf_counter()
        if prior.get('status') == 'complete':
            child = Path(prior['output_folder'])
            digest, status = verify(child)
            executed = False
        elif mode == 'collect_latest':
            child = latest_complete(prefix)
            digest, status = verify(child)
            executed = False
        else:
            before = set(output_root.glob(prefix + '*'))
            env = os.environ.copy()
            log = folder / f'{label}.log'
            partial = prior.get('partial_output')
            if label == 'SA05_confounding':
                env['REPLICATION_SA01_RUN_DIR'] = progress['steps']['SA01_adjustment']['output_folder']
            if label == 'SA07_sample_size':
                partial = partial or str(output_root / f'{prefix}{datetime.now().astimezone():%Y%m%d_%H%M%S_%f}')
                env['REPLICATION_SA07_RESUME_DIR'] = partial
            progress['steps'][label] = {'status': 'running', 'log': str(log),
                                        'partial_output': partial}
            save_progress(folder / 'progress.json', progress)
            command = ([sys.executable, str(HELPER), str(HERE / notebook)]
                       if notebook.endswith('.ipynb') else [sys.executable, str(HERE / notebook)])
            with log.open('w') as stream:
                completed = subprocess.run(
                    command,
                    cwd=HERE.parent.parent, env=env, stdout=stream,
                    stderr=subprocess.STDOUT, check=False)
            candidates = sorted(set(output_root.glob(prefix + '*')) - before,
                                key=lambda item: item.stat().st_mtime, reverse=True)
            if partial:
                candidates.insert(0, Path(partial))
            if completed.returncode != 0:
                progress['steps'][label] = {
                    'status': 'failed', 'log': str(log),
                    'partial_output': str(candidates[0]) if candidates else None}
                save_progress(folder / 'progress.json', progress)
                raise RuntimeError(f'{label} failed; see {log}')
            child = next((item for item in candidates if (item / 'manifest.json').is_file()), None)
            if child is None:
                raise RuntimeError(f'{label} completed without a new manifest; see {log}')
            digest, status = verify(child)
            executed = True
        row = {'step': label, 'notebook': notebook, 'output_folder': str(child),
               'manifest_sha256': digest, 'analysis_status': status,
               'executed_in_this_run': executed,
               'elapsed_seconds': round(time.perf_counter() - started, 3)}
        rows.append(row)
        progress['steps'][label] = {'status': 'complete', **row}
        save_progress(folder / 'progress.json', progress)
        with (folder / 'run_index.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print(f'{label}: {status} -> {child.name}', flush=True)
    (folder / 'README.md').write_text(
        '# Combined sensitivity run\n\n'
        f'Mode: `{mode}`. The nine analyses remain in their separately timestamped, '
        'verified output folders; `run_index.csv` links them and preserves source notebook '
        'and manifest hashes. `collect_latest` reuses completed outputs without fitting. '
        '`rerun_all` executes every notebook sequentially and can be resumed by passing this '
        'folder as `resume_dir`. Each child execution also exports its own timed HTML report. '
        'SA07 can take about 45 minutes; the full rerun may take about an hour. '
        'The confounding grid is hypothetical and not formally calibrated to PSYPROB. '
        'The S4 generator/evaluator table keeps both model identities.\n')
    report_dir = HERE / 'reports'
    report_dir.mkdir(exist_ok=True)
    report = report_dir / f'S1_04_sensitivity_{datetime.now().astimezone():%Y%m%d_%H%M%S_%f}_summary.html'
    cells = ''.join(
        '<tr>' + ''.join(f'<td>{html.escape(str(row[key]))}</td>'
                           for key in ('step', 'analysis_status', 'elapsed_seconds', 'output_folder')) + '</tr>'
        for row in rows)
    report.write_text(
        '<!doctype html><html><head><meta charset="utf-8"><title>Combined sensitivity review</title>'
        '<style>body{font:15px system-ui;max-width:1100px;margin:3em auto}td,th{padding:.5em;'
        'border:1px solid #ccc;text-align:left}table{border-collapse:collapse}</style></head><body>'
        f'<h1>Combined sensitivity review</h1><p>Status: complete. Mode: {html.escape(mode)}. '
        f'Total stage runtime: {sum(row["elapsed_seconds"] for row in rows):.1f} seconds.</p>'
        '<table><tr><th>Step</th><th>Status</th><th>Seconds</th><th>Output folder</th></tr>'
        + cells + '</table></body></html>')
    files = {p.name: sha(p) for p in folder.iterdir() if p.is_file() and p.name != 'manifest.json'}
    (folder / 'manifest.json').write_text(json.dumps({
        'status': 'complete_combined_sensitivity_index',
        'created_at': datetime.now().astimezone().isoformat(), 'mode': mode,
        'steps': rows, 'report': str(report), 'report_sha256': sha(report),
        'output_file_sha256': files}, indent=2) + '\n')
    print('Combined run:', folder)
    return folder


if __name__ == '__main__':
    run(mode=os.environ.get('REPLICATION_SENSITIVITY_MODE', 'rerun_all'),
        resume_dir=os.environ.get('REPLICATION_SENSITIVITY_RESUME_DIR'))
