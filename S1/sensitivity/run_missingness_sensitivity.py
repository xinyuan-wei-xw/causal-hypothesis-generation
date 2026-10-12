"""Describe complete-case selection in the frozen completion samples."""

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parent.parent
PRIMARY = BASE.parent / 'outputs/S1_02_policy_trees_20260926_195240_351369'
SPEC = json.loads((BASE.parent / 'helper/assets/sample_specification.json').read_text())
FIELDS = [x for x in SPEC['fields'] if x not in ('STFIPS', 'DISYR')]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, rows):
    assert rows
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run():
    out = BASE / 'outputs' / f'S1_covariate_missingness_{datetime.now().astimezone():%Y%m%d_%H%M%S_%f}'
    out.mkdir(parents=True)
    sources = {}
    grouped, fields = [], []
    for branch in ('residential', 'outpatient'):
        manifest = json.loads((PRIMARY / f'{branch}_eligible/manifest.json').read_text())
        source = PROJECT / 'S1/fixed_inputs' / Path(manifest['source_file']).parent.name / Path(manifest['source_file']).name
        assert sha(source) == manifest['source_file_sha256']
        sources[str(source.relative_to(PROJECT))] = manifest['source_file_sha256']
        df = pd.read_csv(source, usecols=['episode_source_id', 'STFIPS', 'DISYR', 'treatment', 'outcome', 'complete_case', *FIELDS],
                         dtype={'episode_source_id': str})
        assert df.episode_source_id.is_unique
        assert len(df) == manifest['n_train'] + manifest['n_build'] + manifest['n_test']
        assert set(df.complete_case.unique()) <= {0, 1}
        assert df.outcome.isin([0, 1]).all()
        assert df.complete_case.eq(df[FIELDS].notna().all(axis=1)).all()
        for (state, year, treatment), part in df.groupby(['STFIPS', 'DISYR', 'treatment'], sort=True):
            ncc = int(part.complete_case.sum())
            grouped.append({'branch': branch, 'state': state, 'year': year, 'treatment': treatment,
                            'eligible_n': len(part), 'complete_case_n': ncc,
                            'complete_case_fraction': ncc / len(part),
                            'recorded_completion_all': float(part.outcome.mean()),
                            'recorded_completion_complete_case': float(part.loc[part.complete_case.eq(1), 'outcome'].mean()) if ncc else '',
                            'recorded_completion_incomplete_covariates': float(part.loc[part.complete_case.eq(0), 'outcome'].mean()) if ncc < len(part) else ''})
        for field in FIELDS:
            fields.append({'branch': branch, 'field': field, 'eligible_n': len(df),
                           'missing_n': int(df[field].isna().sum()),
                           'missing_fraction': float(df[field].isna().mean())})
        print(branch, 'eligible', len(df), 'complete case', int(df.complete_case.sum()), flush=True)
    save(out / 'state_year_treatment_selection.csv', grouped)
    save(out / 'covariate_missingness.csv', fields)
    (out / 'README.md').write_text(
        '# Complete-case selection in primary completion populations\n\n'
        'These tables use the frozen residential and outpatient eligible episode files. '
        'They report covariate availability and recorded completion by state, year, and treatment. '
        'All records in these files already have the selected completion outcome; these tables '
        'do not assess selection into the eligible data, missing discharge outcomes, or missing-not-at-random assumptions. '
        'Differences in observed completion by complete-case status are descriptive, not causal effects. '
        'The 2023 state-specific coding crosswalk remains unverified.\n')
    files = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    (out / 'manifest.json').write_text(json.dumps({'status': 'complete_descriptive_covariate_missingness',
        'created_at': datetime.now().astimezone().isoformat(), 'source_file_sha256': sources,
        'output_file_sha256': files}, indent=2) + '\n')
    print(out)
    return out


if __name__ == '__main__':
    run()
