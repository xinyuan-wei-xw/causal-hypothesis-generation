"""Descriptive frequency-outcome observation scenarios for both service settings."""

import csv
import hashlib
import itertools
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
S1 = BASE.parent / 'outputs'
SOURCES = {
    'residential': S1 / 'S1_01_data_preparation_20260926_163212_739830',
    'outpatient': S1 / 'S1_03_outpatient_data_preparation_20260926_180001_385313',
}
PROBABILITIES = (0, .25, .5, .75, 1)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def setting_rows(setting, source):
    manifest = json.loads((source / 'manifest.json').read_text())
    file = source / 'frequency_final_eligible.csv.gz'
    observed = pd.read_csv(file, usecols=['episode_source_id', 'treatment', 'outcome'],
                           dtype={'episode_source_id': str})
    assert observed.episode_source_id.is_unique
    if setting == 'residential':
        assert len(observed) == manifest['outputs'][file.name]['n']
        assert sha(file) == manifest['outputs'][file.name]['sha256']
        availability = pd.read_csv(source / 'availability.csv')
        availability = availability.loc[availability.outcome.eq('frequency')].set_index('arm').to_dict('index')
    else:
        sample = next(x for x in manifest['samples'] if x['outcome'] == 'frequency')
        assert len(observed) == sample['n']
        member_hash = hashlib.sha256(('\n'.join(sorted(observed.episode_source_id)) + '\n').encode()).hexdigest()
        assert member_hash == sample['membership_sha256']
        availability = {x['arm']: x for x in manifest['outcome_availability'] if x['outcome'] == 'frequency'}
    labels = ('short_term', 'long_term') if setting == 'residential' else ('nonintensive', 'intensive')
    counts = []
    for arm in (0, 1):
        d = observed.loc[observed.treatment.eq(arm), 'outcome']
        assert d.isin([0, 1]).all()
        meta = availability[arm]
        assert len(d) == meta['observed_n'] and meta['target_n'] >= len(d)
        counts.append({'setting': setting, 'treatment': arm, 'treatment_label': labels[arm],
                       'target_n': int(meta['target_n']), 'observed_n': len(d),
                       'unobserved_n': int(meta['target_n'] - len(d)),
                       'improved_observed_n': int(d.sum()),
                       'observed_improvement_fraction': float(d.mean()),
                       'observation_fraction': len(d) / meta['target_n']})
    by_arm = {r['treatment']: r for r in counts}
    scenarios = []
    for p0, p1 in itertools.product(PROBABILITIES, repeat=2):
        risks = {arm: (by_arm[arm]['improved_observed_n'] + p * by_arm[arm]['unobserved_n']) / by_arm[arm]['target_n']
                 for arm, p in ((0, p0), (1, p1))}
        scenarios.append({'setting': setting, 'missing_improvement_probability_arm0': p0,
                          'missing_improvement_probability_arm1': p1,
                          'assumed_arm0_improvement_fraction': risks[0],
                          'assumed_arm1_improvement_fraction': risks[1],
                          'arm1_minus_arm0_descriptive': risks[1] - risks[0]})
    hashes = {'manifest_sha256': sha(source / 'manifest.json'), 'frequency_sha256': sha(file)}
    if setting == 'residential':
        hashes['availability_sha256'] = sha(source / 'availability.csv')
    return counts, scenarios, hashes


def run():
    out = BASE / 'outputs' / f'S1_outcome_observation_{datetime.now().astimezone():%Y%m%d_%H%M%S_%f}'
    out.mkdir(parents=True)
    counts, scenarios, sources = [], [], {}
    for setting, source in SOURCES.items():
        c, s, hashes = setting_rows(setting, source)
        counts.extend(c)
        scenarios.extend(s)
        sources[setting] = hashes
    save(out / 'outcome_availability.csv', counts)
    save(out / 'frequency_missing_outcome_scenarios.csv', scenarios)
    (out / 'README.md').write_text(
        '# Frequency-reduction outcome-observation scenarios\n\n'
        'For each treatment arm, missing outcomes are assigned hypothetical improvement '
        'probabilities 0, .25, .5, .75, or 1. Each setting has 25 scenarios. Arm 1 minus '
        'arm 0 is long-term minus short-term residential or intensive minus non-intensive '
        'outpatient. These are unadjusted descriptive contrasts across the target episodes, '
        'not causal effects or confidence bounds. They do not address selection into the '
        'completion sample, state/year coding, or missing covariates.\n')
    files = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    (out / 'manifest.json').write_text(json.dumps({
        'status': 'complete_descriptive_frequency_scenarios',
        'created_at': datetime.now().astimezone().isoformat(),
        'source_hashes': sources, 'assumptions': PROBABILITIES,
        'output_file_sha256': files}, indent=2) + '\n')
    assert len(counts) == 4 and len(scenarios) == 50
    print(out)
    return out


if __name__ == '__main__':
    run()
