"""Compare saved augmented, weighted and standardized completion contrasts."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
SOURCE = BASE.parent / 'outputs/S1_03_completion_inference_20260926_202737_859797'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    out = BASE / 'outputs' / f'S1_estimator_sensitivity_{datetime.now().astimezone():%Y%m%d_%H%M%S_%f}'
    out.mkdir(parents=True)
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    assert manifest['status'] == 'exploratory_conditional_inference_complete_not_validated_for_causal_claims'
    data = pd.read_csv(SOURCE / 'draw_estimates.csv')
    assert len(data) == 40 and data.groupby('branch').size().to_dict() == {'outpatient': 20, 'residential': 20}
    cols = {'estimate': 'overlap_augmented', 'ipw_contrast': 'overlap_ipw',
            'outcome_model_component': 'overlap_outcome_standardization'}
    long = data.melt(id_vars=['branch', 'chain', 'iteration', 'seed', 'source_file_sha256'],
                     value_vars=list(cols), var_name='method', value_name='contrast')
    long['method'] = long.method.map(cols)
    assert len(long) == 120 and long.contrast.notna().all()
    long.to_csv(out / 'draw_method_contrasts.csv', index=False)
    summary = long.groupby(['branch', 'method'], sort=True).contrast.agg(
        draws='size', mean='mean', minimum='min', maximum='max', draw_sd='std').reset_index()
    summary.to_csv(out / 'estimator_summary.csv', index=False)
    (out / 'README.md').write_text(
        '# Completion estimator sensitivity\n\n'
        'Three estimates are read from each of the 20 completed datasets per service-setting contrast: '
        'overlap-weighted augmented estimation, normalized overlap inverse-probability weighting, '
        'and overlap outcome standardization. They use the same completed episodes and saved '
        'within-draw nuisance fits. Draw ranges and standard deviations are descriptive; completed '
        'datasets are dependent imputations, not independent study replications. No validated '
        'state-clustered inference or proof against unmeasured confounding is supplied.\n')
    files = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    (out / 'manifest.json').write_text(json.dumps({'status': 'complete_descriptive_estimator_sensitivity',
        'created_at': datetime.now().astimezone().isoformat(), 'source': str(SOURCE),
        'source_manifest_sha256': sha(SOURCE / 'manifest.json'),
        'source_draw_estimates_sha256': sha(SOURCE / 'draw_estimates.csv'),
        'output_file_sha256': files}, indent=2) + '\n')
    print(summary.to_string(index=False))
    print(out)
    return out


if __name__ == '__main__':
    run()
