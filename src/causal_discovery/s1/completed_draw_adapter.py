"""Decode constrained completed draws without fitting or approving MI inference.

All joins use episode IDs. Nominal source codes stay nominal; the downstream
encoder must treat them categorically. A successful audit is an engineering
check, not evidence of convergence, unbiased estimation, or interval coverage.
"""
import numpy as np
import pandas as pd
from .imputation_rules import additional_count_bounds


def _indexed(frame):
    if 'episode_source_id' not in frame:
        raise ValueError('episode_source_id required')
    if frame.episode_source_id.isna().any() or frame.episode_source_id.duplicated().any():
        raise ValueError('Unique nonmissing episode IDs required')
    return frame.set_index('episode_source_id')


def decode_completed_draw(source, imputer_input, completed, *, fields):
    """Return decoded adjustment data and preserved treatment/outcome/metadata.

    Reordering an exported CSV is safe; missing or extra episodes are not.
    The input defines this imputer's categorical support, not a universal codebook.
    Joint employment codes are 10/20/30 and 41..45 (structural detail zero for
    nonapplicable parents). Unobserved values may change; observed values may not.
    """
    source, original, draw = map(_indexed, (source, imputer_input, completed))
    if not source.index.difference(original.index).empty or not original.index.difference(source.index).empty:
        raise ValueError('Source and imputer input episode sets differ')
    if set(draw.index) != set(source.index):
        raise ValueError('Completed draw episode set differs')
    original = original.loc[source.index]
    draw = draw.loc[source.index]
    if set(draw.columns) != set(original.columns):
        raise ValueError('Completed draw schema differs from imputer input')
    if draw.isna().any().any() or not np.isfinite(draw.to_numpy(dtype=float)).all():
        raise ValueError('Completed draw must be finite and complete')
    for column in original:
        observed = original[column].notna()
        if not draw.loc[observed,column].eq(original.loc[observed,column]).all():
            raise ValueError(f'Observed imputer input changed: {column}')
        if not draw[column].isin(original[column].dropna().unique()).all():
            raise ValueError(f'Unsupported imputer category: {column}')
    for column in ['los_gt30','analysis_outcome']:
        if not source[column].isin([0,1]).all() or not draw[column].eq(source[column]).all():
            raise ValueError(f'Observed binary source changed or missing: {column}')
    joint = draw['employment_joint']
    if not joint.isin([10,20,30,41,42,43,44,45]).all():
        raise ValueError('Invalid joint employment code')
    result = draw.drop(columns='employment_joint').copy()
    result['EMPLOY'] = (joint//10).astype(int)
    result['DETNLF'] = (joint%10).astype(int)
    if len(fields) != len(set(fields)) or not set(fields).issubset(result.columns):
        raise ValueError('Unique complete adjustment fields required')
    # Check against source as well: an incorrectly prepared imputer input must
    # not conceal an observed parent/detail or other source-value change.
    for column in fields:
        observed = source[column].notna()
        if not result.loc[observed,column].eq(source.loc[observed,column]).all():
            raise ValueError(f'Observed source value changed: {column}')
    bounds = additional_count_bounds(source.SUB2, source.SUB3)
    if not result.additional_substances_total.between(bounds.lower,bounds.upper).all():
        raise ValueError('Completed additional count violates source slot bounds')
    metadata = [c for c in ['DISYR','partition'] if c in source]
    model = result[fields+['los_gt30','analysis_outcome']].join(source[metadata])
    audit = pd.DataFrame([{'variable':c,'observed':int(source[c].notna().sum()),
                           'completed_missing':int(source[c].isna().sum())} for c in fields])
    return model.reset_index(), audit
