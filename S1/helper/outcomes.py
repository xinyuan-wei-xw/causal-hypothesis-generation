"""Approved substance matching; analytical outcome populations remain separate."""
import pandas as pd


def derive_audit_fields(data):
    """Return prespecified audit/sensitivity fields without changing eligibility."""
    result = pd.DataFrame(index=data.index)
    result['employment_joint_observed'] = (data.EMPLOY * 10).astype('Int64')
    outside = data.EMPLOY.eq(4).fillna(False)
    result.loc[outside, 'employment_joint_observed'] = (40 + data.loc[outside, 'DETNLF']).astype('Int64')
    result['referral_joint_observed'] = (data.PSOURCE * 10).astype('Int64')
    criminal = data.PSOURCE.eq(7).fillna(False)
    result.loc[criminal, 'referral_joint_observed'] = (70 + data.loc[criminal, 'DETCRIM']).astype('Int64')
    result['known_non_none_primary_substance'] = data.SUB1.between(2, 19).astype('Int64')
    result['additional_substance_partial_information'] = data[['SUB2', 'SUB3']].isna().sum(axis=1).eq(1).astype('Int64')
    result['completion_reason_1_vs_2_3'] = pd.Series(pd.NA, index=data.index, dtype='Int64')
    known = data.REASON.isin([1, 2, 3])
    result.loc[known, 'completion_reason_1_vs_2_3'] = data.loc[known, 'REASON'].eq(1).astype('Int64')
    return result


def match_admission_substance(data):
    """Locate admission SUB1 in discharge slots without treating absence as abstinence.

    Expects harmonized codes with special missing values already converted to NA.
    Duplicate substance matches fail explicitly because no resolution is approved.
    Returns frequencies and audit flags; binary outcome classification is a separate step.
    """
    columns = ['SUB1', 'FREQ1'] + [f'{v}{k}_D' for k in (1, 2, 3) for v in ('SUB', 'FREQ')]
    d = data[columns]
    for col in columns:
        domain = range(1, 20) if col.startswith('SUB') else (1, 2, 3)
        if (~(d[col].isna() | d[col].isin(domain))).any():
            raise ValueError(f'Unexpected harmonized code in {col}')
    matches = pd.DataFrame({k: (d.SUB1.notna() & d.SUB1.ne(1) & d.SUB1.eq(d[f'SUB{k}_D'])).fillna(False) for k in (1, 2, 3)})
    if matches.sum(axis=1).gt(1).any():
        raise ValueError('Multiple discharge slots match the admission substance; resolution requires review.')
    result = pd.DataFrame(index=d.index)
    result['admission_frequency'] = d.FREQ1.astype('Int64')
    result['matched_discharge_slot'] = pd.Series(pd.NA, index=d.index, dtype='Int64')
    result['matched_discharge_frequency'] = pd.Series(pd.NA, index=d.index, dtype='Int64')
    for k in (1, 2, 3):
        result.loc[matches[k], 'matched_discharge_slot'] = k
        result.loc[matches[k], 'matched_discharge_frequency'] = d.loc[matches[k], f'FREQ{k}_D'].astype('Int64')
    result['match_status'] = 'unmatched'
    result.loc[d.SUB1.isna() | d.SUB1.eq(1).fillna(False), 'match_status'] = 'admission_substance_unknown_or_none'
    result.loc[result.matched_discharge_slot.notna(), 'match_status'] = 'matched_frequency_missing'
    result.loc[result.matched_discharge_frequency.notna(), 'match_status'] = 'matched_frequency_known'
    result['same_substance_pair_observed'] = result.admission_frequency.notna() & result.matched_discharge_frequency.notna()
    result['primary_slot_discharge_frequency'] = d.FREQ1_D.astype('Int64')
    result['primary_slot_pair_observed'] = d.FREQ1.notna() & d.FREQ1_D.notna()
    result['baseline_no_use'] = d.FREQ1.eq(1).astype('boolean').mask(d.FREQ1.isna())
    return result


def derive_frequency_outcomes(data):
    """Separate frequency reduction from sustained abstinence, preserving NA.

    A zero means an observed non-reduction (same or higher frequency) among
    baseline users, or observed use at discharge among baseline nonusers.
    Outcomes outside their baseline population stay NA, never zero.
    """
    result = match_admission_substance(data)
    admission = result.admission_frequency
    for prefix, discharge in [('same_substance', result.matched_discharge_frequency),
                              ('primary_slot', result.primary_slot_discharge_frequency)]:
        observed = admission.notna() & discharge.notna()
        for name, population, value in [
            ('frequency_reduction', admission.isin([2, 3]), discharge.lt(admission)),
            ('sustained_abstinence', admission.eq(1).fillna(False), discharge.eq(1)),
        ]:
            column = f'{prefix}_{name}'
            result[column] = pd.Series(pd.NA, index=result.index, dtype='Int64')
            valid = observed & population
            result.loc[valid, column] = value.loc[valid].astype('Int64')
        change = f'{prefix}_frequency_change'
        result[change] = pd.Series(pd.NA, index=result.index, dtype='Int64')
        users = observed & admission.isin([2, 3])
        result.loc[users, change] = (admission.loc[users] - discharge.loc[users]).clip(-1, 1).astype('Int64')
    return result


def _self_check():
    data = pd.DataFrame({
        'SUB1': [2, 2, 2], 'FREQ1': [3, 2, 2],
        'SUB1_D': [2, 2, 2], 'FREQ1_D': [2, 2, 3],
        'SUB2_D': [1, 1, 1], 'FREQ2_D': [1, 1, 1],
        'SUB3_D': [1, 1, 1], 'FREQ3_D': [1, 1, 1],
    }, dtype='Int64')
    outcome = derive_frequency_outcomes(data)
    assert outcome.same_substance_frequency_change.tolist() == [1, 0, -1]
    assert outcome.same_substance_frequency_reduction.tolist() == [1, 0, 0]
    audit = derive_audit_fields(pd.DataFrame({
        'EMPLOY': [1, 4], 'DETNLF': [0, 3], 'PSOURCE': [2, 7],
        'DETCRIM': [pd.NA, 4], 'SUB1': [1, 2], 'SUB2': [pd.NA, 2],
        'SUB3': [1, 3], 'REASON': [4, 1],
    }, dtype='Int64'))
    assert audit.employment_joint_observed.tolist() == [10, 43]
    assert audit.referral_joint_observed.tolist() == [20, 74]
    assert audit.known_non_none_primary_substance.tolist() == [0, 1]
    assert audit.additional_substance_partial_information.tolist() == [1, 0]


if __name__ == '__main__':
    _self_check()
