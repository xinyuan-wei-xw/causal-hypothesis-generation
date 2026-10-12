"""Approved substance matching; analytical outcome populations remain separate."""
import pandas as pd


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
    return result
