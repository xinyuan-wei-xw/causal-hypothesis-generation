"""Approved representations; these functions do not select an imputation protocol."""
import pandas as pd

PAIRS=((0,0),(1,0),(2,0),(0,1),(1,1),(0,2))
PAIR_LABELS=[f'known_{k}_unknown_{u}' for k,u in PAIRS]

def additional_substance_category(known,unknown):
    """Preserve partial information as six nominal categories, with missing inputs kept missing."""
    if not known.index.equals(unknown.index):
        raise ValueError('Known and unknown counts must have matching episode indices.')
    k=pd.to_numeric(known,errors='raise');u=pd.to_numeric(unknown,errors='raise')
    for values in [k,u]:
        if (~values.dropna().isin([0,1,2])).any():raise ValueError('Substance-slot counts must be integers from zero to two.')
    valid=k.notna()&u.notna()
    if ((k+u)>2).loc[valid].any():raise ValueError('Known additional substances plus unknown slots cannot exceed two.')
    result=pd.Series(pd.NA,index=k.index,dtype='string')
    for a,b in PAIRS:result.loc[valid&k.eq(a)&u.eq(b)]=f'known_{a}_unknown_{b}'
    return pd.Series(pd.Categorical(result,categories=PAIR_LABELS,ordered=False),index=k.index,name='additional_substance_information_category')

def conditional_detail_status(parent,detail,*,applicable_code,parent_codes,detail_codes):
    """Audit observation/applicability status; not an imputation or final model encoding.

    Source fields must be retained. Observed detail under an inapplicable or unknown
    parent is flagged, not used to change the parent.
    """
    if not parent.index.equals(detail.index):raise ValueError('Parent/detail indices must match.')
    if not parent.dropna().isin(parent_codes).all() or not detail.dropna().isin(detail_codes).all():
        raise ValueError('Unexpected parent or detail source code.')
    status=pd.Series('applicability_unknown',index=parent.index,dtype='string')
    applicable=parent.eq(applicable_code)&parent.notna();inapplicable=parent.notna()&~applicable
    status.loc[inapplicable]='not_applicable'
    status.loc[applicable&detail.isna()]='applicable_missing'
    for code in detail_codes:status.loc[applicable&detail.eq(code)]=f'detail_{code}'
    return pd.DataFrame({'status':status,'detail_present_when_not_applicable':inapplicable&detail.notna(),'detail_present_with_unknown_parent':parent.isna()&detail.notna()},index=parent.index)
