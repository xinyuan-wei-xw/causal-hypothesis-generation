"""Audit fixed-rule evaluations across imputations; no inferential pooling.

Identical rule hashes do not imply identical episode membership when split
variables are missing. Report that movement rather than forcing a modal leaf.
"""
import pandas as pd
import numpy as np
import joblib


def build_reference(forest, build_features, *, forest_training_ids, reserved_ids,
                    imputation_number, imputation_seed, forest_artifact_sha256,
                    max_depth=3, min_samples_leaf=100):
    """Build draw-1 summaries without accepting reserved outcomes or fitting a forest.

    Caller supplies a forest already fitted on draw 1 and its saved artifact hash.
    ID checks validate declared provenance, not the internals of an arbitrary forest.
    Tree seed 42, depth 3 and leaf minimum 100 retain the existing preview settings.
    """
    from econml.cate_interpreter import SingleTreeCateInterpreter, SingleTreePolicyInterpreter
    if imputation_number != 1 or imputation_seed != 42:
        raise ValueError('Reference must be imputation 1 under recorded seed 42.')
    _check_features(build_features)
    train=set(forest_training_ids);build=set(build_features.index);reserved=set(reserved_ids)
    if not train or not reserved or train & build or (train | build) & reserved:
        raise ValueError('Training, tree-build and reserved IDs must be disjoint and nonempty.')
    if len(forest_artifact_sha256)!=64 or any(c not in '0123456789abcdef' for c in forest_artifact_sha256):
        raise ValueError('Saved forest SHA-256 required.')
    x=build_features.to_numpy()
    cate=SingleTreeCateInterpreter(max_depth=max_depth,min_samples_leaf=min_samples_leaf,
                                  include_model_uncertainty=False,random_state=42).interpret(forest,x)
    policy=SingleTreePolicyInterpreter(max_depth=max_depth,min_samples_leaf=min_samples_leaf,
                                      include_model_uncertainty=False,random_state=42).interpret(forest,x,sample_treatment_costs=0)
    bundle={'cate':cate,'policy':policy,'feature_names':list(build_features.columns),
            'development_ids':sorted(train | build),'reserved_ids':sorted(reserved),
            'imputation_number':1,'imputation_seed':42,'forest_sha256':forest_artifact_sha256,
            'inference_validated':False}
    bundle['rule_hash']=_rule_hash(bundle)
    return bundle


def _check_features(x):
    if not isinstance(x,pd.DataFrame) or x.empty or not x.index.is_unique or x.index.hasnans or x.columns.duplicated().any():
        raise ValueError('Unique episode IDs and feature columns required.')
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError('Encoded features must be complete and finite.')


def _rule_hash(bundle):
    # Both tree structures and predicted policy values are part of the frozen rule.
    return joblib.hash((bundle['feature_names'],bundle['cate'].tree_model_,bundle['policy'].tree_model_),hash_name='sha1')


def apply_reference(bundle, evaluation_features):
    """Apply frozen rules; never refit, choose draws, or access outcomes."""
    _check_features(evaluation_features)
    if list(evaluation_features.columns)!=bundle['feature_names']:
        raise ValueError('Feature schema/order differs from the frozen tree.')
    ids=set(evaluation_features.index)
    if ids & set(bundle['development_ids']) or not ids <= set(bundle['reserved_ids']):
        raise ValueError('Only declared reserved episodes can be evaluated.')
    if _rule_hash(bundle)!=bundle['rule_hash']:
        raise ValueError('Frozen tree changed after construction.')
    x=evaluation_features.to_numpy()
    return pd.DataFrame({'episode_source_id':evaluation_features.index,
                         'leaf':bundle['cate'].tree_model_.apply(x),
                         'policy':bundle['policy'].treat(x),
                         'rule_hash':bundle['rule_hash']})


def audit_membership(evaluations, *, frozen_rule_hash, development_ids):
    """Each draw supplies episode_source_id, leaf, treatment, outcome, rule_hash.

    Require one row per reserved episode per draw, a shared frozen rule, unchanged
    observed treatment/outcome and no overlap with development/build observations.
    Leaves may change across draws. This audit does not supply valid MI intervals.
    """
    if len(evaluations) < 2 or not frozen_rule_hash:
        raise ValueError('At least two imputations and a frozen rule hash required.')
    required={'episode_source_id','leaf','treatment','outcome','rule_hash'}
    reference=None; tables=[]; support=[]
    for draw, frame in evaluations.items():
        if not required.issubset(frame.columns) or frame.empty:
            raise ValueError('Missing required columns or empty evaluation.')
        if frame[list(required)].isna().any().any() or frame.episode_source_id.duplicated().any():
            raise ValueError('Missing values or duplicate episode IDs.')
        if not frame.rule_hash.eq(frozen_rule_hash).all():
            raise ValueError('Different tree rules cannot be pooled as fixed leaves.')
        if set(frame.episode_source_id) & set(development_ids):
            raise ValueError('Development/build episodes overlap evaluation.')
        if not frame.treatment.isin([0,1]).all() or not frame.outcome.isin([0,1]).all():
            raise ValueError('Observed binary treatment/outcome required.')
        f=frame.set_index('episode_source_id').sort_index()
        if reference is None:
            reference=f[['treatment','outcome']]
        elif not f.index.equals(reference.index) or not f[['treatment','outcome']].equals(reference):
            raise ValueError('Episode membership or observed treatment/outcome changed.')
        tables.append(f.leaf.rename(draw))
        for leaf,g in f.groupby('leaf'):
            support.append({'draw':draw,'leaf':leaf,'n':len(g),'treated':int(g.treatment.sum()),'control':int((1-g.treatment).sum())})
    assignments=pd.concat(tables,axis=1)
    episodes=pd.DataFrame({'distinct_leaves':assignments.nunique(axis=1)})
    episodes['membership_changes']=episodes.distinct_leaves.gt(1)
    return {'episode_audit':episodes.reset_index(), 'support_by_draw':pd.DataFrame(support),
            'summary':{'draws':len(evaluations),'episodes':len(episodes),
                       'membership_changes':int(episodes.membership_changes.sum()),
                       'membership_change_fraction':float(episodes.membership_changes.mean()),
                       'inference_validated':False}}
