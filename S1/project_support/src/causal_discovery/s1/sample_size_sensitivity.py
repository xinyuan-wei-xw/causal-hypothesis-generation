"""Development sample-size stability, not an endogeneity or causal-validity test.

The same comparison features are used at every size. Their outcomes never enter
fitting or policy-rule selection. Reported CATE summaries are model predictions,
not independently estimated causal effects or validated policy values.
"""
import hashlib
import time
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def nested_samples(frame, fractions, seeds, strata):
    """Reproducible nested sampling within treatment/setting strata.

    Sort IDs before shuffling so input row order cannot change membership. Small
    strata may contribute zero rows at small fractions; never silently oversample
    them. Actual sizes and treatment counts must therefore be reported.
    """
    if not frame.index.is_unique or frame.index.hasnans or frame.empty:
        raise ValueError('Nonempty data with unique, observed episode IDs required.')
    fractions = sorted(set(fractions))
    if not fractions or fractions[-1] != 1 or any(not 0 < f <= 1 for f in fractions):
        raise ValueError('Fractions must be in (0, 1] and include the full reference 1.')
    if not seeds or len(set(seeds)) != len(seeds) or not strata:
        raise ValueError('Distinct seeds and explicit strata required.')
    if frame[strata].isna().any().any():
        raise ValueError('Strata must be observed; define missingness handling first.')
    d = frame.sort_index()
    for seed in seeds:
        rng = np.random.default_rng(seed)
        groups = [rng.permutation(v) for v in d.groupby(strata, observed=True, sort=True).indices.values()]
        for fraction in fractions:
            positions = np.concatenate([g[:int(np.floor(len(g)*fraction))] for g in groups])
            yield seed, fraction, d.iloc[positions].sort_index()


def run_stability(development, comparison, *, fractions, seeds, strata,
                  fit_predict, output, model_seed=123):
    """Run an interchangeable estimator callback; preserve every membership.

    fit_predict(train, comparison_features, model_seed) returns one row per
    comparison ID with cate, policy and leaf. The callback must fit preprocessing
    on train only, and build its tree without using comparison features/outcomes.
    Model seed stays fixed: repetitions vary sampling, not algorithm randomness.
    """
    from pathlib import Path
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    if not comparison.index.is_unique or comparison.index.hasnans or comparison.empty:
        raise ValueError('Comparison IDs must be unique, observed and nonempty.')
    if len(development.index.intersection(comparison.index)):
        raise ValueError('Development and comparison IDs overlap.')
    features = comparison.drop(columns=['outcome', 'treatment'], errors='ignore').sort_index()
    pd.Series(features.index, name='episode_source_id').to_csv(output/'comparison_ids.csv', index=False)
    rows = []
    for seed in seeds:
        predictions = {}; seed_rows = []
        for _, fraction, train in nested_samples(development, fractions, [seed], strata):
            folder = output/f'seed_{seed}_fraction_{fraction:g}'; folder.mkdir()
            train.index.to_series(name='episode_source_id').to_csv(folder/'development_ids.csv', index=False)
            if set(train.treatment.unique()) != {0, 1}:
                raise ValueError('A sampled development set lacks a treatment arm.')
            started = time.monotonic()
            pred = fit_predict(train, features.copy(), model_seed)
            if not pred.index.equals(features.index) or not {'cate','policy','leaf'} <= set(pred):
                raise ValueError('Callback must return aligned cate/policy/leaf predictions.')
            if not np.isfinite(pred[['cate','policy','leaf']].to_numpy()).all():
                raise ValueError('Nonfinite predictions.')
            pred.to_csv(folder/'comparison_predictions.csv', index_label='episode_source_id')
            predictions[fraction] = pred
            row = dict(seed=seed, fraction=fraction, n_development=len(train),
                       n_treated=int(train.treatment.sum()), n_control=int((train.treatment==0).sum()),
                       mean_predicted_effect=float(pred.cate.mean()), seconds=time.monotonic()-started,
                       membership_sha256=hashlib.sha256((folder/'development_ids.csv').read_bytes()).hexdigest())
            seed_rows.append(row)
        # Node numbers are arbitrary across trees: adjusted Rand compares partitions
        # without requiring identical numeric leaf labels. Policy agreement compares
        # actual recommendations; neither quantity is a significance test.
        reference = predictions[1.0]
        for row in seed_rows:
            pred = predictions[row['fraction']]
            row.update(cate_rmse_vs_full=float(np.sqrt(np.mean((pred.cate-reference.cate)**2))),
                       policy_agreement_vs_full=float((pred.policy==reference.policy).mean()),
                       subgroup_adjusted_rand_vs_full=float(adjusted_rand_score(reference.leaf,pred.leaf)))
        rows.extend(seed_rows)
        pd.DataFrame(rows).to_csv(output/'stability_metrics.csv', index=False)
    return pd.DataFrame(rows)


def forest_preview(train, comparison_features, seed):
    """Fixed-setting complete-input adapter for feasibility checks only.

    This deliberately does not inherit the existing runner's adaptive trimming or
    tuned settings: it is NOT a sensitivity result against the final main model.
    Before research execution, supply a callback matching the frozen main protocol,
    including its MI, overlap, tree-cost and uncertainty procedures.
    """
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from econml.dml import CausalForestDML
    from econml.cate_interpreter import SingleTreePolicyInterpreter
    fields = list(comparison_features.columns)
    if train[fields+['outcome','treatment']].isna().any().any() or comparison_features.isna().any().any():
        raise ValueError('Preview requires complete inputs; no implicit deletion or imputation.')
    # All preparation is fitted separately inside each subsample.
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    z = enc.fit_transform(train[fields]); ze = enc.transform(comparison_features)
    names = enc.get_feature_names_out(fields)
    state = np.array([n.startswith('STFIPS_') for n in names])
    x, xe = z[:,~state], ze[:,~state]
    w = z[:,state] if state.any() else None
    est = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=30,max_depth=4,n_jobs=1,random_state=seed),
        model_t=RandomForestClassifier(n_estimators=30,max_depth=4,n_jobs=1,random_state=seed),
        n_estimators=40,min_samples_leaf=10,cv=2,honest=True,
        discrete_treatment=True,inference=False,n_jobs=1,random_state=seed)
    est.fit(train.outcome.to_numpy(),train.treatment.to_numpy(),X=x,W=w)
    tree = SingleTreePolicyInterpreter(max_depth=2,min_samples_leaf=20,
                                      include_model_uncertainty=False,random_state=seed)
    # Zero is a fixed illustrative threshold, NOT the current main-run ATE cost.
    # Building rules on development X leaves comparison records out of discovery.
    tree.interpret(est,x,sample_treatment_costs=0.0)
    return pd.DataFrame({'cate':est.effect(xe).ravel(), 'policy':tree.treat(xe),
                         'leaf':tree.tree_model_.apply(xe)},index=comparison_features.index)
