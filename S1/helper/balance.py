"""Descriptive covariate balance with a fixed unweighted reference scale."""
import numpy as np
import pandas as pd


def balance_table(x, treatment, weights=None, *, reference_scale=None):
    """Compare weighted group means; the reference scale never changes with weights.

    Uses population variances within each unweighted treatment group for the
    baseline denominator. Constant covariates are identified explicitly.
    This is a descriptive diagnostic, not an effect estimator or significance test.
    """
    x = pd.DataFrame(x).astype(float)
    t = np.asarray(treatment)
    if t.shape != (len(x),) or set(np.unique(t)) != {0, 1}:
        raise ValueError('Both binary treatment groups are required.')
    w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != t.shape or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError('Weights must be finite, nonnegative, and row-aligned.')
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError('Balance inputs must be fully observed and finite.')
    if any(w[t == g].sum() <= 0 for g in [0, 1]):
        raise ValueError('Both groups must have positive total weight.')
    scale = np.sqrt((x.loc[t == 1].var(ddof=0) + x.loc[t == 0].var(ddof=0)) / 2)
    if reference_scale is not None:
        scale = pd.Series(reference_scale).reindex(x.columns)
        if scale.isna().any() or (scale < 0).any() or not np.isfinite(scale).all():
            raise ValueError('Reference scales must cover every covariate.')
    means = [np.average(x.to_numpy()[t == g], axis=0, weights=w[t == g]) for g in [0, 1]]
    difference = means[1] - means[0]
    # Globally constant columns have identical arm means despite weighted-sum rounding.
    constant = x.nunique(dropna=False).eq(1).to_numpy()
    difference[constant] = 0.0
    smd = np.divide(difference, scale.to_numpy(), out=np.full(len(scale), np.nan), where=scale.to_numpy() > 0)
    smd[(scale.to_numpy() == 0) & (difference == 0)] = 0
    return pd.DataFrame({'variable': x.columns, 'control_mean': means[0], 'treated_mean': means[1],
                         'difference': difference, 'reference_scale': scale.to_numpy(),
                         'smd': smd, 'abs_smd': np.abs(smd), 'zero_reference_scale': scale.to_numpy() == 0})


def weight_summary(treatment, weights):
    t, w = np.asarray(treatment), np.asarray(weights, dtype=float)
    if t.shape != w.shape or set(np.unique(t)) != {0, 1} or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError('Expected aligned binary treatment and finite nonnegative weights.')
    rows = []
    for group in [0, 1]:
        a = w[t == group]
        if a.sum() <= 0:
            raise ValueError('Each group requires positive weight.')
        rows.append({'treatment': group, 'n': len(a), 'positive_weight_n': int((a > 0).sum()),
                     'weight_sum': a.sum(), 'effective_sample_size': a.sum() ** 2 / np.square(a).sum(),
                     'max_weight': a.max(), 'p99_weight': np.quantile(a, .99)})
    return pd.DataFrame(rows)


def plot_propensity_distributions(scores, treatment, weights, *, labels, title, score_label):
    """Show arm-normalized score densities before and after overlap weighting."""
    import matplotlib.pyplot as plt

    e = np.asarray(scores, dtype=float)
    t = np.asarray(treatment)
    w = np.asarray(weights, dtype=float)
    if e.ndim != 1 or t.shape != e.shape or w.shape != e.shape or set(np.unique(t)) != {0, 1}:
        raise ValueError('Expected aligned scores, binary treatment, and weights.')
    if not np.isfinite(e).all() or (e <= 0).any() or (e >= 1).any():
        raise ValueError('Propensity scores must lie strictly between zero and one.')
    if not np.isfinite(w).all() or (w < 0).any() or any(w[t == arm].sum() <= 0 for arm in (0, 1)):
        raise ValueError('Both arms need finite, positive total weight.')
    bins = np.linspace(0, 1, 41)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for arm in (0, 1):
        mask = t == arm
        for ax, arm_weights in zip(axes, (None, w[mask])):
            ax.hist(e[mask], bins=bins, weights=arm_weights, density=True,
                    histtype='step', linewidth=1.8, label=f'{labels[arm]} (n={mask.sum():,})')
    for ax, heading in zip(axes, ('Original episodes', 'Overlap-weighted target')):
        ax.axvline(.05, color='0.5', linestyle=':', linewidth=1)
        ax.axvline(.95, color='0.5', linestyle=':', linewidth=1)
        ax.set(xlim=(0, 1), xlabel=score_label, title=heading)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Within-arm density')
    fig.suptitle(title)
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    fig = plot_propensity_distributions([.2, .3, .7, .8], [0, 0, 1, 1],
                                        [.2, .3, .3, .2], labels=('Control', 'Treated'),
                                        title='Smoke check', score_label='Propensity score')
    assert len(fig.axes) == 2
    plt.close(fig)
