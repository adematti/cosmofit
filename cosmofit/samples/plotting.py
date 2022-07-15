import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec

from cosmofit import utils
from cosmofit.plotting import savefig
from . import diagnostics


def _make_list(obj, length=None, default=None):
    """
    Return list from ``obj``.

    Parameters
    ----------
    obj : object, tuple, list, array
        If tuple, list or array, cast to list.
        Else return list of ``obj`` with length ``length``.

    length : int, default=1
        Length of list to return.

    Returns
    -------
    toret : list
    """
    if utils.is_sequence(obj):
        obj = list(obj)
        if length is not None:
            obj += [default] * (length - len(obj))
    else:
        obj = [obj]
        if length is not None:
            obj *= length
    return obj


def _get_default_chain_params(chains, varied=True, output=False, **kwargs):
    list_params = [chain.names(varied=varied, output=output, **kwargs) for chain in chains]
    return [params for params in list_params[0] if all(params in lparams for lparams in list_params[1:])]


def plot_trace(chains, params=None, figsize=None, colors=None, labelsize=None, fn=None, kw_plot=None, kw_save=None):
    """
    Make trace plot.

    Parameters
    ----------
    chains : list, default=None
        List of :class:`Chain` instances.

    params : list, ParameterCollection
        The parameter names.

    fn : string, default=None
        If not ``None``, file name where to save figure.

    Returns
    -------
    lax : array
        Array of axes.
    """
    chains = _make_list(chains)
    if params is None:
        params = _get_default_chain_params(chains)
    params = _make_list(params)
    nparams = len(params)
    colors = _make_list(colors, length=len(chains), default=None)
    kw_plot = kw_plot or {'alpha': 0.2}

    steps = 1 + np.arange(max(chain.size for chain in chains))
    figsize = figsize or (8, 1.5 * nparams)
    fig, lax = plt.subplots(nparams, sharex=True, sharey=False, figsize=figsize, squeeze=False)
    lax = lax.ravel()

    for ax, param in zip(lax, params):
        ax.grid(True)
        ax.set_ylabel(chains[0][param].param.latex(inline=True), fontsize=labelsize)
        ax.set_xlim(steps[0], steps[-1])
        for ichain, chain in enumerate(chains):
            ax.plot(steps, chain[param].ravel(), color=colors[ichain], **kw_plot)

    lax[-1].set_xlabel('step', fontsize=labelsize)

    if fn is not None:
        savefig(fn, **(kw_save or {}))
    return lax


def plot_gelman_rubin(chains, params=None, multivariate=False, threshold=None, slices=None, labelsize=None, ax=None, fn=None, kw_save=None, **kwargs):
    """
    Plot Gelman-Rubin statistics.

    Parameters
    ----------
    chains : list
        List of :class:`Chain` instances.

    params : list, ParameterCollection
        The parameter names.

    multivariate : bool, default=False
        If ``True``, add line for maximum of eigen value of Gelman-Rubin matrix.
        See :meth:`Samples.gelman_rubin`.

    threshold : float, default=1.1
        If not ``None``, plot horizontal line at this value.

    slices : list, array
        List of increasing number of steps to include in calculation of Gelman-Rubin statistics.

    ax : matplotlib.axes.Axes, default=None
        Axes where to plot samples. If ``None``, takes current axes.

    fn : string, default=None
        If not ``None``, file name where to save figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if params is None:
        params = _get_default_chain_params(chains)
    params = [str(param) for param in _make_list(params)]
    if slices is None:
        nsteps = np.min([chain.size for chain in chains])
        slices = np.arange(100, nsteps, 500)
    gr_multi = []
    gr = {param: [] for param in params}
    for end in slices:
        chains_sliced = [chain.ravel()[:end] for chain in chains]
        if multivariate: gr_multi.append(diagnostics.gelman_rubin(chains_sliced, params, method='eigen', **kwargs).max())
        for param in gr: gr[param].append(diagnostics.gelman_rubin(chains_sliced, param, method='diag', **kwargs))
    for param in gr: gr[param] = np.asarray(gr[param])

    fig = None
    if ax is None: fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_xlabel('step', fontsize=labelsize)
    ax.set_ylabel(r'$\hat{R}$', fontsize=labelsize)

    if multivariate: ax.plot(slices, gr_multi, label='multi', linestyle='-', linewidth=1, color='k')
    for param in params:
        ax.plot(slices, gr[param], label=chains[0][param].param.latex(inline=True), linestyle='--', linewidth=1)
    if threshold is not None: ax.axhline(y=threshold, xmin=0., xmax=1., linestyle='--', linewidth=1, color='k')
    ax.legend()

    if fn is not None:
        savefig(fn, fig=fig, **(kw_save or {}))
    return ax


def plot_geweke(chains, params=None, threshold=None, slices=None, labelsize=None, ax=None, fn=None, kw_save=None, **kwargs):
    """
    Plot Geweke statistics.

    Parameters
    ----------
    chains : list
        List of :class:`Chain` instances.

    params : list, ParameterCollection
        The parameter names.

    threshold : float, default=1.1
        If not ``None``, plot horizontal line at this value.

    slices : list, array
        List of increasing number of steps to include in calculation of Gelman-Rubin statistics.

    ax : matplotlib.axes.Axes, default=None
        Axes where to plot samples. If ``None``, takes current axes.

    fn : string, default=None
        If not ``None``, file name where to save figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if params is None:
        params = _get_default_chain_params(chains)
    params = [str(param) for param in _make_list(params)]
    if slices is None:
        nsteps = np.min([chain.size for chain in chains])
        slices = np.arange(100, nsteps, 500)
    geweke = {param: [] for param in params}
    for end in slices:
        chains_sliced = [chain.ravel()[:end] for chain in chains]
        for param in geweke: geweke[param].append(diagnostics.geweke(chains_sliced, param, **kwargs))
    for param in geweke: geweke[param] = np.asarray(geweke[param]).mean(axis=-1)

    fig = None
    if ax is None: fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_xlabel('step', fontsize=labelsize)
    ax.set_ylabel(r'geweke', fontsize=labelsize)

    for param in params:
        ax.plot(slices, geweke[param], label=chains[0][param].param.latex(inline=True), linestyle='-', linewidth=1)
    if threshold is not None: ax.axhline(y=threshold, xmin=0., xmax=1., linestyle='--', linewidth=1, color='k')
    ax.legend()

    if fn is not None:
        savefig(fn, fig=fig, **(kw_save or {}))
    return ax


def plot_autocorrelation_time(chains, params=None, threshold=50, slices=None, labelsize=None, ax=None, fn=None, kw_save=None):
    r"""
    Plot integrated autocorrelation time.

    Parameters
    ----------
    chains : list
        List of :class:`Chain` instances.

    params : list, ParameterCollection
        The parameter names.

    threshold : float, default=50
        If not ``None``, plot :math:`y = x/\mathrm{threshold}` line.
        Integrated autocorrelation time estimation can be considered reliable when falling under this line.

    slices : list, array
        List of increasing number of steps to include in calculation of autocorrelation time.

    ax : matplotlib.axes.Axes, default=None
        Axes where to plot samples. If ``None``, takes current axes.

    fn : string, default=None
        If not ``None``, file name where to save figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    chains = _make_list(chains)
    if params is None:
        params = _get_default_chain_params(chains)
    params = [str(param) for param in _make_list(params)]
    if slices is None:
        nsteps = np.min([chain.size for chain in chains])
        slices = np.arange(100, nsteps, 500)
    autocorr = {param: [] for param in params}
    for end in slices:
        chains_sliced = [chain.ravel()[:end] for chain in chains]
        for param in autocorr:
            tmp = diagnostics.integrated_autocorrelation_time(chains_sliced, param)
            autocorr[param].append(tmp)
    for param in autocorr: autocorr[param] = np.asarray(autocorr[param])

    fig = None
    if ax is None: fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_xlabel('step $N$', fontsize=labelsize)
    ax.set_ylabel('$\tau$', fontsize=labelsize)

    for param in params:
        ax.plot(slices, autocorr[param], label=chains[0][param].param.latex(inline=True), linestyle='--', linewidth=1)
    if threshold is not None:
        ax.plot(slices, slices * 1. / threshold, label='$N/{:d}$'.format(threshold), linestyle='--', linewidth=1, color='k')
    ax.legend()

    if fn is not None:
        savefig(fn, fig=fig, **(kw_save or {}))
    return ax


def plot_triangle(chains, params=None, labels=None, fn=None, kw_save=None, **kwargs):
    """
    Triangle plot.

    Parameters
    ----------
    chains : list
        List of :class:`Chain` instances.

    params : list, ParameterCollection
        The parameter names.

    fn : string, default=None
        If not ``None``, file name where to save figure.

    Returns
    -------
    lax : array
        Array of axes.
    """
    from getdist import plots
    g = plots.get_subplot_plotter()
    chains = _make_list(chains)
    labels = _make_list(labels, length=len(chains), default=None)
    if params is None:
        params = _get_default_chain_params(chains)
    params = [str(param) for param in _make_list(params)]
    chains = [chain.to_getdist(label=label) for chain, label in zip(chains, labels)]
    lax = g.triangle_plot(chains, params, **kwargs)
    if fn is not None:
        savefig(fn, **(kw_save or {}))
    return lax


def _get_default_profiles_params(profiles, varied=True, output=False, **kwargs):
    list_params = [profile.bestfit.names(varied=varied, output=output, **kwargs) for profile in profiles]
    return [params for params in list_params[0] if all(params in lparams for lparams in list_params[1:])]


def plot_aligned(profiles, param, ids=None, labels=None, colors=None, truth=None, errors='parabolic_errors',
                 labelsize=None, ticksize=None, kw_scatter=None, yband=None, kw_mean=None, kw_truth=None, kw_yband=None,
                 kw_legend=None, ax=None, fn=None, kw_save=None):
    """
    Plot best fit estimates for single parameter.

    Parameters
    ----------
    profiles : list
        List of :class:`Profiles` instances.

    param : Parameter, string
        Parameter name.

    ids : list, string
        Label(s) for input profiles.

    labels : list, string
        Label(s) for best fits within each :class:`Profiles` instance.

    truth : float, string, default=None
        Plot this truth / reference value for parameter.
        If ``'value'``, take :attr:`Parameter.value`.

    yband : float, tuple, default=None
        If not ``None``, plot horizontal band.
        If tuple and last element set to ``'abs'``,
        absolute lower and upper y-coordinates of band;
        lower and upper fraction around truth.
        If float, fraction around truth.

    ax : matplotlib.axes.Axes, default=None
        Axes where to plot profiles. If ``None``, takes current axes.

    fn : string, default=None
        If not ``None``, file name where to save figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    profiles = _make_list(profiles)
    if truth is None and kw_truth is not None:
        truth = profiles[0].bestfit[param].param.value
    kw_truth = kw_truth or {}
    maxpoints = max(map(lambda prof: len(prof.bestfit), profiles))
    ids = _make_list(ids, length=len(profiles), default=None)
    labels = _make_list(labels, length=maxpoints, default=None)
    colors = _make_list(colors, length=maxpoints, default=None)
    add_legend = any(label is not None for label in labels)
    add_mean = kw_mean is not None
    if add_mean:
        kw_mean = kw_mean if isinstance(kw_mean, dict) else {'marker': 'o'}
    kw_scatter = dict(kw_scatter or {'marker': 'o'})
    kw_yband = dict(kw_yband or {})
    kw_legend = dict(kw_legend or {})

    xmain = np.arange(len(profiles))
    xaux = np.linspace(-0.15, 0.15, maxpoints)
    fig = None
    if ax is None: fig, ax = plt.subplots()
    for iprof, prof in enumerate(profiles):
        if param not in prof.bestfit: continue
        ibest = prof.bestfit.logposterior.argmax()
        for ipoint, point in enumerate(prof.bestfit[param]):
            yerr = None
            if errors:
                try:
                    yerr = prof.get(errors)[param]
                except IndexError:
                    yerr = None
                if len(yerr) == 1:
                    yerr = yerr[0]  # only for best fit
                else:
                    yerr = yerr[ibest]
            label = labels[ipoint] if iprof == 0 else None
            ax.errorbar(xmain[iprof] + xaux[ipoint], point, yerr=yerr, color=colors[ipoint], label=label, linestyle='none', **kw_scatter)
        if add_mean:
            ax.errorbar(xmain[iprof], prof.bestfit[param].mean(), yerr=prof.bestfit[param].std(ddof=1), linestyle='none', **kw_mean)
    if truth is not None:
        ax.axhline(y=truth, xmin=0., xmax=1., **kw_truth)
    if yband is not None:
        if np.ndim(yband) == 0:
            yband = (yband, yband)
        if yband[-1] == 'abs':
            low, up = yband[0], yband[1]
        else:
            if truth is None:
                raise ValueError('Plotting relative band requires truth value.')
            low, up = truth * (1 - yband[0]), truth * (1 + yband[1])
        ax.axhspan(low, up, **kw_yband)

    ax.set_xticks(xmain)
    ax.set_xticklabels(ids, rotation=40, ha='right', fontsize=ticksize)
    ax.grid(True, axis='y')
    ax.set_ylabel(profiles[0].bestfit[param].param.latex(inline=True), fontsize=labelsize)
    ax.tick_params(labelsize=ticksize)
    if add_legend: ax.legend(**{**{'ncol': maxpoints}, **kw_legend})
    if fn is not None:
        savefig(fn, fig=fig, **(kw_save or {}))
    return ax


def plot_aligned_stacked(profiles, params=None, ids=None, labels=None, truths=None, ybands=None, ylimits=None, figsize=None, fn=None, kw_save=None, **kwargs):
    """
    Plot best fit estimates for several parameters.

    Parameters
    ----------
    profiles : list, default=None
        List of :class:`Profiles` instances.

    param : Parameter, string
        Parameter name.

    ids : list, string
        Label(s) for input profiles.

    labels : list, string
        Label(s) for best fits within each :class:`Profiles` instance.

    truths : list, default=None
        Plot this truths / reference values for parameters.
        See :meth:`get_default_truths`.

    ybands : list, default=None
        If not ``None``, plot horizontal bands.
        See :meth:`plot_aligned`.

    ylimits : list, default=None
        If not ``None``, limits  for y axis.

    filename : string, default=None
        If not ``None``, file name where to save figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure.

    lax : array
        Array of axes.
    """
    if params is None:
        params = _get_default_profiles_params(profiles)
    params = [str(param) for param in _make_list(params)]
    truths = _make_list(truths, length=len(params), default=None)
    ybands = _make_list(ybands, length=len(params), default=None)
    ylimits = _make_list(ybands, length=len(params), default=None)
    maxpoints = max(map(lambda prof: len(prof.bestfit), profiles))

    nrows = len(params)
    ncols = len(profiles) if len(profiles) > 1 else maxpoints
    figsize = figsize or (ncols, 3. * nrows)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows, 1, wspace=0.1, hspace=0.1)

    lax = []
    for iparam1, param1 in enumerate(params):
        ax = plt.subplot(gs[iparam1])
        plot_aligned(profiles, param=param1, ids=ids, labels=labels, truth=truths[iparam1], yband=ybands[iparam1], ax=ax, **kwargs)
        if (iparam1 < nrows - 1) or not ids: ax.get_xaxis().set_visible(False)
        ax.set_ylim(ylimits[iparam1])
        if iparam1 != 0:
            leg = ax.get_legend()
            if leg is not None: leg.remove()
        lax.append(ax)

    if fn is not None:
        savefig(fn, fig=fig, **(kw_save or {}))
    return np.array(lax)
