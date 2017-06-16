# import needed packages
import numpy as np, matplotlib.pylab as plt, pandas as pd, mpmath as mp, scipy.special as ss, sys
import scipy.stats as stats
from pdf_ccdf import pdf_ccdf

def find_jumps(series, jump_df):
    # find start of jump
    start_ind = series.jumps.loc[series.jumps == 1].index

    if start_ind.size:

        start_ind = start_ind[0]
        # find end of jump
        end_ind = series.jumps.ix[start_ind:].loc[series.jumps.ix[start_ind:] == -1].index

        if end_ind.size:

            end_ind = end_ind[0]
            # start and end dates and the difference between them
            start_date = series.date.ix[start_ind]
            end_date = series.date.ix[end_ind]
            jump_len = end_date - start_date
            if jump_df.end.size:
                recess_len = start_date - jump_df.end.ix[jump_df.end.size-1]
                # recess_len = start_date - jump_df.start.ix[jump_df.start.size-1]
            else:
                recess_len = start_date - series.date.ix[series.date.index[0]]
            # start and end discharge and the difference between them
            start_q = series.Q.ix[start_ind]
            end_q = series.Q.ix[end_ind]
            jump_q = float(end_q) - float(start_q)

            new_jump = pd.DataFrame({
                'start': [start_date],
                'start_ind': [start_ind],
                'end': [end_date],
                'end_ind': [end_ind],
                'jump_len': [jump_len],
                'recess_len': [recess_len],
                'Q1': [start_q],
                'Q2': [end_q],
                'jump_mag': [jump_q]
            })

            jump_df = jump_df.append(new_jump, ignore_index=True)
            series = series.loc[end_ind:]

            # call function again
            jump_df = find_jumps(series, jump_df)

    return jump_df


def sampler(sample, estimate):
    res = stats.probplot(sample, dist=stats.gamma, sparams=(estimate['alpha'], 0, estimate['beta']), fit=False, plot=None)
    r2 = (1. - np.sum((res[1] - res[0])**2) / np.sum(res[1]**2))**2
    return estimate['alpha'], r2, res


def assess_IRA(d, A_hat, B_hat, tw=100, axs=None):
    sys.setrecursionlimit(5000)
    Q = pd.DataFrame({'q':d.tolist()}).q.values
    dates = d.index
    jumps = np.asarray(Q.copy()).astype('float')
    # find jumps
    jumps[:-1] = np.diff(jumps)
    jumps[-1] = jumps[-2]
    # normalize jumps
    jumps = jumps/jumps.max()
    # select periods where discharge is increasing sufficiently fast
    jumps = np.where(jumps>0.005, 1, 0)
    # get derivative of boolean jumps vector
    jumps[1:] = np.diff(jumps)
    jumps[0] = 0
    # save to dataframe
    df = pd.DataFrame({'jumps': jumps, 'date': dates, 'Q': Q})
    # find jumps using new jumps vector
    jump_df = pd.DataFrame({'start': [], 'start_ind': [], 'end': [], 'end_ind': [], 'jump_len': [], 'recess_len': [], 'Q1': [], 'Q2': [], 'jump_mag': []})
    jump_df = find_jumps(df, jump_df)

    # get arrival spacing and jump magnitudes
    arrivals = (jump_df.recess_len.as_matrix()/864e11).astype(int) # Normalize from nanoseconds to days
    jump_mags = jump_df.jump_mag.as_matrix()

    storage_mags = np.zeros(jump_mags.size)
    if (2-B_hat) < 0:
        C = -1/(A_hat*(2.-B_hat))
        for ii in range(jump_mags.size):
            storage_mags[ii] = C*((np.asarray(jump_df.Q1[ii]).astype(float)**(2.-B_hat)) - (np.asarray(jump_df.Q2[ii]).astype(float)**(2.-B_hat)))

    elif (2-B_hat) >= 0:
        C = 1/(A_hat*(2.-B_hat))
        for ii in range(jump_mags.size):
            storage_mags[ii] = C*((np.asarray(jump_df.Q2[ii]).astype(float)**(2.-B_hat)) - (np.asarray(jump_df.Q1[ii]).astype(float)**(2.-B_hat)))

    storage_mags = storage_mags[np.isfinite(storage_mags)]
    arrivals = arrivals[arrivals<=tw]

    estimate_arrivals = {'alpha': 1, 'beta': arrivals.mean()}
    estimate_storage_mags = {'alpha': 1, 'beta': storage_mags.mean()}

    gamma_arr, r2_arr, res_arr = sampler(arrivals, estimate_arrivals)
    gamma_storage, r2_storage, res_storage = sampler(storage_mags, estimate_storage_mags)

    if axs != None:
        colors = {'arr': 'y', 'mags': 'b'}
        samples = {'arr': arrivals, 'mags': storage_mags}
        betas = {'arr': arrivals.mean(), 'mags': storage_mags.mean()}
        reses = {'arr': res_arr, 'mags': res_storage}
        r2s = {'arr': r2_arr, 'mags': r2_storage}
        strs = {'arr': 'Interarrival period (days)', 'mags': 'Magnitude of storage recharge (m^3 / day)'}

        for ii, key in enumerate(['arr', 'mags']):
            pdf, x_axis = pdf_ccdf(samples[key])
            axs[ii].loglog(x_axis, pdf, 'o', color=colors[key], alpha=0.5, label='Observed pdf')
            pdf_2 = stats.gamma.pdf(x_axis, 1, scale=betas[key])
            axs[ii].loglog(x_axis, pdf_2, 'k--', label='Estimated pdf')
            axs[ii].legend(loc=0)
            axs[ii].set_xlabel(strs[key])
            axs[ii].set_ylabel('Probability density')

            axs[ii+2].plot(reses[key][0], reses[key][1], 'o', color=colors[key], alpha=0.5)
            max_x = max([max(reses[key][0]), max(reses[key][1])])
            x = np.linspace(0, max_x+2, 10)
            axs[ii+2].plot(x, x, 'k', label='r2 = %g' % r2s[key])
            axs[ii+2].legend(loc=0)
            axs[ii+2].set_xlabel('Theoretical Quantiles')
            axs[ii+2].set_ylabel('Observed Quantiles')
            axs[ii+2].set_xlim([0, np.round(max_x+1)])
            axs[ii+2].set_ylim([0, np.round(max_x+1)])

    return jump_df, arrivals, storage_mags, r2_arr, r2_storage
