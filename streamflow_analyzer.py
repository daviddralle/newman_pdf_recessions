# import needed packages
import numpy as np, matplotlib.pylab as plt, pandas as pd, mpmath as mp, scipy.special as ss
import sys, urllib2, scipy.stats as stats, os, glob, scipy.integrate as sint
from peakdetect import peakdet as peakdet
from pdf_ccdf import pdf_ccdf
from scipy.optimize import curve_fit
from scipy.special import gammainc as lower_gamma
from scipy.special import gammaincc as upper_gamma
from ars import ARS
from statsmodels.base.model import GenericLikelihoodModel


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
    # res = stats.probplot(sample, dist=stats.gamma, sparams=(estimate['alpha'], 0, estimate['beta']), fit=False, plot=None)
    # r2 = (1. - np.sum((res[1] - res[0])**2) / np.sum(res[1]**2))**2
    # return estimate['alpha'], r2, res

    alpha = estimate['alpha']
    beta = estimate['beta']
    Qe = np.log(1. - np.linspace(0.,1.,sample.size))[:-1]
    Qt = np.log(1. - stats.gamma.cdf(np.sort(sample), alpha, loc=0, scale=beta))[:-1]
    return 1. - (np.nansum((Qt - Qe)**2) / np.nansum((Qe - Qe.mean())**2)), Qe, Qt


def assess_IRA(sample, A_hat, B_hat, tw=100, axs=None):
    sys.setrecursionlimit(5000)
    Q = pd.DataFrame({'q': sample})
    dates = Q.index
    jumps = sample.copy()
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
    df = pd.DataFrame({'jumps': jumps, 'date': dates, 'Q': sample})
    # find jumps using new jumps vector
    jump_df = pd.DataFrame({'start': [], 'start_ind': [], 'end': [], 'end_ind': [], 'jump_len': [], 'recess_len': [], 'Q1': [], 'Q2': [], 'jump_mag': []})
    jump_df = find_jumps(df, jump_df)

    # get arrival spacing and jump magnitudes
    arrivals = (jump_df.recess_len.as_matrix()).astype(int) # Normalize from nanoseconds to days
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

    if arrivals.size == 0:
        r2_arr, Qe_arr, Qt_arr = np.nan, np.nan, np.nan
    else:
        r2_arr, Qe_arr, Qt_arr = sampler(arrivals, estimate_arrivals)

    if storage_mags.size == 0:
        r2_storage, Qe_mags, Qt_mags = np.nan, np.nan, np.nan
    else:
        r2_storage, Qe_mags, Qt_mags = sampler(storage_mags, estimate_storage_mags)

    if axs != None:
        colors = {'arr': 'y', 'mags': 'b'}
        samples = {'arr': arrivals, 'mags': storage_mags}
        betas = {'arr': arrivals.mean(), 'mags': storage_mags.mean()}
        Qe = {'arr': Qe_arr, 'mags': Qe_mags}
        Qt = {'arr': Qt_arr, 'mags': Qt_mags}
        r2s = {'arr': r2_arr, 'mags': r2_storage}
        strs = {'arr': 'Interarrival period (days)', 'mags': 'Magnitude of storage recharge (m^3 / day)'}

        for ii, key in enumerate(['arr', 'mags']):
            if samples[key][np.isfinite(samples[key])].size == 0:
                continue
            else:
                pdf, x_axis = pdf_ccdf(samples[key], n=100, output='ccdf')
                axs[ii].loglog(x_axis, pdf, 'o', color=colors[key], alpha=0.5, label='Observed pdf')
                pdf_2 = 1. - stats.gamma.cdf(x_axis, 1, scale=betas[key])
                axs[ii].loglog(x_axis, pdf_2, 'k--', label='Estimated pdf')
                axs[ii].legend(loc=0)
                axs[ii].set_xlabel(strs[key])
                axs[ii].set_ylabel('Probability density')

                axs[ii+2].loglog(np.sort(samples[key][:-1]), Qt[key]/Qe[key], 'o', color=colors[key], alpha=0.5)
                # max_x = 0
                # min_x = -5
                x = np.linspace(samples[key].min(), np.sort(samples[key])[-1], 10)
                axs[ii+2].plot(x, np.ones(10), 'k', label='r2 = %g' % r2s[key])
                axs[ii+2].legend(loc=0)
                axs[ii+2].set_xlabel('Theoretical Quantiles')
                axs[ii+2].set_ylabel('Observed Quantiles')
                # axs[ii+2].set_xlim([min_x, max_x])
                axs[ii+2].set_ylim([0.1, 10])

    return jump_df, arrivals, storage_mags, r2_arr, r2_storage


def g(q,p):
    if np.size(np.array(q)) == 1: return np.exp(np.sum([p[i]*np.log(q)**(len(p)-i-1) for i in range(len(p))]))
    return [np.exp(np.sum([p[i]*np.log(qq)**(len(p)-i-1) for i in range(len(p))])) for qq in np.array(q)]


def KirchnerG(q,a,b,c):
    return np.exp(a*np.log(q)**2 + b*np.log(q) + c)


def KirchnerBinning(df, loud=False):
    df = df.sort_values(by='q',ascending=False)

    logQ = np.array(np.log(df.q))

    logRange = np.max(logQ) - np.min(logQ)
    minBinSize = logRange*.01

    binBoundaries = [0]
    for i in range(1,len(df)):
        if abs(logQ[i] - logQ[binBoundaries[-1]]) < minBinSize:
            if loud: print('Bin too small')
            continue

        curr = df.Dunsmooth[binBoundaries[-1]:i]
        if np.std(-curr)/np.sqrt(abs(i-binBoundaries[-1])) > np.mean(-curr)/2:
            if loud: print('Bin too heterogeneous')
            continue

        binBoundaries.append(i)

    return binBoundaries


def _finditem(obj, key):
    if key in obj: return obj[key]
    for k, v in obj.items():
        if isinstance(v,dict):
            return _finditem(v, key)


def kirchner_fitter(sample, option=1, start=1, selectivity=200, window=3, minLen=5, ax=None):

    dateList = []
    blist = []
    alist = []
    d = pd.DataFrame({'q': sample})
    dates = d.index
    selector = (d.q.max()-d.q.min())/selectivity
    [maxtab, mintab]=peakdet(d.q, selector)
    d['peaks']=-1
    d.ix[maxtab[:,0].astype(int),'peaks']=maxtab[:,1]
    d['smooth']= d.q.rolling(window).mean(); d['smooth'][0:2] = d.q[0:2]
    d['Dunsmooth']= d.q.diff().shift(-1)
    d['DDsmooth']=d['smooth'].diff().shift(-1).diff().shift(-1)
    d['DDunsmooth'] = d.q.diff().shift(-1).diff().shift(-1)
    d = d[:-2]

    #boolean vector for recession periods
    if option==0:
        d['choose']=d['Dunsmooth']<=0
    else:
        d['choose']=(d['Dunsmooth']<0) & ((d['DDsmooth']>=0)|(d['DDunsmooth']>=0))

    datesMax = d.ix[d['peaks']>0].index
    def func(t, q0, a, b):
            return ((-1+b)*(q0**(1-b)/(b-1)+a*t))**(1/(1-b))
    for i in np.arange(len(datesMax)-1):
        recStart = datesMax[i]; peak1 = datesMax[i]+start; peak2 = datesMax[i+1]
        recEnd = d[peak1:peak2][d[peak1:peak2]['choose']==False].index[0]
        if (len(d[recStart:recEnd])<minLen) | (np.any(d.q[recStart:recEnd]<0)):
            continue
        t = np.arange(len(d.q[recStart:recEnd]))
        q0_data = d.q.loc[recStart]
        try:
            popt, cov = curve_fit(func,t,d.q[recStart:(recEnd)],[q0_data, .1, 1.5])
            if (popt[2]>0)&(popt[2]<10):
                alist.append(popt[1])
                blist.append(popt[2])

        except RuntimeError:
            print('Error encountered in fitting')

        dateList.append(dates[i])

    # Perform Kirchner fitting
    # if fittingType == 'KirchnerBins':
    recessions = d.loc[d.choose]
    binBoundaries = KirchnerBinning(d.loc[d.choose])
    qs = [np.mean(recessions.q[binBoundaries[i]:binBoundaries[i+1]]) for i in range(len(binBoundaries)-1)]
    dqs =   np.array([np.mean(recessions.Dunsmooth[binBoundaries[i]:binBoundaries[i+1]]) for i in range(len(binBoundaries)-1)])
    sigmas = np.array([np.std(recessions.Dunsmooth[binBoundaries[i]:binBoundaries[i+1]])/np.sqrt(binBoundaries[i+1]-binBoundaries[i]) for i in range(len(binBoundaries)-1)])
    sigmas = np.ones(np.shape(sigmas))
    p = np.polyfit(x=np.log(qs), y=np.log(-dqs), deg=2, w=1/(sigmas+1e-12))
    p_powerlaw = np.polyfit(x=np.log(qs), y=np.log(-dqs), deg=1, w=1/(sigmas+1e-12))
    bcurr = p_powerlaw[0]
    acurr = np.exp(p_powerlaw[1])
    tau = d.q.mean()**(1-bcurr) / acurr
    p[1] = p[1]-1

    if ax != None:
        ax.plot(np.log(qs), np.log(-dqs), c='g', alpha=0.5, lw=0, marker='o', label='log(-dq/dt)')
        ax.plot(np.log(qs), p_powerlaw[1] + p_powerlaw[0]*np.log(qs), label='power law')
        ax.plot(np.sort(np.log(qs)), p[2] + (p[1]+1)*np.sort(np.log(qs)) + p[0]*np.sort(np.log(qs))**2, label='kirchner quadratic', c='k')
        ax.legend(frameon=True, fancybox=True, loc=0)
        ax.set_title('b = %g, tau = %g' % (bcurr, tau))
        ax.set_ylabel('log(-dq/dt)')
        ax.set_xlabel('log(q)')

    # elif fittingType == 'KirchnerNonlinear':
    #     ## use the linear coefficients as the init cond for the nonlinear solver
    #     p = np.polyfit(np.log(recessions.q),np.log(-recessions.dq/recessions.q),2)
    #     p, cov = curve_fit(lambda x, a, b, c: KirchnerG(x,a,b,c), recessions.q, -recessions.dq/recessions.q, p0=p)
    # else:
    #     p = np.polyfit(np.log(recessions.q),np.log(-recessions.dq/recessions.q),2)

    return acurr, bcurr, p, dateList, alist, blist


# function to produce function that calculates prob density for any value of x
def pdfq_calc(b, nu, mu=1, qc=0, qm=mp.inf):
    k = 1./nu
    if b == 1.:
        def pdfq(x):
            return ((k**k)/ss.gamma(k)) * (mu)**(-k) * np.exp(-k*x/mu) * (x)**(k-1.)
        return pdfq

    elif b == 2.:
        def pdfq(x):
            return ((k**k)/ss.gamma(k)) * (mu)**(k+1.) * np.exp(-k*mu/x) * (x)**(-k-2.)
        return pdfq

    else:
        with mp.workdps(50):
            b = mp.mpf(b)
            mu = mp.mpf(mu)
            k = mp.mpf(k)
            two = mp.mpf(2.0)
            one = mp.mpf(1.0)
            c1 = (((two*mp.pi*(two-b)*(b-one))/((mu**two) * k))**(one/two))
            c2 = (((k/(two-b))**(k/(two-b)))/mp.gamma(k/(two-b)))
            c3 = (((k/(b-one))**(k/(b-one)))/mp.gamma(k/(b-one)))
            C = c1 * c2 * c3
            f = lambda x: C * (x/mu)**(-b) * mp.exp(-k*((x/mu)**(two-b))/(two-b)) * mp.exp(k*((x/mu)**(one-b))/(one-b))
            Cn = C/mp.quad(f, [qc, qm])

        def pdfq(x):
            out = Cn * (x/mu)**(-b) * mp.exp(-k*((x/mu)**(two-b))/(two-b)) * mp.exp(k*((x/mu)**(1-b))/(1-b))
            if (qm > 0) & (x > qm):
                out = 0
            return mp.sqrt(out.real**2 + out.imag**2)

        return np.frompyfunc(lambda *a: float(pdfq(*a)), 1, 1)


# function that computes Cn for any value of b and v
def Cn_calc(b, nu, mu=1, Qm8=-1):
    if b == 1.:
        Cn = 1.

    elif b == 2.:
        Cn = 1.

    else:
        with mp.workdps(50):
            b = mp.mpf(b)
            mu = mp.mpf(mu)
            k = mp.mpf(1. / nu)
            two = mp.mpf(2.0)
            one = mp.mpf(1.0)
            z1 = k/(b-one)
            z2 = k/(two-b)
            c1 = (((two*mp.pi*(two-b)*(b-one))/((mu**two) * k))**(one/two))
            c2 = (((k/(two-b))**(k/(two-b)))/mp.gamma(k/(two-b)))
            c3 = (((k/(b-one))**(k/(b-one)))/mp.gamma(k/(b-one)))
            C = c1 * c2 * c3
            f = lambda x: C * (x/mu)**(-b) * mp.exp(-k*((x/mu)**(two-b))/(two-b)) * mp.exp(k*((x/mu)**(one-b))/(one-b))
            Cn = C/mp.quad(f, [0, mp.inf])

    return float(mp.sqrt(Cn.real**2 + Cn.imag**2))


def f(x, b, nu, Cn, mu=1):
    k = 1. / nu
    if b == 1.:
        return k*np.log(k) - ss.loggamma(k).real - k*np.log(mu) + (k-1.)*np.log(x) - k*x/mu

    elif b == 2.:
        return k*np.log(k) + (k+1.)*np.log(mu) - ss.loggamma(k).real - (k+2.)*np.log(x) - k*mu/x

    else:
        return np.log(Cn) - b*np.log(x) + (b-1.)*np.log(mu) + k*( ( x**(1.-b) ) / ((1.-b)*(mu**(1.-b))) - ( x**(2.-b) ) / ((2.-b)*(mu**(2.-b))) )


def fprima(x, b, nu, Cn, mu=1):
    k = 1. / nu
    if b == 1.:
        return (k-1.)/x - k/mu

    elif b == 2.:
        return k*mu/(x**2) - (k+2.)/x

    else:
        return -b/x + k*((mu/x)**b)*(1./mu - x/(mu*mu))### Parameter estimation


b = 1.5
mu = 1.
nu = 1.
def bot_pdf(x, b=b, nu=nu, mu=mu):
    k = np.exp(nu)
    qc = x.min()
    qm = x.max()
    if (b == 1.):
        return ((k**k)/ss.gamma(k)) * np.exp(-k*x/mu) * (x/mu)**(k-1.)

    elif (b == 2.):
        return ((k**k)/ss.gamma(k)) * np.exp(-k*mu/x) * (x/mu)**(-k-2.)

    else:
        ### Slow Accurate Method ###
        with mp.workdps(35):
            b = mp.mpf(b)
            mu = mp.mpf(1)
            k = mp.mpf(k)
            two = mp.mpf(2.0)
            one = mp.mpf(1.0)
            c1 = (((two*mp.pi*(two-b)*(b-one))/((mu**two) * k))**(one/two))
            c2 = (((k/(two-b))**(k/(two-b)))/mp.gamma(k/(two-b)))
            c3 = (((k/(b-one))**(k/(b-one)))/mp.gamma(k/(b-one)))
            C = c1 * c2 * c3
            f = lambda x: C * (x/mu)**(-b) * mp.exp(-k*((x/mu)**(two-b))/(two-b)) * mp.exp(k*((x/mu)**(one-b))/(one-b))
            Cn = C/mp.quad(f, [qc, qm])
            # Cn = C/mp.quad(f, [0, mp.inf])


            def pdfq(x_in):
                numb = Cn * (x_in/mu)**(-b) * mp.exp(-k*((x_in/mu)**(two-b))/(two-b)) * mp.exp(k*((x_in/mu)**(one-b))/(one-b))
                return mp.sqrt(numb.real**2 + numb.imag**2)

            out = np.frompyfunc(lambda *a: float(pdfq(*a)), 1, 1)

            return np.asarray(out(x) / mu).astype(float)


class BotterDischargeDistribution(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = np.zeros_like(endog)

        super(BotterDischargeDistribution, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        b = params[0]
        nu = params[1]

        return -np.log(bot_pdf(self.endog, b=b, nu=nu))

    def fit(self, start_params=None, maxiter=200, maxfun=200, **kwds):
        if start_params is None:
            nu_start = np.log(.01)
            b_start = 1.

            start_params = np.array([b_start, nu_start])

        return super(BotterDischargeDistribution, self).fit(start_params=start_params,
                                                    maxiter=maxiter, maxfun=maxfun, **kwds)


def make_fixed_b_likelihood(b_in):
    mu = 1.
    nu = 1.
    def bot_pdf_fixed_b(x, b=b, nu=nu, mu=mu):
        k = np.exp(nu)
        qc = x.min()
        qm = x.max()
        if (b == 1.):
            return ((k**k)/ss.gamma(k)) * np.exp(-k*x/mu) * (x/mu)**(k-1.)

        elif (b == 2.):
            return ((k**k)/ss.gamma(k)) * np.exp(-k*mu/x) * (x/mu)**(-k-2.)

        else:
            ### Slow Accurate Method ###
            with mp.workdps(35):
                b = mp.mpf(b)
                mu = mp.mpf(1)
                k = mp.mpf(k)
                two = mp.mpf(2.0)
                one = mp.mpf(1.0)
                c1 = (((two*mp.pi*(two-b)*(b-one))/((mu**two) * k))**(one/two))
                c2 = (((k/(two-b))**(k/(two-b)))/mp.gamma(k/(two-b)))
                c3 = (((k/(b-one))**(k/(b-one)))/mp.gamma(k/(b-one)))
                C = c1 * c2 * c3
                f = lambda x: C * (x/mu)**(-b) * mp.exp(-k*((x/mu)**(two-b))/(two-b)) * mp.exp(k*((x/mu)**(one-b))/(one-b))
                Cn = C/mp.quad(f, [qc, qm])
                # Cn = C/mp.quad(f, [0, mp.inf])

                def pdfq(x_in):
                    numb = Cn * (x_in/mu)**(-b) * mp.exp(-k*((x_in/mu)**(two-b))/(two-b)) * mp.exp(k*((x_in/mu)**(one-b))/(one-b))
                    return mp.sqrt(numb.real**2 + numb.imag**2)

                out = np.frompyfunc(lambda *a: float(pdfq(*a)), 1, 1)

                return np.asarray(out(x) / mu).astype(float)
    return bot_pdf_fixed_b


def pdf_fitter(sample, b=None, b_fit=True, nu=None, qmin=0, qmax=np.inf, ax=None, b_ls='-', b_marker='o', b_name='b'):

    if b_fit == True:
        if b != None:
            b_fix = True
        else:
            b_fix = False

        # crop sample below min and above max
        xbar = sample[(sample>=qmin) & (sample<=qmax)].mean()

        if b_fix == False:
            model = BotterDischargeDistribution(sample/xbar)
            results = model.fit()
            b = results.params[0]
            try:
                b_bse = results.bse[0]
            except:
                b_bse = np.nan

            nu = 1./np.exp(results.params[1])
            try:
                nu_bse = results.bse[1]
            except:
                nu_bse = np.nan

        elif b_fix == True:
            bot_pdf_fixed_b = make_fixed_b_likelihood(b)

            class BotterDischargeDistributionFixedB(GenericLikelihoodModel):
                def __init__(self, endog, exog=None, **kwds):
                    if exog is None:
                        exog = np.zeros_like(endog)

                    super(BotterDischargeDistributionFixedB, self).__init__(endog, exog, **kwds)

                def nloglikeobs(self, params):
                    nu = params[0]

                    return -np.log(bot_pdf_fixed_b(self.endog, nu=nu))

                def fit(self, start_params=None, maxiter=200, maxfun=200, **kwds):
                    if start_params is None:
                        nu_start = np.log(.01)

                        start_params = np.array([nu_start])

                    return super(BotterDischargeDistributionFixedB, self).fit(start_params=start_params,
                                                                maxiter=maxiter, maxfun=maxfun, **kwds)

            model = BotterDischargeDistributionFixedB(sample/xbar)
            results = model.fit()
            nu = 1./np.exp(results.params[0])
            try:
                nu_bse = results.bse[0]
            except:
                nu_bse = np.nan

            b_bse = np.nan


    # calculate r2 for fit distribution
    Qe = np.log(1. - np.linspace(0.,1.,sample.size))[:-1]
    pdfq_r2 = pdfq_calc(b, nu, 1)
    qx = np.sort(sample/sample.mean())
    Qt = np.log(1. - np.asarray(sint.cumtrapz(pdfq_r2(qx), qx, initial=0).astype(float)))[:-1]

    b_r2 = 1. - (np.nansum((Qt - Qe)**2) / np.nansum((Qe - Qe.mean())**2))

    # calculate theoretical mean, actual covariance and theoretical covariance
    mu_e = np.nan

    mu_t = np.nan #sample.mean()/C_mean(b, nu, 1, sample.max())

    CV_e = sample.var() / sample.mean()

    CV_t = np.nan

    if ax != None:
        (ax1, ax2) = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=ax)

        sample = sample/xbar
        pdf, x = pdf_ccdf(sample)
        ax1.loglog(x, pdf, 'ko')

        pdfq = pdfq_calc(b, nu, 1)
        x = np.logspace(np.log10(x.min()/10),np.log10(x.max()*10),1000)
        ax1.plot(x, pdfq(x), 'b', ls=b_ls, label='%s: nu = %g, b = %g, r2 = %0.4f' % (b_name, nu, b, b_r2))
        ax1.set_xlabel('Normalized daily discharge magnitude')
        ax1.set_ylabel('Probability density')
        # ax.set_title('b = %g, nu = %g' % (b, nu))
        ax1.legend(frameon=True, fancybox=True, loc=0)
        ax1.set_ylim(1e-7, pdf.max()*10)
        ax1.set_xlim(x.min(), x.max())

        ax2.loglog(np.sort(sample[:-1]), Qt/Qe, color='r', marker=b_marker, alpha=0.5)
        x = np.linspace(sample.min(), np.sort(sample)[-1], 10)
        ax2.plot(x, np.ones(10), 'k', label='r2 = %g' % r2s[key])
        ax2.legend(loc=0)
        ax2.set_xlabel('Theoretical Quantiles')
        ax2.set_ylabel('Observed Quantiles')
        ax2.set_ylim([0.1, 10])

    return b, nu, b_bse, nu_bse, mu_t, b_r2


def streamflow_analyzer(sample, b_in=[], tw=365., plot=False, return_data=False):

    if plot == True:

        fig, axes = plt.subplots(3,2, figsize=(10,12))

        A_hat, B_hat, P_hat, dateList, alist, blist = kirchner_fitter(sample, ax=axes[0,0])

        if len(b_in) == 0:
            B_pdf_hat, nu_hat, B_pdf_bse, nu_bse, mu_t, r2b = pdf_fitter(sample, ax=axes[0,1], b_ls='-', b_marker='o', b_name='Botter')

            B, nu_k, B2, nu_k_bse, mu_kt, r2b_k = pdf_fitter(sample, b=B_hat, ax=axes[0,1], b_ls='--', b_marker='o', b_name='Kirchner')

        elif len(b_in) == 1:
            B_pdf_hat, nu_hat, B_pdf_bse, nu_bse, mu_t, r2b = pdf_fitter(sample, b=b_in[0], fit_b=False, ax=axes[0,1])

            B, nu_k, B2, nu_k_bse, mu_kt, r2b_k = pdf_fitter(sample, b=B_hat, ax=axes[0,1])

        elif len(b_in) == 2:
            B_pdf_hat, nu_hat, B_pdf_bse, nu_bse, mu_t, r2b = pdf_fitter(sample, b=b_in[0], fit_b=False, ax=axes[0,1])

            B, nu_k, B2, nu_k_bse, mu_kt, r2b_k = pdf_fitter(sample, b=b_in[0], fit_b=False, ax=axes[0,1])

        jump_df, arrivals, storage_mags, r2_arr, r2_storage = assess_IRA(sample, A_hat, B_hat, tw=tw, axs=(axes.flatten())[2:])

    else:

        A_hat, B_hat, P_hat, dateList, alist, blist = kirchner_fitter(sample)
        print(alist, blist)

        if len(b_in) == 0:
            B_pdf_hat, nu_hat, B_pdf_bse, nu_bse, mu_t, r2b = pdf_fitter(sample)

            B, nu_k, B2, nu_k_bse, mu_kt, r2b_k = pdf_fitter(sample, b=B_hat)

        elif len(b_in) == 1:
            B_pdf_hat, nu_hat, B_pdf_bse, nu_bse, mu_t, r2b = pdf_fitter(sample, b=b_in[0], fit_b=False)

            B, nu_k, B2, nu_k_bse, mu_kt, r2b_k = pdf_fitter(sample, b=B_hat)

        elif len(b_in) == 2:
            B_pdf_hat, nu_hat, B_pdf_bse, nu_bse, mu_t, r2b = pdf_fitter(sample, b=b_in[0], fit_b=False)

            B, nu_k, B2, nu_k_bse, mu_kt, r2b_k = pdf_fitter(sample, b=b_in[0], fit_b=False)

        jump_df, arrivals, storage_mags, r2_arr, r2_storage = assess_IRA(sample, A_hat, B_hat, tw=tw)

    if return_data == True:

        return {
                'a_k': A_hat,
                'b_k': B_hat,
                'r2_b_k': r2b_k,
                'P_k': [P_hat],
                'nu_k': nu_k,
                'nu_k_bse': nu_k_bse,

                'a_ind': [alist],
                'b_ind': [blist],

                'b_pdf': B_pdf_hat,
                'b_pdf_bse': B_pdf_bse,
                'r2_b_pdf': r2b,
                'nu_pdf': nu_hat,
                'nu_bse': nu_bse,

                'mu_t': mu_t,
                # 'mu_e': mu_e,
                # 'CV_t': CV_t,
                # 'CV_e': CV_e,

                'tau_s': arrivals.mean(),
                'alpha_h': storage_mags.mean(),
                'r2_tau_s': r2_arr,
                'r2_alpha_h': r2_storage
                }
