# import needed packages
import numpy as np, matplotlib.pylab as plt, pandas as pd, mpmath as mp, scipy.special as ss
from scipy.special import gammainc as lower_gamma
from scipy.special import gammaincc as upper_gamma
import scipy.integrate as sint
from ars import ARS
from pdf_ccdf import pdf_ccdf
from statsmodels.base.model import GenericLikelihoodModel


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
        with mp.workdps(20):
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


def C_mean(b, nu, mu, qm):
    k = 1. / nu
    if b < 1.001:
        tm = k*qm/mu
        return (lower_gamma(k+1., tm) / lower_gamma(k, tm))

    elif b > 1.999:
        tm = k*mu/qm
        return (upper_gamma(k, tm)) / upper_gamma(k+1, tm)

    else:
        with mp.workdps(50):
            b = mp.mpf(b)
            mu = mp.mpf(mu)
            k = mp.mpf(1. / nu)
            two = mp.mpf(2.0)
            one = mp.mpf(1.0)
            c1 = (((two*mp.pi*(two-b)*(b-one))/((mu**two) * k))**(one/two))
            c2 = (((k/(two-b))**(k/(two-b)))/mp.gamma(k/(two-b)))
            c3 = (((k/(b-one))**(k/(b-one)))/mp.gamma(k/(b-one)))
            C = c1 * c2 * c3
            f = lambda x: C * (x/mu)**(-b) * mp.exp(-k*((x/mu)**(two-b))/(two-b)) * mp.exp(k*((x/mu)**(one-b))/(one-b))
            Cn = one/mp.quad(f, [0, mp.inf]) *  C

            f = lambda x: Cn * (x)**(one-b) * mu**b * mp.exp(-k*((x/mu)**(two-b))/(two-b)) * mp.exp(k*((x/mu)**(one-b))/(one-b))

            return float(mp.quad(f, [0, qm]).real / mu)


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
        with mp.workdps(20):
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
            with mp.workdps(20):
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


def pdf_fitter(sample, b=None, qmin=0, qmax=np.inf, ax=None):

    if b != None:
        b_fix = True
    else:
        b_fix = False

    # crop sample below min and above max
    xbar = sample[(sample>=qmin) & (sample<=qmax)].mean()

    if b == None:
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

    else:
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
    sample_r2 = sample/xbar
    Qe = np.percentile(sample_r2, range(1,100))
    q_r2 = np.logspace(np.log10(sample_r2.min()), np.log10(sample_r2.max()), 10000)
    pdfq_r2 = pdfq_calc(b, nu, 1)
    pdf_r2 = pdfq_r2(q_r2)
    X = sint.cumtrapz(pdf_r2,q_r2)
    Qt = np.zeros(99)
    for ii in range(1,100):
        try:
            idx = np.where(X>=ii/100.)[0][0]
            Qt[ii-1] = q_r2[idx]
        except:
            Qt[ii-1] = np.nan

    b_r2 = 1. - (np.nansum((Qt - Qe)**2) / np.nansum((Qe - Qe.mean())**2))

    # calculate theoretical mean
    mu_t = sample.mean()/C_mean(b, nu, 1, sample.max())


    if ax != None:
        pdf, x = pdf_ccdf(sample)
        ax.loglog(x, pdf, 'ro', alpha=0.3)

        pdfq = pdfq_calc(b, nu, xbar)
        x = np.logspace(-5,5,1000)
        if b_fix == True:
            ax.plot(x, pdfq(x), 'k--', label='Kirchner: nu = %g, b = %g, r2 = %0.4f' % (nu, b, b_r2))
        else:
            ax.plot(x, pdfq(x), 'k-', label='Botter: nu = %g, b = %g, r2 = %0.4f' % (nu, b, b_r2))

        ax.set_xlabel('Normalized daily discharge magnitude')
        ax.set_ylabel('Probability density')
        # ax.set_title('b = %g, nu = %g' % (b, nu))
        ax.legend(frameon=True, fancybox=True, loc=0)
        ax.set_ylim(1e-6, 1e2)
        ax.set_xlim(1e-3, 1e3)

    return b, nu, b_bse, nu_bse, mu_t, b_r2
