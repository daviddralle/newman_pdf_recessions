from math import sqrt
import matplotlib
matplotlib.use('Agg')
from joblib import Parallel, delayed
import numpy as np, matplotlib.pylab as plt, seaborn as sns, mpmath as mp, scipy.special as ss, sys
sys.path.append('os.getcwd()')
import geopandas as gp, pandas as pd, urllib2
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
import os
import datetime
import pickle
from pdf_ccdf import pdf_ccdf
from pdf_functions import pdf_fitter as pdf_fitter
from a_b_functions import kirchner_fitter as kirchner_fitter, KirchnerBinning
from IRA_functions import assess_IRA as assess_IRA
from a_b_functions import getFlow as getFlow
import a_b_functions
import warnings
warnings.filterwarnings("ignore")
site_data = gp.read_file('./USGS_Streamgages-NHD_Locations.shp')
seasons = ['spring', 'summer', 'fall', 'winter', 'wet', 'annual']
# seasons = ['fall', 'wet']


def run_newman(fh):
    A = {}
    B = {}
    P = {}
    B_pdf = {}
    nu_pdf = {}
    datedict = {}
    MU_E = {}
    LAM_H = {}
    ALPHA_H = {}
    MU_KT = {}
    MU_T = {}
    NU_K = {}
    NU_K_BSE = {}
    R2_ARR = {}
    R2_STORAGE = {}
    R2B = {}
    R2B_K = {}
    A_EVENT = {}
    B_EVENT = {}

    site = fh.split('/')[-1][:8]
    # weather = pickle.load( open('./daymet_newman/'+site+'_daymet.p', 'rb') )
    df = pd.read_csv(fh, delim_whitespace=True, header=-1)
    df.columns = ['gagenum', 'Year', 'Month', 'Day', 'q', 'e']
    df['date'] = df[['Year', 'Month', 'Day']].apply(lambda s : datetime.datetime(*s),axis = 1)
    df = df[['q', 'date']]
    df.set_index('date', inplace=True)
    df['date'] = df.index
    df.q += 1e-12 # having flow exactly equal to zero can cause problems with logs
    df.q *= 2.447e9 #cm^3/day
    area = float(site_data['DA_SQ_MILE'].loc[site_data.SITE_NO==site])*2.58998811e10 #cm^2
    df.q = df.q/area # cm/day
    df = df.loc[df.q>0]
    
    for ind in range(len(seasons)):
        if seasons[ind]=='winter':
            tw = len(pd.date_range('12-2015', '3-2016'))
            d = df.q.loc[(df.index.month>=12)|(df.index.month<=2)]
        elif seasons[ind]=='spring': 
            tw = len(pd.date_range('3-2016', '6-2016'))
            d = df.q.loc[(df.index.month>=3)&(df.index.month<=5)]
        elif seasons[ind]=='summer':
            tw = len(pd.date_range('6-2016', '9-2016'))
            d = df.q.loc[(df.index.month>=6)&(df.index.month<=8)]
        elif seasons[ind]=='fall':
            tw = len(pd.date_range('9-2016', '12-2016'))
            d = df.q.loc[(df.index.month>=9)&(df.index.month<=11)]
        elif seasons[ind]=='wet':
            tw = len(pd.date_range('11-2016', '4-2017'))
            d = df.q.loc[(df.index.month>=11)|(df.index.month<=4)]
        elif seasons[ind]=='annual':
            tw = 365
            d = df.q.loc[(df.index.month>=1)|(df.index.month<=12)]

        
        fig, axes = plt.subplots(3,2, figsize=(10,12))
        print('made it past plotting')
        A_hat, B_hat, P_hat, dateList, alist, blist = kirchner_fitter(d, ax=axes[0,0])
        print('finished ab fits')
        A[(site, seasons[ind])] = A_hat
        A_EVENT[(site, seasons[ind])] = alist
        B_EVENT[(site, seasons[ind])] = blist
        B[(site, seasons[ind])] = B_hat
        P[(site, seasons[ind])] = P_hat
        datedict[(site, seasons[ind])] = dateList
         

        # pdf_fitter needs a numnpy array of all daily discharge magnitudes in timeseries. If you pass it an axis, it will plot pdf of sample against best fit. 
        sample = pd.DataFrame({'q':d.tolist()}).q
        MU_E[(site, seasons[ind])] = sample.mean()
        try: 
            B_pdf_hat, nu_hat, B_pdf_bse, nu_bse, mu_t, r2b = pdf_fitter(sample, ax=axes[0,1])
            B_whatever, nu_k, B_junk, nu_k_bse, mu_kt, r2b_k = pdf_fitter(sample, ax=axes[0,1], b=B_hat)
        except: 
            B_pdf_hat, nu_hat, B_pdf_bse, nu_bse, mu_t, r2b = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            B_whatever, nu_k, B_junk, nu_k_bse, mu_kt, r2b_k = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            
        MU_KT[(site, seasons[ind])] = mu_kt
        MU_T[(site, seasons[ind])] = mu_t
        B_pdf[(site, seasons[ind])] = B_pdf_hat
        nu_pdf[(site, seasons[ind])] = nu_hat
        jump_df, arrivals, storage_mags, r2_arr, r2_storage = assess_IRA(d, A_hat, B_hat, tw=tw, axs=(axes.flatten())[2:])
        R2_ARR[(site, seasons[ind])] = r2_arr
        R2_STORAGE[(site, seasons[ind])] = r2_storage
        LAM_H[(site, seasons[ind])] = np.mean(arrivals)
        ALPHA_H[(site, seasons[ind])] = np.mean(storage_mags)
        NU_K[(site, seasons[ind])] = nu_k
        NU_K_BSE[(site, seasons[ind])] = nu_k_bse
        savestr = site + '_' + seasons[ind] + '.png'
        fig.savefig('./plots/'+savestr)

    return (A, B, datedict, B_pdf, nu_pdf, MU_E, LAM_H, ALPHA_H, NU_K, NU_K_BSE, MU_KT, MU_T, R2B, R2B_K, A_EVENT, B_EVENT)


def main():
    flow_files = a_b_functions.getFlowFileList()
    res = Parallel(n_jobs=23)(delayed(run_newman) (flow_files[i]) for i in range(len(flow_files)))
    # res = Parallel(n_jobs=4)(delayed(run_newman) (flow_files[i]) for i in [1])
    pickle.dump(res, open('./results.p', 'wb'))

if __name__ == '__main__':
	main()