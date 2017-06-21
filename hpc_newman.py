from math import sqrt
import matplotlib
matplotlib.use('Agg')
from joblib import Parallel, delayed
import numpy as np, matplotlib.pylab as plt, seaborn as sns, mpmath as mp, scipy.special as ss, sys
sys.path.append('os.getcwd()')
import geopandas as gp, pandas as pd, urllib2
from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('text', usetex=True)
import os
import datetime
import pickle
from a_b_functions import getFlow as getFlow
import a_b_functions
from streamflow_analyzer import streamflow_analyzer
import warnings
warnings.filterwarnings("ignore")
site_data = gp.read_file('./USGS_Streamgages-NHD_Locations.shp')
seasons = ['spring', 'winter', 'wet', 'annual']
snowfrac = pickle.load( open('./snow_fraction.p','rb') )
# seasons = ['winter', 'wet']


def run_newman(fh):
    site = fh.split('/')[-1][:8]
    try: 
        if snowfrac.loc[site] > 0.01: 
            return
    except:
        return
    site_dict = {}
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

        sample = np.array(d.tolist())
        site_season = streamflow_analyzer(sample, tw = tw, return_data = True)
        site_dict[(site, seasons[ind])] = site_season

    savestr = './results_analyzer/' + site + '_results.p'
    pickle.dump(site_dict, open(savestr, 'wb'))


def main():
    flow_files = a_b_functions.getFlowFileList()
    # flow_files = flow_files[:337]  # david's list
    # flow_files = flow_files[337:]  # eric's list

    # lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]
    # flow_files_lists = lol(flow_files, 46)
    # for flow_files_list in flow_files_lists:
    #     Parallel(n_jobs=23)(delayed(run_newman) (flow_files_list[i]) for i in range(len(flow_files_list)))
    # Parallel(n_jobs=4)(delayed(run_newman) (flow_files[i]) for i in range(len(flow_files)))
    Parallel(n_jobs=4)(delayed(run_newman) (flow_files[i]) for i in range(len(flow_files)) )


    # Parallel(n_jobs=3)(delayed(run_newman) (flow_files[i]) for i in [200, 201, 202])
    # pickle.dump(res, open('./results.p', 'wb'))

if __name__ == '__main__':
	main()