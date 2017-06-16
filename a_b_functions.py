# import needed packages
import numpy as np, matplotlib.pylab as plt, pandas as pd, urllib2
from peakdetect import peakdet as peakdet
from scipy.optimize import curve_fit
import os
import glob

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


def getFlow(site):
    '''
        Input: USGS gage number

        Output: pandas data frame of volumetric streamflow (over period of record at the gage) indexed by datetime
    '''

    try:
        f = open('flow_data/'+ site + '.txt','r')
    except:
        print('downloading data')
        site = str(site)

        url = 'https://waterdata.usgs.gov/nwis/dv?cb_00060=on&format=rdb&site_no=' + site + '&referred_module=sw&period=&begin_date=1950-01-01&end_date=2017-05-01'
        response = urllib2.urlopen(url)
        content = response.read()

        f = open('flow_data/'+ site+'.txt','w')
        f.write(content)
        f.close()

    count = 0
    for line in open('flow_data/'+ site+'.txt','r').readlines():
        if line[0] == '#':
            count += 1
        else:
            break

    df = pd.read_csv('flow_data/'+ site+'.txt', header=count, delimiter='\t')

    q_col = next(col for col in df.columns if col.endswith('00060_00003'))

    df.rename(columns={q_col: 'q', 'datetime':'date'}, inplace=True)
    df = df.iloc[1:,:]
    df = df.dropna()

    df.date = pd.to_datetime(df.date)
    df.q = pd.to_numeric(df.q)

    df.index = pd.to_datetime(df.date)

    return df


def _finditem(obj, key):
    if key in obj: return obj[key]
    for k, v in obj.items():
        if isinstance(v,dict):
            return _finditem(v, key)


def kirchner_fitter(d, option=1, start=1, selectivity=200, window=3, minLen=5, ax=None):

    dateList = []
    blist = []
    alist = []
    dates = d.index
    d = pd.DataFrame({'q':d.tolist()})
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


def getFlowFileList():
    folders = os.listdir('./usgs_streamflow')
    fh = []
    fh = [folders[i] for i in range(len(folders)) if len(folders[i])==2]

    flow_files = []
    for f in fh:
        try:

            flow_files.append(glob.glob('./usgs_streamflow/' + f + '/*.txt'))
        except RuntimeError:
            print 'Cannot find streamflow file.'

    return [item for sublist in flow_files for item in sublist]
