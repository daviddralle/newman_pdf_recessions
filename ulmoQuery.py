import pandas as pd
import ulmo

#functions written by Paul Inkenbrandt
#need ulmo package, 'conda install -c IOOS ulmo'
def getUSGSdata(siteno, startdate, enddate):
    sitename = {};
    sitename = ulmo.usgs.nwis.get_site_data(siteno, service="iv", start =startdate, end = enddate);
    sitename = pd.DataFrame(sitename)
    sitename = pd.DataFrame(sitename['00060:00000']['values']);
    sitename['dates'] = pd.to_datetime(pd.Series(sitename['datetime']))- pd.Timedelta('8 hours');
    sitename.set_index(['dates'],inplace=True);
    df = sitename.value.astype(float)
    df = df.resample('15min', how='mean')
    return df;