import matplotlib.pyplot as plt
import numpy as np
import datetime
import pandas as pd
import os

def plot():
    NB = pd.read_csv(str(os.environ["HOURLY_SECRET"]), sep=',')
    #NB = pd.read_csv(str(os.environ["DAILY_SECRET"]), sep=',')
    NB['date'] = pd.to_datetime(NB['time'])
    NB = (NB.loc[np.where(NB['date'] >= pd.Timestamp('2021-07-01 00:00:00'))[0][0]:]).reset_index()

    # rotate old data by 180 degrees
    if NB['wdir_u'][0] < 200:
        NB.loc[:np.where(NB['date'] <= pd.Timestamp('2023-09-12 17:00:00'))[0][-1],'wdir_u'] += 180
    NB.loc[NB['wdir_u']>360,'wdir_u'] -= 360

    # field notes
    rotation_dates = [pd.Timestamp('2021-09-02 12:00:00'), pd.Timestamp('2023-09-12 17:00:00')]
    rotation_offset = [np.nan, 60]

    # estimate MSLP
    NB_gps_alt = NB[NB['gps_alt'].notnull()]
    h = NB_gps_alt['gps_alt'].mean() # potential error of a few hPa based on changing altitude
    NB['mslp'] = NB['p_u']*(1-0.0065*h/(NB['t_u']+273.15+0.0065*h))**(-5.257)

    ### plot data

    start = np.where(NB['date'] >= pd.Timestamp('2023-10-10 00:00:00'))[0][0]
    end = -1#np.where(NB['date'] >= pd.Timestamp('2022-10-11 00:00:00'))[0][0]

    c = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']

    fig,(ax1) = plt.subplots(1,figsize=(20,8),dpi=50)
    plt.rcParams.update({'font.size': 22})

    ax1.plot(NB['date'],NB['p_u'],'-',c=c[0])
    ax1.set_title('last datetime in csv-file: %s\ncsv-file last checked: %s' % (NB['date'].max(),datetime.datetime.now()))
    #ax.plot(NB['date'],NB['p_u'],marker='.',c=c[0])
    ax1b = ax1.twinx()
    ax1b.plot(NB['date'],NB['mslp'],c=c[1])
    ax1.set_ylabel('pressure (hPa)', c=c[0])
    ax1b.set_ylabel('mean sea level pressure (hPa)', rotation=270, labelpad=25, c=c[1])
    ax1.tick_params(axis='y', colors=c[0])
    ax1b.tick_params(axis='y', colors=c[1])
    ax1.set_xlim(xmin=NB['date'].values[start],xmax=NB['date'].values[end])
    #plt.xticks(rotation=45)#, ha='right')
    fig.autofmt_xdate(rotation=45)

    plt.savefig('fig.png')

if __name__ == "__main__":
    plot()
