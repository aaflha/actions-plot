import matplotlib.pyplot as plt
import numpy as np
import datetime
import pandas as pd
import os
import matplotlib.path as mpath


def get_hull():
    verts1 = np.array([[0,-128],[70,-128],[128,-70],[128,0],
                      [128,32.5],[115.8,61.5],[96,84.6],[96,288],
                      [96,341],[53,384],[0,384]])
    verts2 = verts1[:-1,:] * np.array([-1,1])
    codes1 = [1,4,4,4,4,4,4,2,4,4,4]
    verts3 = np.array([[0,-80],[44,-80],[80,-44],[80,0],
                      [80,34.3],[60.7,52],[48,66.5],[48,288],
                      [48,314],[26.5,336],[0,336]])
    verts4 = verts3[:-1,:] * np.array([-1,1])
    verts = np.concatenate((verts1, verts2[::-1], verts4, verts3[::-1]))
    codes = codes1 + codes1[::-1][:-1]
    return mpath.Path(verts/256., codes+codes)

def get_mercury(s=1):
    a = 0; b = 64; c = 35
    d = 320 - b
    e = (1-s)*d
    verts1 = np.array([[a,-b],[c,-b],[b,-c],[b,a],[b,c],[c,b],[a,b]])
    verts2 = verts1[:-1,:] * np.array([-1,1])
    verts3 = np.array([[0,0],[32,0],[32,288-e],[32,305-e],
                       [17.5,320-e],[0,320-e]])
    verts4 = verts3[:-1,:] * np.array([-1,1])
    codes = [1] + [4]*12 + [1,2,2,4,4,4,4,4,4,2,2]
    verts = np.concatenate((verts1, verts2[::-1], verts3, verts4[::-1]))
    return mpath.Path(verts/256., codes)

def scatter(self, x,y, temp=1, tempnorm=None, ax=None, **kwargs):
    self.ax = ax or plt.gca()
    temp = np.atleast_1d(temp)
    ec = kwargs.pop("edgecolor", "black")
    kwargs.update(linewidth=0)
    self.inner = self.ax.scatter(x,y, **kwargs)
    kwargs.update(c=None, facecolor=ec, edgecolor=None, color=None)
    self.outer = self.ax.scatter(x,y, **kwargs)
    self.outer.set_paths([self.get_hull()])
    if not tempnorm:
        mi, ma = np.nanmin(temp), np.nanmax(temp)
        if mi == ma:
            mi=0
        tempnorm = plt.Normalize(mi,ma)
    ipaths = [self.get_mercury(tempnorm(t)) for t in temp]
    self.inner.set_paths(ipaths)

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

    fig, ax = plt.subplots(figsize=(10,5))
    n = 42
    x = NB['date'].values[-49:-1]
    y = NB['t_u'].values[-49:-1]
    ymin = round(np.min(y))-1
    ymax = round(np.max(y))+3
    sind = 3
    ind = 6
    
    ax.plot(x,y, color="darkgrey", lw=2.5)
    
    p = TemperaturePlot()
    p.scatter(x[sind::ind],y[sind::ind]+2, s=300, temp=y[sind::ind], c=y[sind::ind], edgecolor="k", cmap="RdYlBu_r")
    
    ax.set_ylim(ymin,ymax)
    ax.set_xticks([NB['date'].values[-49],NB['date'].values[-37],NB['date'].values[-25],NB['date'].values[-13],NB['date'].values[-1]])
    ax.set_xticklabels(['48','36','24', '12', '0'])
    ax.set_xlabel('Hours ago')
    ax.set_ylabel('Temperature (\u00b0C)')

    plt.savefig('fig.png')

if __name__ == "__main__":
    plot()
