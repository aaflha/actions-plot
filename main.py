import matplotlib.pyplot as plt
import numpy as np
import datetime
import pandas as pd
import os
import matplotlib.path as mpath

class TemperaturePlot():

    @staticmethod
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

    @staticmethod
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

def get_wind_direction(degrees):
    if degrees < 0 or degrees >= 360:
        raise ValueError("Degrees must be between 0 and 359")
    if (degrees >= 337.5) or (degrees < 22.5):
        return 'north'
    elif 22.5 <= degrees < 67.5:
        return 'northeast'
    elif 67.5 <= degrees < 112.5:
        return 'east'
    elif 112.5 <= degrees < 157.5:
        return 'southeast'
    elif 157.5 <= degrees < 202.5:
        return 'south'
    elif 202.5 <= degrees < 247.5:
        return 'southwest'
    elif 247.5 <= degrees < 292.5:
        return 'west'
    elif 292.5 <= degrees < 337.5:
        return 'northwest'

def plot():
    NB = pd.read_csv(str(os.environ["HOURLY_SECRET"]), sep=',')
    #NB = pd.read_csv(str(os.environ["DAILY_SECRET"]), sep=',')
    NB['date'] = pd.to_datetime(NB['time'])
    NB = (NB.loc[np.where(NB['date'] >= pd.Timestamp('2021-07-01 00:00:00'))[0][0]:]).reset_index()

    # last observations
    
    last_48 = NB.tail(48)
    
    wind_speeds = last_48['wspd_u'].values
    wind_directions = last_48['wdir_u'].values
    #temps = last_48['wdir_u'].values
    
    if pd.isna(wind_speeds[-1]) or pd.isna(wind_directions[-1]):
        last_observation = last_48.iloc[-2]  # Use second last if last is NaN
    else:
        last_observation = last_48.iloc[-1]  # Last observation
    
    last_wind_speed = last_observation['wspd_u']
    last_wind_direction = last_observation['wdir_u']
    last_temperature = last_observation['t_u']
    
    ### plot data
    
    plt.rcParams.update({'font.size': 28})
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), subplot_kw={'projection': None})
    
    fig.suptitle('Temperature and wind at Nigardsbreen last 48 hours', x=.48, y=0.95)
    fig.text(0.48, 0.84, f'Weather now: {last_temperature:.0f}$\u00b0$C and {last_wind_speed:.0f} m/s from {get_wind_direction(last_wind_direction)}     ',
             ha='center', va='bottom', fontsize=18)#, fontweight='bold')
    
    # temperature subplot
    
    n = 42
    x = NB['date'].values[-49:-1]
    y = NB['t_u'].values[-49:-1]
    ymin = round(np.min(y)) - 1
    ymax = round(np.max(y)) + 4
    sind = 5
    ind = 6
    
    ax1.plot(x, y, color="darkgrey", lw=2.5)
    
    p = TemperaturePlot()
    p.scatter(x[sind::ind], y[sind::ind] + 2, s=300, temp=y[sind::ind], c=y[sind::ind], edgecolor="k", cmap="RdYlBu_r", ax=ax1)
    
    ax1.set_ylim(ymin, ymax)
    ax1.set_xticks([NB['date'].values[-49], NB['date'].values[-37], NB['date'].values[-25], NB['date'].values[-13], NB['date'].values[-1]])
    ax1.set_xticklabels(['48', '36', '24', '12', '0'])
    ax1.set_xlabel('Hours ago')
    ax1.set_ylabel('Temperature (\u00b0C)')
    
    # wind subplot
    
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    for spine in ax2.spines.values():
        spine.set_visible(False)
    
    ax2 = fig.add_subplot(122, polar=True)
    
    angles = np.radians(wind_directions)
    last_angle = np.radians(last_wind_direction)
    
    ax2.set_theta_direction(-1)  # Reverse the direction of the angles (clockwise)
    ax2.set_theta_offset(np.pi / 2.0)  # Set 0 degrees to North (top of the plot)
    
    ax2.scatter(angles, wind_speeds, edgecolors='black', facecolors='none', s=100, label='last 48 hours')  # Previous observations
    ax2.scatter(last_angle, last_wind_speed, color='red', s=200, zorder=5, label='last hour')
    
    ax2.set_ylim(0, np.ceil(last_48['wspd_u'].max()))  # Extend limit slightly above max wind speed
    ax2.set_yticks(np.arange(1, int(last_48['wspd_u'].max()) + 1, 1))
    
    y_labels = ['' if i % 2 != 0 else f'{i} m/s' for i in range(1, int(last_48['wspd_u'].max()) + 1)]
    ax2.set_yticklabels([])
    for i, label in enumerate(y_labels):
        angle = i * (np.pi / len(y_labels))  # Calculate the angle for each label
        ax2.text(0.8, (i+1), label, ha='center', va='center', rotation=-45, 
                 fontsize=22, color='black')  # Adjust rotation as needed
    
    
    ax2.set_xticks(np.radians([0, 90, 180, 270]))  # North, East, South, West
    ax2.set_xticklabels(['North', '    East', 'South', 'West     '])
    
    legend = ax2.legend(loc='lower right', title='Wind', bbox_to_anchor=(.3, -.08), fontsize=18)
    legend.get_frame().set_alpha(None)
    
    timestamp = pd.Timestamp(NB['date'].values[-1]+pd.Timedelta(hours=2)).strftime('%Y-%m-%d %H:%M')
    ax2.text(1.15, -.06, f'Last measurement:\n{timestamp}  ', 
             fontsize=10, transform=ax2.transAxes, ha='right', va='bottom')
    
    plt.tight_layout()
    plt.savefig('fig.png')

if __name__ == "__main__":
    plot()
