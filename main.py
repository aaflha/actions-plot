import matplotlib.pyplot as plt
import numpy as np
import datetime
import pandas as pd
import os
import matplotlib.path as mpath
import requests
from datetime import datetime
from matplotlib import gridspec
import geopandas as gpd

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

def get_met_data():
    client_id = str(os.environ["MET_ID"])
    
    now = datetime.now()+pd.Timedelta(days=1)
    then = now-pd.Timedelta(hours=72)
    now = now.strftime("%Y-%m-%d")
    then = then.strftime("%Y-%m-%d")
    
    # Define endpoint and parameters
    endpoint = 'https://frost.met.no/observations/v0.jsonld'
    parameters = {
        'sources': 'SN55430',
        'elements': 'min(air_temperature PT1H), max(air_temperature PT1H), sum(precipitation_amount PT1H)', # mean PT1H doesn't work...
        'referencetime': f'{then}/{now}', #'2010-04-01/2010-04-03',
    }
    # Issue an HTTP GET request
    r = requests.get(endpoint, parameters, auth=(client_id,''))
    # Extract JSON data
    json = r.json()
    
    # Check if the request worked, print out any errors
    if r.status_code == 200:
        data = json['data']
        print('Data retrieved from frost.met.no!')
    else:
        print('Error! Returned status code %s' % r.status_code)
        print('Message: %s' % json['error']['message'])
        print('Reason: %s' % json['error']['reason'])
    
    # based on https://frost.met.no/python_example.html

    MG = pd.DataFrame()
    for i in range(len(data)):
        row = pd.DataFrame(data[i]['observations'])
        row['referenceTime'] = data[i]['referenceTime']
        row['sourceId'] = data[i]['sourceId']
        MG = pd.concat([MG, row], ignore_index=True)
    MG = MG.reset_index()
    
    MG['time'] = pd.to_datetime(MG['referenceTime']) # time is in UTC
    MG = MG.pivot(index='time', columns='elementId', values='value')
    MG.reset_index(inplace=True)
    MG.columns.name = None
    MG = MG.rename(columns={'max(air_temperature PT1H)': 'T_max', 'min(air_temperature PT1H)': 'T_min', 'sum(precipitation_amount PT1H)': 'precip'})

    return MG
    
def plot():

    # load observations
    
    NB = pd.read_csv(str(os.environ["HOURLY_SECRET"]), sep=',')
    #NB = pd.read_csv(str(os.environ["DAILY_SECRET"]), sep=',')
    NB['date'] = pd.to_datetime(NB['time'])
    NB = (NB.loc[np.where(NB['date'] >= pd.Timestamp('2021-07-01 00:00:00'))[0][0]:]).reset_index()

    MG = get_met_data()

    # load map

    path = "https://raw.githubusercontent.com/krifla/nigardsbreen_data-campaign/main/data/map/"

    file = "Basisdata_4644_Luster_25833_N50Hoyde_GML.gml"
    url = path+file
    terrain = gpd.read_file(url, layer='Høydekurve')
    
    file = "Basisdata_4644_Luster_25833_N50Arealdekke_GML.gml"
    url = path+file
    lakes = gpd.read_file(url, layer='Innsjø')
    lakes = lakes.loc[(~np.isnan(lakes['vatnLøpenummer']))&(lakes['høyde']>200)&(lakes['høyde']<400)]
    glaciers = gpd.read_file(url, layer='SnøIsbre')
    rivers = gpd.read_file(url, layer='Elv')    
    
    terrain_100 = terrain[terrain['høyde'] % 100 == 0]
    terrain_500 = terrain[terrain['høyde'] % 500 == 0]
    glaciers_simp = glaciers.simplify(20, preserve_topology=True)
    glaciers_buffered = glaciers_simp.buffer(.1, resolution=4)
    terrain_glaciers_100 = terrain_100.intersection(glaciers_buffered.union_all())
    terrain_glaciers_500 = terrain_500.intersection(glaciers_buffered.union_all())
    
    file = "ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp"
    url = path+file
    world = gpd.read_file(url)
    norway = world[world['ADMIN'] == 'Norway']
    
    # define last observations
    
    last_48 = NB.tail(48)

    wind_speeds = last_48['wspd_u'].values
    wind_directions = last_48['wdir_u'].values
    
    if pd.isna(wind_speeds[-1]) or pd.isna(wind_directions[-1]):
        last_observation = last_48.iloc[-2]  # Use second last if last is NaN
    else:
        last_observation = last_48.iloc[-1]  # Last observation
    
    last_wind_speed = last_observation['wspd_u']
    last_wind_direction = last_observation['wdir_u']
    last_temperature = last_observation['t_u']
    last_temperature_MG = ((MG['T_max'].tail(1)+MG['T_min'].tail(1))/2).values[0]
    
    # plot
    
    plt.rcParams.update({'font.size': 28})
    
    res=1.2
    fig = plt.figure(figsize=(18*res, 8*res))
    
    # Create a GridSpec with different width ratios
    gs = gridspec.GridSpec(1, 3, width_ratios=[2, 3, 1.2])  # The third subplot will be twice as wide
    
    # Create subplots using the GridSpec
    ax1 = fig.add_subplot(gs[0])
    ax3 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    
    # text
    
    fig.suptitle('Weather last 48 hours |', x=.066, y=.97, ha='left', va='top', fontsize=60) # x=.48
    fig.text(0.53, 0.98, f'Nigardsbreen$^1$ now: in winter sleep')#{last_temperature:.0f}$\u00b0$C and {last_wind_speed:.0f} m/s from {get_wind_direction(last_wind_direction)}', #     \nWeather near Breheimsenteret* now: {last_temperature_MG:.0f}$\u00b0$C \n ',
             #ha='left', va='top', fontsize=27, color='C9')
    fig.text(0.53, 0.87, f'Mjølversgrendi$^2$ now: {last_temperature_MG:.0f}$\u00b0$C',
             ha='left', va='bottom', fontsize=27, color='C1')
    
    fig.text(0.066, 0.01, f'1: Nigardsbreen weather station, operated by Western Norway University of Applied Sciences\n2: Mjølversgrendi weather station, operated by Norwegian Meteorological Institute and located near Breheimsenteret',
             ha='left', va='bottom', fontsize=14)
    
    timestamp = pd.Timestamp(NB['date'].values[-1]+pd.Timedelta(hours=2)).strftime('%Y-%m-%d %H:%M')
    fig.text(.949, .01, f'Last measurement: {timestamp}', 
             fontsize=14, ha='right', va='bottom')
    
    # temperature subplot
    
    n = 42
    x_NB = NB['date'].values[-49:-1]
    y_NB = NB['t_u'].values[-49:-1]
    x_MG = MG['time'].tail(48)
    y_MG = (MG['T_max'].tail(48)+MG['T_min'].tail(48))/2
    if np.any(~np.isnan(y_NB)):
        ymin = round(np.nanmin([np.nanmin(y_NB), np.nanmin(y_MG), 20])) - 1
        ymax = round(np.nanmax([np.nanmax(y_NB), np.nanmax(y_MG), 10])) + 2
    else:
        ymin = round(np.nanmin([np.nanmin(y_MG), 20])) - 1
        ymax = round(np.nanmax([np.nanmax(y_MG), 10])) + 2
    sind = 3
    ind = 6
    
    #ax1.plot(x_NB, y_NB, color='C9', lw=2.5, label='Nigardsbreen$^1$')
    ax1.plot(x_MG, y_MG, color='C1', lw=2.5, label='Mjølversgrendi$^2$')
    ax1.scatter(x_NB[-1:], y_NB[-1:], 100, marker='o', color='C9')
    ax1.scatter(x_MG[-1:], y_MG[-1:], 100, marker='o', color='C1')
    
    ax1.set_ylim(ymin, ymax)
    yticks = ax1.get_yticks()
    ax1.set_yticks(yticks[1:-1])
    ax1.set_xticks([NB['date'].values[-49], NB['date'].values[-25], NB['date'].values[-1]])
    #ax1.set_xticks([x_MG[0], x_MG[24], x_MG[-1])
    ax1.set_xticklabels(['48 h ago', '24 h ago', 'Now'])
    ax1.set_xlabel(' ')
    ax1.set_ylabel('Temperature (\u00b0C)')
    ax1.legend(fontsize=16)
    
    # wind subplot
    
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    for spine in ax2.spines.values():
        spine.set_visible(False)
    
    ax2 = fig.add_subplot(133, polar=True)
    
    angles = np.radians(wind_directions)
    last_angle = np.radians(last_wind_direction)
    
    ax2.set_theta_direction(-1)  # Reverse the direction of the angles (clockwise)
    ax2.set_theta_offset(np.pi / 2.0)  # Set 0 degrees to North (top of the plot)
    
    ax2.scatter(angles, wind_speeds, edgecolors='black', facecolors='none', s=100, label='last 48 hours')  # Previous observations
    ax2.scatter(last_angle, last_wind_speed, color='C9', s=200, zorder=5, label='last hour')
    
    if np.any(~np.isnan(y_NB)):
        ax2.set_ylim(0, np.ceil(last_48['wspd_u'].max()))  # Extend limit slightly above max wind speed
        ax2.set_yticks(np.arange(1, int(last_48['wspd_u'].max()) + 1, 1))
    else:
        ax2.set_ylim(0, 10)
        ax2.set_yticks(np.arange(1, 11, 1))
    
    y_labels = ['' if i % 2 != 0 else f'{i} m/s' for i in range(1, int(last_48['wspd_u'].max()) + 1)]
    ax2.set_yticklabels([])
    for i, label in enumerate(y_labels):
        angle = i * (np.pi / len(y_labels))  # Calculate the angle for each label
        ax2.text(0.8, (i+1), label, ha='center', va='center', rotation=-45, 
                 fontsize=24, color='black')  # Adjust rotation as needed
    
    ax2.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))  # North, East, South, West
    ax2.set_xticklabels(['N', 'NE', 'E ', 'SE', 'S', 'SW', 'W', 'NW'])
    
    legend = ax2.legend(loc='lower right', title='Wind Nigardsbreen', bbox_to_anchor=(.92, -.55), fontsize=18, title_fontsize=22) # bbox_to_anchor=(.3, -.08)
    legend.get_frame().set_alpha(None)
    
    xticklabels = ax2.get_xticklabels()
    xticklabels[0].set_verticalalignment('bottom')  # Align N label
    xticklabels[1].set_horizontalalignment('left')  # Align NE label
    xticklabels[2].set_horizontalalignment('left')  # Align E label
    xticklabels[3].set_horizontalalignment('left')  # Align SE label
    xticklabels[4].set_verticalalignment('top')     # Align S label
    xticklabels[5].set_horizontalalignment('right') # Align SW label
    xticklabels[6].set_horizontalalignment('right') # Align W label
    xticklabels[7].set_horizontalalignment('right') # Align NW label
    
    # plot map
    
    ax3.scatter(7.197794684331172, 61.686051540061946, ec='k', c='C9',       s=500)
    ax3.scatter(7.275990675443110, 61.659358589432706, ec='k', c='C1',       s=500)
    ax3.scatter(7.275437085733799, 61.65134930139541,  ec='k', c='darkgrey', s=500)
    txt = ax3.text(7.197794684331172+.007, 61.686051540061946-.007, 'Nigardsbreen\nweather station', fontsize=22, ha='left', va='bottom')
    txt.set_bbox(dict(facecolor='snow', alpha=.6, edgecolor='none'))
    txt = ax3.text(7.3, 61.659358589432706+.003, 'Mjølversgrendi\nweather station', fontsize=22, ha='right', va='bottom')
    txt.set_bbox(dict(facecolor='snow', alpha=.6, edgecolor='none'))
    txt = ax3.text(7.3, 61.65134930139541-.003, 'Breheimsenteret', fontsize=22, ha='right', va='top')
    txt.set_bbox(dict(facecolor='snow', alpha=.6, edgecolor='none'))
    
    terrain_100.to_crs(epsg=4326).plot(ax=ax3, lw=1, color='tan', zorder=-1000)
    terrain_500.to_crs(epsg=4326).plot(ax=ax3, lw=2, color='tan', zorder=-1000)
    
    glaciers.to_crs(epsg=4326).plot(ax=ax3, color='white', edgecolor='k', zorder=-500)
    terrain_glaciers_100.to_crs(epsg=4326).plot(ax=ax3, color='skyblue', lw=1, zorder=-100)
    terrain_glaciers_500.to_crs(epsg=4326).plot(ax=ax3, color='skyblue', lw=2, zorder=-100)
    rivers.to_crs(epsg=4326).plot(ax=ax3, color='skyblue', zorder=-100)
    lakes.to_crs(epsg=4326).plot(ax=ax3, color='lightblue', edgecolor='skyblue', zorder=-100)
    
    ax3.set_facecolor('snow')
    
    xmin=7.15; xmax=7.303; ymin=61.64; ymax=61.71
    
    ax3.set_xlim(xmin,xmax)
    ax3.set_ylim(ymin,ymax)
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    plt.tight_layout(w_pad=-3)
    plt.savefig('fig.png')

if __name__ == "__main__":
    plot()
