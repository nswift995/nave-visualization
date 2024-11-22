###################################################################################################
#  For collaboration with Nave (from Val)
###################################################################################################


# general imports 
import datetime
import geopandas as gpd
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import math
import numpy as np
import os
import pandas as pd
import shapely 
import sys
import xarray as xr
from rasterio import plot
from osgeo import gdal

###################################################################################################
#  Get the granule 
###################################################################################################

input_path = '/granule/location/directory/' 

btm_block = None # bottom
wth_block = None # weather
twr_block = None # top at work resolution
thr_block = None # top at high resolution
sls_block = None # soil properties
ssm_block = None # surface soil moisture

for (root,dirs,files) in os.walk(input_path, topdown=True):
    for file in files:
        print(file)
        if 'SSMCV4' in file :
            ssm_block = xr.open_dataset(os.path.join(input_path, file))
        if 'SGR2' in file :
            sls_block = xr.open_dataset(os.path.join(input_path, file))
        if 'FBTM' in file :
            btm_block = xr.open_dataset(os.path.join(input_path, file))
        if 'FW1D' in file :
            wth_block = xr.open_dataset(os.path.join(input_path, file))
        if 'FTOP' in file :
            if 'HRes' in file :
                thr_block = xr.open_dataset(os.path.join(input_path, file))
            if 'WRes' in file :
                twr_block = xr.open_dataset(os.path.join(input_path, file))    

# forecast transit date 
forecast_transit_date = pd.to_datetime(wth_block.attrs['Forecast_transit_date'])
up_to_date = forecast_transit_date
print(forecast_transit_date)

###################################################################################################
#  make ET maps 
###################################################################################################

#standard
up_to_date = forecast_transit_date- pd.Timedelta('1 days') 

# for eye candy
past_3d = up_to_date - pd.timedelta_range(start='1 day', periods=3) 
past_7d = up_to_date - pd.timedelta_range(start='1 day', periods=7) 

eta_3d_da = twr_block.ETA_est_mm_day.sel(time = past_3d)
eta_7d_da = twr_block.ETA_est_mm_day.sel(time = past_7d)

eta_3d_da_cs = eta_3d_da.sum(dim='time')
eta_7d_da_cs = eta_7d_da.sum(dim='time')
etlabel ="ET mm"

eta_3d_da_cs = eta_3d_da.sum(dim='time')/25.4 # Imperial Units
eta_7d_da_cs = eta_7d_da.sum(dim='time')/25.4 # Imperial Units
eta_cm_da_cs = np.concatenate((eta_3d_da_cs, eta_7d_da_cs))
etlabel ="ET in"

# Prepare the figure
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 3))
# Color levels
levels_3day = np.array([(i+1)/8.0 for i in range(8)])*(np.nanmax(eta_3d_da_cs)+0.1- np.nanmin(eta_3d_da_cs))+np.nanmin(eta_3d_da_cs)
levels_7day = np.array([(i+1)/8.0 for i in range(8)])*(np.nanmax(eta_7d_da_cs)+0.1- np.nanmin(eta_7d_da_cs))+np.nanmin(eta_7d_da_cs)
levels_cmbn = np.array([(i+1)/8.0 for i in range(8)])*(np.nanmax(eta_cm_da_cs)+0.1- np.nanmin(eta_cm_da_cs))+np.nanmin(eta_cm_da_cs)


# Plot data
levels = levels_cmbn.tolist()
eta_7d_da_cs.plot(ax=ax1, cmap='Oranges', levels=levels, cbar_kwargs={"label": etlabel, "ticks": levels, "spacing": "proportional"})

levels = levels_cmbn.tolist()
eta_3d_da_cs.plot(ax=ax2, cmap='Oranges', levels=levels, cbar_kwargs={"label": etlabel, "ticks": levels, "spacing": "proportional"})
geom = shapely.wkt.loads(twr_block.aoi)
g = gpd.GeoSeries([geom])
g.plot(ax=ax1, facecolor='none', edgecolor='royalblue',linewidth=4)
g.plot(ax=ax2, facecolor='none', edgecolor='royalblue',linewidth=4)


ax1.title.set_text("Since 7 days ago")
ax2.title.set_text("Since 3 days ago")
#ax1.set_axis_off()
#ax2.set_axis_off()
ax1.xaxis.set_major_locator(ticker.NullLocator())
ax1.yaxis.set_major_locator(ticker.NullLocator())
ax2.xaxis.set_major_locator(ticker.NullLocator())
ax2.yaxis.set_major_locator(ticker.NullLocator())
ax1.set(xlabel=None)
ax1.set(ylabel=None)
ax2.set(xlabel=None)
ax2.set(ylabel=None)
# Show plots,
plt.tight_layout()


###################################################################################################
#  make Leaching maps 
###################################################################################################


#standard
up_to_date = forecast_transit_date - pd.Timedelta('1 days') 

# for eye candy
past_1d = up_to_date 
past_7d = up_to_date - pd.Timedelta('7 days') 
futr_7d = up_to_date + pd.Timedelta('7 days') 

leach_risk_p1d_da = btm_block.wexcess_risk.sel(time = past_1d)
leach_risk_p7d_da = btm_block.wexcess_risk.sel(time = past_7d)
leach_risk_f7d_da = btm_block.wexcess_risk.sel(time = futr_7d)

etlabel ="Leaching Risk %"

# Prepare the figure
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
# Color levels
levels=np.array([q for q in range(0,11)])*10

# Plot data
leach_risk_p7d_da.plot(ax=ax1, cmap = plt.cm.YlGnBu, levels=levels, cbar_kwargs={"label": etlabel, "ticks": levels, "spacing": "proportional"})
leach_risk_p1d_da.plot(ax=ax2, cmap = plt.cm.YlGnBu, levels=levels, cbar_kwargs={"label": etlabel, "ticks": levels, "spacing": "proportional"})
leach_risk_f7d_da.plot(ax=ax3, cmap = plt.cm.YlGnBu, levels=levels, cbar_kwargs={"label": etlabel, "ticks": levels, "spacing": "proportional"})

                       
geom = shapely.wkt.loads(twr_block.aoi)
g = gpd.GeoSeries([geom])
g.plot(ax=ax1, facecolor='none', edgecolor='royalblue',linewidth=4)
g.plot(ax=ax2, facecolor='none', edgecolor='royalblue',linewidth=4)
g.plot(ax=ax3, facecolor='none', edgecolor='royalblue',linewidth=4)

ax1.xaxis.set_major_locator(ticker.NullLocator())
ax1.yaxis.set_major_locator(ticker.NullLocator())
ax2.xaxis.set_major_locator(ticker.NullLocator())
ax2.yaxis.set_major_locator(ticker.NullLocator())
ax3.xaxis.set_major_locator(ticker.NullLocator())
ax3.yaxis.set_major_locator(ticker.NullLocator())
ax1.set(xlabel=None)
ax1.set(ylabel=None)
ax2.set(xlabel=None)
ax2.set(ylabel=None)
ax3.set(xlabel=None)
ax3.set(ylabel=None)

#ax1.title.set_text("Week ago")
#ax2.title.set_text("Today")
#ax3.title.set_text("Next week")


# Show plots,
plt.tight_layout()


###################################################################################################
#  make precipitation / irrigation time series 
###################################################################################################


precip_array = wth_block.Precip_m_d.data #* 100.0 # to mm Note it has to be 100
precip_array = precip_array.reshape([precip_array.shape[0]],)/25.4
precip_df = pd.DataFrame({'precip': precip_array.tolist()},
                   index=wth_block.Precip_m_d.time.values)    

#Plotting:
plt.figure(figsize=(12,1))
date = precip_df.index.to_list()
plt.bar(date, precip_df['precip'].tolist())


# irrigation part

if 'irrigtn_mm_d' in list(twr_block.keys()):
    irr_array = twr_block.irrigtn_mm_d.isel(lat=0,lon=0).data
    irr_array = irr_array.reshape([irr_array.shape[0]],)/25.4
    irr_df = pd.DataFrame({'irrigtn': irr_array.tolist()},
                   index=twr_block.irrigtn_mm_d.time.values)
    plt.bar(date, irr_df['irrigtn'].tolist(), color = 'orange')
    
# forecast split bar 
plt.plot([forecast_transit_date, forecast_transit_date], 
         [0 , precip_df['precip'].max()], color = 'red', ls= '--')
plt.text(forecast_transit_date, precip_df['precip'].max(), 'Observations ',
        verticalalignment='bottom', horizontalalignment='right',
        color='green', fontsize=10)
plt.text(forecast_transit_date, precip_df['precip'].max(), ' Forecast',
        verticalalignment='bottom', horizontalalignment='left',
        color='red', fontsize=10)



###################################################################################################
#  make ET time series with signas
###################################################################################################
# set first to 0 always 
twr_block.ETA_est_mm_day.loc[dict(time=twr_block.time.min())] = 0
twr_block.ETA_sgm_mm_day.loc[dict(time=twr_block.time.min())] = 0
# then as usual
eta_est_arr = twr_block.ETA_est_mm_day.median(dim =['lat', 'lon'])
eta_sgm_arr = twr_block.ETA_sgm_mm_day.median(dim =['lat', 'lon'])
# imperial units 
#eta_est_arr = eta_est_arr/25.4
#eta_sgm_arr = eta_sgm_arr/25.4


eta_est_df = pd.DataFrame({0: eta_est_arr.data.tolist()},
                    index=eta_est_arr.time.values)
eta_sig_df = pd.DataFrame({0: eta_sgm_arr.data.tolist()},
                   index=eta_sgm_arr.time.values)
under_line     = (eta_est_df-eta_sig_df)[0]
over_line      = (eta_est_df+eta_sig_df)[0]
under_line[under_line < 0] = 0
etlabel ="ET mm"
i2xw = eta_sig_df*0+24.5*0.3 # 2 week line metric
# etlabel ="ET in" imperial
#i2xw = eta_sig_df*0+03 # 2 week line imperial
#Plotting:
plt.figure(figsize=(12,2))
plt.ylabel(etlabel)
plt.plot(eta_est_df, color='orange', linewidth=2) #mean curve.
plt.fill_between(eta_sig_df.index, under_line, over_line, color='C1', alpha=.2) #std curves.
plt.plot(i2xw, linewidth=1, color='r', linestyle='dashed' ) 
plt.plot(i2xw*0, linewidth=1, color='r', linestyle='dashed', ) 

