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


def date_list_generator(date):
    firt_date = pd.to_datetime(date)
    date_list = [firt_date]
    for iw in range(11):
        next_date = firt_date + pd.to_timedelta((iw+1)*7, 'D')
        #print(next_date)
        date_list.append(next_date)
    return(date_list)


def create_granule_dict(input_dir):
    '''
    this is where we talk

    '''
    input_path = input_dir
    granule = dict({
            'btm1_block' : None, # bottom
            'wth1_block' : None, # weather
            'twr1_block' : None, # top at work resolution
            'thr1_block' : None, # top at high resolution
            'soi1_block' : None, # soil

            'btm1_path' : None, # bottom
            'wth1_path' : None, # weather
            'twr1_path' : None, # top at work resolution
            'thr1_path' : None, # top at high resolution
            'soi1_path' : None, # tsoil
    })

    for (root,dirs,files) in os.walk(input_path, topdown=True):
        for file in files:
            #print(file)
            if 'FBTM_SO' in file :
                granule['soi1_path'] = os.path.join(input_path,file)
                granule['soi1_block'] = xr.open_dataset(os.path.join(input_path, file))
            if 'FBTM_DO' in file :
                granule['btm1_path'] = os.path.join(input_path,file)
                granule['btm1_block'] = xr.open_dataset(os.path.join(input_path, file))
            if 'FW1D' in file :
                granule['wth1_path'] = os.path.join(input_path,file)
                granule['wth1_block'] = xr.open_dataset(os.path.join(input_path, file))
                fsd =  granule['wth1_block'].attrs['Forecast_transit_date']
            if 'FTOP' in file :
                if 'HRes' in file :
                    granule['thr1_path'] = os.path.join(input_path,file)
                    granule['thr1_block'] = xr.open_dataset(os.path.join(input_path, file))
                if 'WRes' in file :
                    granule['twr1_path'] = os.path.join(input_path,file)
                    granule['twr1_block'] = xr.open_dataset(os.path.join(input_path, file))
    return granule

def build_evapotranspiration_map(forecast_date, twr_block):
    '''
    this is whrere we talk 
    '''
    up_to_date = forecast_date- pd.Timedelta('1 days') 

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
    #plt.tight_layout()
    return plt

def build_leaching_map(forecast_date, btm_block):
    '''
    '''
    #standard
    up_to_date = forecast_date - pd.Timedelta('1 days') 

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

    geom = shapely.wkt.loads(btm_block.aoi)
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
    # Show plots,
    plt.tight_layout()
    return plt

def build_precip_irri_map(wth_block, twr_block, forecast_transit_date):

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
    return plt

def et_time_series_with_sigmas(twr_block):
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
    under_line = (eta_est_df-eta_sig_df)[0]
    over_line = (eta_est_df+eta_sig_df)[0]
    under_line[under_line < 0] = 0
    etlabel ="ET mm"
    i2xw = eta_sig_df*0+24.5*0.3 # 2 week line metric

    #Plotting:
    plt.figure(figsize=(12,2))
    plt.ylabel(etlabel)
    plt.plot(eta_est_df, color='orange', linewidth=2) #mean curve.
    plt.fill_between(eta_sig_df.index, under_line, over_line, color='C1', alpha=.2) #std curves.
    plt.plot(i2xw, linewidth=1, color='r', linestyle='dashed' ) 
    plt.plot(i2xw*0, linewidth=1, color='r', linestyle='dashed', )

    return plt


def field_health_pdf_page(date, thr_block, levels, output):
    # set dates
    date_list = date_list_generator(date)

    lai_da = thr_block.LAI.sel(time = date_list)
    geom = shapely.wkt.loads(twr_block.aoi)
    g = gpd.GeoSeries([geom])

    # Irregular levels to illustrate the use of a proportional colorbar
    levels = levels
    # Prepare the figure
    f, ax_gr = plt.subplots(4, 3, figsize=(13, 18))
    for liy in range(4):
        for lix in range(3):
            ax1 = ax_gr[liy, lix]
        
            lai_da.isel(time=(liy*3+lix)).plot(
                ax=ax1, cmap='YlGn',levels=levels, cbar_kwargs={"ticks": levels, "spacing": "proportional"}
            )
            g.plot(ax=ax1, facecolor='none', edgecolor='royalblue',linewidth=4)
            ax1.xaxis.set_major_locator(ticker.NullLocator())
            ax1.yaxis.set_major_locator(ticker.NullLocator())
            ax1.set(xlabel=None)
            ax1.set(ylabel=None)

    # Show plots
    plt.tight_layout()
    plt.savefig(output)

def leaching_risk_page(date, btm_block, output):
    date_list = date_list_generator(date) 
    rsk_da = btm_block.wexcess_risk.sel(time = date_list)
    rsklabel ="Leaching Risk (%)"
    # Irregular levels to illustrate the use of a proportional colorbar
    levels_cmbn = np.array([q for q in range(0,11)])*10+0.001
    levels = levels_cmbn.tolist()
    geom = shapely.wkt.loads(btm_block.aoi)
    g = gpd.GeoSeries([geom])


    f, ax_gr = plt.subplots(4, 3, figsize=(13, 18))
    for liy in range(4):
        for lix in range(3):
            ax1 = ax_gr[liy, lix]
            rsk_da.isel(time=(liy*3+lix)).plot(ax=ax1, cmap='YlGnBu', levels=levels, 
                       cbar_kwargs={"label": rsklabel, "ticks": levels, "spacing": "proportional"})
        
            g.plot(ax=ax1, facecolor='none', edgecolor='Orange',linewidth=4)
            ax1.xaxis.set_major_locator(ticker.NullLocator())
            ax1.yaxis.set_major_locator(ticker.NullLocator())
            ax1.set(xlabel=None)
            ax1.set(ylabel=None)

    # Show plots
    plt.tight_layout()

    # save the file 4

    plt.savefig(output)

def drought_page(date, btm_block, output):
    date_list = date_list_generator(date)    
    rsk_da = btm_block.wshort_risk.sel(time = date_list)
    rsklabel ="Leaching Risk (%)"
    # Irregular levels to illustrate the use of a proportional colorbar
    levels_cmbn = np.array([q for q in range(0,11)])*10+0.001
    levels = levels_cmbn.tolist()
    geom = shapely.wkt.loads(btm_block.aoi)
    g = gpd.GeoSeries([geom])


    f, ax_gr = plt.subplots(4, 3, figsize=(13, 18))
    for liy in range(4):
        for lix in range(3):
            ax1 = ax_gr[liy, lix]
            rsk_da.isel(time=(liy*3+lix)).plot(ax=ax1, cmap='RdYlGn_r', levels=levels, 
                           cbar_kwargs={"label": rsklabel, "ticks": levels, "spacing": "proportional"})
        
            g.plot(ax=ax1, facecolor='none', edgecolor='Orange',linewidth=4)
            ax1.xaxis.set_major_locator(ticker.NullLocator())
            ax1.yaxis.set_major_locator(ticker.NullLocator())
            ax1.set(xlabel=None)
            ax1.set(ylabel=None)

    # Show plots
    plt.tight_layout()
    # save the file 5


    plt.savefig(output)

def soil_moisture_weekly(date, btm_block, levels, output):
    date_list = date_list_generator(date)
    smc_da = btm_block.assim_rz_sm_est.sel(time = date_list)
    smc_da_in = smc_da*1000/25.4
    smlabel ="Abs Soil Moisture Content (in)"
    # Irregular levels to illustrate the use of a proportional colorbar
    levels_cmbn = (np.array(levels)+4)
    levels = levels_cmbn.tolist()
    geom = shapely.wkt.loads(btm_block.aoi)
    g = gpd.GeoSeries([geom])


    f, ax_gr = plt.subplots(4, 3, figsize=(13, 18))
    for liy in range(4):
        for lix in range(3):
            ax1 = ax_gr[liy, lix]
            smc_da_in.isel(time=(liy*3+lix)).plot(ax=ax1, cmap='YlGnBu', levels=levels, 
                           cbar_kwargs={"label": smlabel, "ticks": levels, "spacing": "proportional"})
        
            g.plot(ax=ax1, facecolor='none', edgecolor='Orange',linewidth=4)
            ax1.xaxis.set_major_locator(ticker.NullLocator())
            ax1.yaxis.set_major_locator(ticker.NullLocator())
            ax1.set(xlabel=None)
            ax1.set(ylabel=None)

    # Show plots
    plt.tight_layout()

    # save the file 3rd

    plt.savefig(output)

def crop_water_use_maps_page(date, twr_block, levels, output):

    # set dates
    first_date = pd.to_datetime(date)
    date_list = date_list_generator(first_date)

    eta_da = twr_block.ETA_est_mm_day.sel(time = date_list)
    eta_da_in = eta_da/25.4*7

    geom = shapely.wkt.loads(twr_block.aoi)
    # global g
    g = gpd.GeoSeries([geom])
    etlabel ="ET in"
    

    # Irregular levels to illustrate the use of a proportional colorbar
    levels_cmbn = np.array(levels)*1.12/25.4*7
    levels = levels_cmbn.tolist()

    f, ax_gr = plt.subplots(4, 3, figsize=(13, 18))
    for liy in range(4):
        for lix in range(3):
            ax1 = ax_gr[liy, lix]
            eta_da_in.isel(time=(liy*3+lix)).plot(ax=ax1, cmap='Oranges', levels=levels, 
                       cbar_kwargs={"label": etlabel, "ticks": levels, "spacing": "proportional"})
        
            g.plot(ax=ax1, facecolor='none', edgecolor='royalblue',linewidth=4)
            ax1.xaxis.set_major_locator(ticker.NullLocator())
            ax1.yaxis.set_major_locator(ticker.NullLocator())
            ax1.set(xlabel=None)
            ax1.set(ylabel=None)

    # Show plots
    plt.title('Crop Water Use Variability')
    plt.tight_layout()
    # save the file 2nd
    plt.savefig(output)
    