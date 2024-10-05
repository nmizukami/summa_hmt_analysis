import numpy as np
import xarray as xr

#error metrics
def nse(obs, sim):
    """
    Calculates The Nash–Sutcliffe efficiency (NSE) between two time series arrays.

    Arguments
    ---------
    sim: array-like
        Simulated time series array.
    obs: array-like
        Observed time series array.

    Returns
    -------
    nse: float
        nse calculated between the two arrays.
    """
    mask = np.logical_and(~np.isnan(obs), ~np.isnan(sim))
    return 1.0 - (np.sum((sim[mask]-obs[mask])**2)/np.sum((obs[mask]-np.mean(obs[mask]))**2))

def corr(obs, sim):
    mask = np.logical_and(~np.isnan(obs), ~np.isnan(sim))
    return np.corrcoef(sim[mask],obs[mask])[0,1]

def alpha(obs, sim):
    """
    Calculates ratio of variabilities of two time series arrays.

    Arguments
    ---------
    sim: array-like
        Simulated time series array.
    obs: array-like
        Observed time series array.

    Returns
    -------
    alpha: float
        variability ratio calculated between the two arrays.
    """
    mask = np.logical_and(~np.isnan(obs), ~np.isnan(sim))
    m_sim=np.mean(sim[mask])
    m_obs=np.mean(obs[mask])
    std_sim = np.std(sim[mask], ddof=1)
    std_obs = np.std(obs[mask], ddof=1)
    return (std_sim/m_sim)/(std_obs/m_obs)

def beta(obs, sim):
    """
    Calculates ratio of means of two time series arrays.

    Arguments
    ---------
    sim: array-like
        Simulated time series array.
    obs: array-like
        Observed time series array.

    Returns
    -------
    beta: float
        mean ratio calculated between the two arrays.
    """
    mask = np.logical_and(~np.isnan(obs), ~np.isnan(sim))
    return np.mean(sim[mask])/np.mean(obs[mask])

def kge(obs, sim):
    """
    Calculates the Kling-Gupta Efficiency (KGE) between two flow arrays.
    Arguments
    ---------
    sim: array-like
        Simulated time series array.
    obs: array-like
        Observed time series array.

    Returns
    -------
    kge: float
        Kling-Gupta Efficiency calculated between the two arrays.
    """
    kge = 1.0 - np.sqrt((corr(obs, sim)-1.0)**2 + (alpha(obs, sim)-1.0)**2 + (beta(obs, sim)-1.0)**2)
    return kge

def rmse(obs, sim):
    """
    Calculates root mean squared of error (rmse) between two flow arrays.
    Arguments
    ---------
    sim: array-like
        Simulated time series array.
    obs: array-like
        Observed time series array.

    Returns
    -------
    rmse: float
        rmse calculated between the two arrays.
    """
    mask = np.logical_and(~np.isnan(obs), ~np.isnan(sim))
    return np.sqrt(np.nanmean((sim[mask]-obs[mask])**2))


def pbias(obs, sim):
    """
    Calculates percentage bias between two flow arrays.
    Arguments
    ---------
    sim: array-like
        Simulated time series array.
    obs: array-like
        Observed time series array.

    Returns
    -------
    pbial: float
        percentage bias calculated between the two arrays.
    """
    mask = np.logical_and(~np.isnan(obs), ~np.isnan(sim))
    m_sim=np.mean(sim[mask])
    m_obs=np.mean(obs[mask])
    return np.sum((m_sim-m_obs))/np.sum(m_obs)


def mae(obs,sim):
    """
    Calculates mean absolute error (mae) two flow arrays.
    Arguments
    ---------
    sim: array-like
        Simulated time series array.
    obs: array-like
        Observed time series array.

    Returns
    -------
    mae: float
        mean absolute error calculated between the two arrays.
    """
    mask = np.logical_and(~np.isnan(obs), ~np.isnan(sim))
    return np.mean(np.absolute((sim[mask] - obs[mask])))


def month_mean_flow_err(obs, sim, sim_time):
    obs = np.asarray(obs)
    sim = np.asarray(sim)
    sim = np.reshape(sim, np.shape(obs))

    #month = [dt.month for dt in sim_time]
    month = sim_time.dt.month

    data = {'sim':sim, 'obs':obs, 'month':month}
    #df = pd.DataFrame(data, index = sim_time)
    df = pd.DataFrame(data)

    gdf = df.groupby(['month'])
    sim_month_mean = gdf.aggregate({'sim':np.nanmean})
    obs_month_mean = gdf.aggregate({'obs':np.nanmean})

    mth_err = np.nanmean(np.absolute(obs_month_mean['obs'] - sim_month_mean['sim']))
    return mth_err


# flow metrics

def FHV(dr: xr.DataArray, calendar="standard", percent=0.9):
    """
    Calculates Flow duration curve high segment volume.
    Arguments
    ---------
    dr: xr.DataArray
        2D DataArray containing daily time series with coordinates of 'site', and 'time'
    Returns
    -------
    ds_FLV: xr.Dataset
        Dataset containing two 2D DataArrays 'ann_max_flow' and 'ann_max_day' with coordinate of 'year', and 'site'
    Notes
    -------
    None
    """
    prob=np.arange(1,float(len(dr['time']+1)))/(1+len(dr['time'])) #probability
    for d in range(len(prob)):
        idx=d
        if prob[d] > percent: break

    t_axis = dr.dims.index('time')
    flow_array_sort = np.sort(dr.values, axis=t_axis)
    if t_axis==0:
        FHV = np.sum(flow_array_sort[idx:,:], axis=t_axis)
    elif t_axis==1:
        FHV = np.sum(flow_array_sort[:,idx:], axis=t_axis)

    ds_FHV = xr.Dataset(data_vars=dict(FHV=(["site"], FHV)), coords=dict(site=dr['site']))

    return ds_FHV


def FLV(dr: xr.DataArray, calendar="standard", percent=0.1):

    #Calculates Flow duration curve low segment volume. default is < 0.1
    # Yilmaz, K. K., et al. (2008), A process-based diagnostic approach to model evaluation: Applicationto the NWS distributed hydrologic model, Water Resour. Res., 44, W09417, doi:10.1029/2007WR006716

    prob=np.arange(1,float(len(dr['time']+1)))/(1+len(dr['time'])) #probability
    for d in range(len(prob)):
        idx=d
        if prob[d] > percent: break

    t_axis = dr.dims.index('time')
    flow_array_sort = np.sort(dr.values, axis=t_axis)
    if t_axis==0:
        FLV = np.sum(flow_array_sort[:idx,:], axis=t_axis)
    elif t_axis==1:
        FLV = np.sum(flow_array_sort[:,:idx], axis=t_axis)

    ds_FLV = xr.Dataset(data_vars=dict(FLV=(["site"], FLV)), coords=dict(site=dr['site']))

    return ds_FLV


def FMS(dr: xr.DataArray, calendar="standard", percent_low=0.3, percent_high=0.7):

    #Calculate Flow duration curve midsegment slope (default between 30 and 70 percentile)

    prob=np.arange(1,float(len(dr['time']+1)))/(1+len(dr['time'])) #probability
    for d in range(len(prob)):
        idx_l=d
        if prob[d] > percent_low: break
    for d in range(len(prob)):
        idx_h=d
        if prob[d] > percent_high: break

    t_axis = dr.dims.index('time')
    flow_array_sort = np.sort(dr.values, axis=t_axis)
    if t_axis==0:
        FMS = (np.log10(flow_array_sort[idx_h,:]) - np.log10(flow_array_sort[idx_l,:]))/(percent_high-percent_low)
    elif t_axis==1:
        FMS = (np.log10(flow_array_sort[:,idx_h]) - np.log10(flow_array_sort[:,idx_l]))/(percent_high-percent_low)

    ds_FMS = xr.Dataset(data_vars=dict(FMS=(["site"], FMS)), coords=dict(site=dr['site']))

    return ds_FMS


def BFI(dr: xr.DataArray, alpha=0.925, npass=3, skip_time=30):

    #Calculate digital filter based Baseflow Index
    # Ladson, A. R., et al. (2013). A Standard Approach to Baseflow Separation Using The Lyne and Hollick Filter. Australasian Journal of Water Resources, 17(1), 25–34.
    # https://doi.org/10.7158/13241583.2013.11465417

    t_axis = dr.dims.index('time')
    tlen = len(dr['time'])

    q_total = dr.values
    if t_axis==1:
        q_total = q_total.T

    q_total_diff = q_total - np.roll(q_total, 1, axis=t_axis) # q(i)-q(i-1)
    q_fast = np.tile(q_total[skip_time+1,:], (tlen,1))

    count=1
    while count <= npass:
        for tix in np.arange(1, tlen):
            q_fast[tix,:] = alpha*q_fast[tix-1,:]+(1.0+alpha)/2.0*q_total_diff[tix,:]
            q_fast[tix,:] = np.where(q_fast[tix,:]>=0, q_fast[tix,:], 0)
        count+=1

    q_base = q_total-q_fast

    BFI = np.sum(q_base[skip_time:,:])/np.sum(q_total[skip_time:,:])

    ds_BFI = xr.Dataset(data_vars=dict(BFI=(["site"], BFI)), coords=dict(site=dr['site']))

    return BFI


def high_q_freq(dr: xr.DataArray, percent_high=0.7):
    # frequency of high-flow days (> 9 times the median daily flow) day/yr
    pass


def high_q_dur(dr: xr.DataArray, percent_high=0.7):
    #  average duration of high-flow events (number of consecutive days > 9 times the median daily flow) day
    pass


def low_q_freq(dr: xr.DataArray, percent_high=0.7):
    #  frequency of low-flow days (< 0.2 times the mean daily flow) day/yr
    pass


def low_q_dur(dr: xr.DataArray, percent_high=0.7):
    # average duration of low-flow events (number of consecutive days < 0.2 times the mean daily flow) day
    pass


def annual_max(dr: xr.DataArray, dayofyear='wateryear'):
    """
    Calculates annual maximum value and dayofyear.
    Arguments
    ---------
    dr: xr.DataArray
        2D DataArray containing daily time series with coordinates of 'site', and 'time'
    Returns
    -------
    ds_ann_max: xr.Dataset
        Dataset containing two 2D DataArrays 'ann_max_flow' and 'ann_max_day' with coordinate of 'year', and 'site'
    Notes
    -------
    dayofyear start with October 1st with dayofyear="wateryear" or January 1st with dayofyear="calendar".
    """
    if dayofyear=='wateryear':
        smon=10; sday=1; emon=9; eday=30; yr_adj=1
    elif dayofyear=='calendar':
        smon=1; sday=1; emon=12; eday=31; yr_adj=0
    else:
        raise ValueError('Invalid argument for "dayofyear"')

    years = np.unique(dr.time.dt.year.values)[:-1]

    ds_ann_max = xr.Dataset(data_vars=dict(
                    ann_max_flow =(["year", "site"], np.full((len(years),len(dr['site'])), np.nan, dtype='float32')),
                    ann_max_day  =(["year", "site"], np.full((len(years),len(dr['site'])), np.nan, dtype='float32')),
                    ),
                    coords=dict(year=years,
                                site=dr['site'],),
                    )

    for yr in years:
        time_slice=slice(f'{yr}-{smon}-{sday}',f'{yr+yr_adj}-{emon}-{eday}')

        max_flow_array = dr.sel(time=time_slice).max(dim='time').values # find annual max flow
        ix = np.argwhere(np.isnan(max_flow_array)) # if whole year data is missing, it is nan, so find that site
        max_day_array = dr.sel(time=time_slice).argmax(dim='time', skipna=False).values.astype('float') # find annual max flow day
        max_day_array[ix]=np.nan

        ds_ann_max['ann_max_flow'].loc[dict(year=yr)] = max_flow_array
        ds_ann_max['ann_max_day'].loc[dict(year=yr)]  = max_day_array

    return ds_ann_max


def annual_min(dr: xr.DataArray, dayofyear='wateryear'):
    """
    Calculates annual minimum value and dayofyear.
    Arguments
    ---------
    dr: xr.DataArray
        2D DataArray containing daily time series with coordinates of 'site', and 'time'
    Returns
    -------
    ds_ann_max: xr.Dataset
        Dataset containing two 2D DataArrays 'ann_min_flow' and 'ann_min_day' with coordinate of 'year', and 'site'
    """
    if dayofyear=='wateryear':
        smon=10; sday=1; emon=9; eday=30; yr_adj=1
    elif dayofyear=='calendar':
        smon=1; sday=1; emon=12; eday=31; yr_adj=0
    else:
        raise ValueError('Invalid argument for "dayofyear"')

    years = np.unique(dr.time.dt.year.values)[:-1]

    ds_ann_min = xr.Dataset(data_vars=dict(
                    ann_min_flow =(["year", "site"], np.full((len(years),len(dr['site'])), np.nan, dtype='float32')),
                    ann_min_day  =(["year", "site"], np.full((len(years),len(dr['site'])), np.nan, dtype='float32')),
                    ),
                    coords=dict(year=years,
                                site=dr['site'],)
                    )

    for yr in years:
        time_slice=slice(f'{yr}-{smon}-{sday}',f'{yr+yr_adj}-{emon}-{eday}')
        min_flow_array = dr.sel(time=time_slice).min(dim='time').values
        ix = np.argwhere(np.isnan(min_flow_array)) # if whole year data is missing, it is nan, so find that site
        min_day_array = dr.sel(time=time_slice).argmin(dim='time', skipna=False).values.astype('float') # find annual max flow day
        min_day_array[ix]=np.nan
        ds_ann_min['ann_min_flow'].loc[dict(year=yr)] = min_flow_array
        ds_ann_min['ann_min_day'].loc[dict(year=yr)] = min_day_array

    return ds_ann_min


def annual_centroid(dr: xr.DataArray, dayofyear='wateryear'):
    """
    Calculates annual time series centroid (in dayofyear).
    Arguments
    ---------
    dr: xr.DataArray
        2D DataArray containing daily time series with coordinates of 'site', and 'time'
    Returns
    -------
    ds_ann_max: xr.Dataset
        Dataset containing one 2D DataArrays 'ann_centroid_day' with coordinate of 'year', and 'site'
    """
    if dayofyear=='wateryear':
        smon=10; sday=1; emon=9; eday=30; yr_adj=1
    elif dayofyear=='calendar':
        smon=1; sday=1; emon=12; eday=31; yr_adj=0
    else:
        raise ValueError('Invalid argument for "dayofyear"')

    years = np.unique(dr.time.dt.year.values)[:-1]

    ds_ann_centroid = xr.Dataset(data_vars=dict(
                                ann_centroid_day = (["year", "site"], np.full((len(years),len(dr['site'])), np.nan, dtype='float32')),
                                ),
                                coords=dict(year=years, site=dr['site'],)
                                )

    for ix, yr in enumerate(years):
        time_slice=slice(f'{yr}-{smon}-{sday}',f'{yr+yr_adj}-{emon}-{eday}')
        for site in dr['site'].values:
            q_array = dr.sel(time=time_slice, site=site).values
            centroid_day = (q_array*np.arange(len(q_array))).sum()/q_array.sum()
            ds_ann_centroid['ann_centroid_day'].loc[dict(year=yr, site=site)] = centroid_day

    return ds_ann_centroid


def season_mean(ds: xr.Dataset, calendar="standard"):
    # Make a DataArray with the number of days in each month, size = len(time)
    month_length = ds.time.dt.days_in_month

    # Calculate the weights by grouping by 'time.season'
    weights = (month_length.groupby("time.season") / month_length.groupby("time.season").sum())

    # Test that the sum of the weights for each season is 1.0
    np.testing.assert_allclose(weights.groupby("time.season").sum().values, np.ones(4))

    # Calculate the weighted average
    return (ds * weights).groupby("time.season").sum(dim="time")
