# import pickle
import numpy as np
from torch.utils.data import Dataset
import torch
import os
import xarray as xr
import pandas as pd
from random import shuffle
import shutil
from tqdm import tqdm
from time import time
import datetime
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    except:
        pass
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except:
        pass
# Set the seed for reproducibility
seed = 1234
set_seed(seed)

    


def _select_nino34(file_data):
    """
    Niño 3.4 box: 5°N-5°S, 170°W-120°W = 190°E-240°E.
    Assumes longitudes are in [0, 360).
    """
    lat_mask = (file_data.latitude >= -5) & (file_data.latitude <= 5)
    lon_mask = (file_data.longitude >= 190) & (file_data.longitude <= 240)
    return file_data.sel(latitude=lat_mask, longitude=lon_mask)


def _monthly_climatology(data, baseline=(1991, 2020)):
    """
    Month-of-year climatology over the baseline (inclusive).
    Pass a monthly DataArray with 'time' coord.
    """
    start, end = baseline
    da_base = data.sel(time=slice(f"{start}-01-01", f"{end}-12-31"))
    # return da_base.groupby("time.month").mean("time", skipna=True)
    return da_base.groupby("time.month").mean(dim=["time", "longitude", "latitude"], skipna=True)



def _coslat_weights(da):
    """cos(latitude) weights, broadcast to da, with NaNs masked."""
    # print('hadisst latitudes: ', da["latitude"])
    w_lat = np.cos(np.deg2rad(da["latitude"]))
    w = w_lat
    # Broadcast weights to data shape for lat/lon mean
    for dim in da.dims:
        if dim != "latitude" and dim in ["longitude"]:
            w = w.broadcast_like(da)
            break
    # Mask weights where data is NaN
    return w.where(da.notnull())

def _area_weighted_mean(da):
    """Area-weighted mean over lat/lon using cos(lat) or provided cell areas."""
    w = _coslat_weights(da)
    # print('area weights: ', w.SST.values)
    num = (da * w).sum(dim=["latitude", "longitude"], skipna=True)
    den = w.sum(dim=["latitude", "longitude"], skipna=True)
    # print('area weights sum: ', den)
    return num / den



def compute_nino34(file_data, baseline=(1991, 2020), climatology=None):
    """
    Returns:
      nino34: DataArray of monthly Niño-3.4 anomalies (°C)
      oni:    DataArray of 3-month running mean of Niño-3.4 (ONI, °C)
    """

    # Subset to Niño 3.4 box (after normalizing longitudes)
    sst_box = _select_nino34(file_data) #(1860, 10, 50)
    # print('sst_box: ', sst_box.SST.values.shape)
    # print('sst_box: ', sst_box.SST.values)
    
    
    if climatology is None:
        # Build month-specific baseline climatology per grid cell
        climatology = _monthly_climatology(sst_box, baseline=baseline)  # dims: (month, lat, lon): (12, 10, 50)
    # print('climatology: ', climatology)
    # Anomalies (subtract month-of-year climatology)
    anom = sst_box.groupby("time.month") - climatology #(1860, 10, 50)
    # print('anom: ', anom.SST.values.shape)
    # print('anom: ', anom.SST.values)
    
    # Area-weighted spatial mean over the box
    nino34 = _area_weighted_mean(anom)
    nino34 = nino34.SST.rename("nino34").assign_attrs({
        "long_name": "Niño 3.4 SST anomaly",
        "units": "degC",
        "region": "5N-5S, 170W-120W",
        "baseline": f"{baseline[0]}-{baseline[1]}",
        "note": "cos(lat)-weighted mean of monthly SST anomalies",
    })
    # ONI: 3-month running mean of Niño 3.4
    # CPC labels seasons by the center month (e.g., DJF -> Jan), which corresponds to center=True.
    oni = nino34.rolling(time=3, center=True).mean().rename("oni").assign_attrs({
        "long_name": "Oceanic Niño Index (3-month running mean of Niño 3.4)",
        "units": "degC",
        "season_labeling": "season assigned to center month (e.g., DJF->Jan)",
    })
    return nino34.values, oni.values, climatology #(1860,), (1860,), (12, 10, 50)




def positional_encoding(position, d_model, shift=0.3):

    pos_vector = torch.zeros((len(position), d_model))

    # Compute the positional encodings
    for i in range(d_model):
        if i % 2 == 0:
            # Apply the sin transformation for even indices
            sin_freq = (i//2)*torch.pi/(6.0)
            pos_vector[:,i] = torch.sin((position+shift) * sin_freq)
        else:
            # Apply the cos transformation for odd indices
            cos_freq = ((i+1)//2)*torch.pi/(6.0)
            pos_vector[:,i] = torch.cos((position+shift) * cos_freq)
    return pos_vector

class ClimateDataset:
    '''Loads data files in a dataframe df'''
    
    def __init__(self, data_dir, mode='train', sin_embed=True, 
                 time_dim=200, dtype='sst', minim=None, maxim=None, start_idx=0,
                 lat_range=(-50, 70), lon_range=(0, 360),
                 sst_window_size=36, sst_window_step=12, flatten_input=False):
        self.data_dir = '/home/mmotame1/climate-research/DVAE/dvae/dataset/HadISST'
        self.mode = mode
        self.dtype = dtype
        self.sin_embed = sin_embed
        self.time_dim = time_dim
        self.minim, self.maxim = minim, maxim
        self.months, self.years, self.time_encodings = None, None, None
        self.sst, self.mask = None, None
        self.sst_window_size = sst_window_size
        self.sst_window_step = sst_window_step
        self.sst_window_overlap = self.sst_window_size - self.sst_window_step
        self.start_idx = start_idx
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.flatten_input = flatten_input
        self.data_type = 'sst'
        self.data_paths = {
            'sst': '/home/mmotame1/climate-research/DVAE/dvae/dataset/HadISST/SST_1870-2024.nc',
            'ohc': '/home/mmotame1/climate-research/DVAE/dvae/dataset/multi_channel_data/ohc_xarray.nc', 
            'u925': '/home/mmotame1/climate-research/DVAE/dvae/dataset/multi_channel_data/U925_75_xarray.nc',
            'multi_channel': '/home/mmotame1/climate-research/DVAE/dvae/dataset/multi_channel_data/mix_data_xarray.nc'
            }
        print('sst_window_size: ', self.sst_window_size)
        print('sst_window_overlap: ', self.sst_window_overlap)
        # self.read_xr(os.path.join(self.data_dir, 'SST_1870-2024.nc'))
        self.read_xr(self.data_paths[self.data_type])
        print(f'{self.data_type} data path: \n{self.data_paths[self.data_type]}')
        

    def correct_time(self, file_data):
        time_values = file_data.time.values
        if self.data_type == 'sst':
            start_date = pd.Timestamp('1870-01-01')  # Start from January 1st
        elif self.data_type in ['ohc', 'u925']:
            start_date = pd.Timestamp('1955-02-01')  # Start from January 1st
        else:
            raise ValueError(f'Invalid data type: {self.data_type}')
        time_converted = pd.date_range(
            start=start_date,
            periods=len(time_values),
            freq='MS'  # Month start
        ) + pd.Timedelta(days=14)  # Adjust to 15th of each month

        # Assign the converted time back to the dataset
        file_data = file_data.assign_coords(time=time_converted)
        return time_converted, file_data
    
    def _unfold(self, data):
        return data.unfold(0, self.sst_window_size, self.sst_window_overlap)
        
    def read_xr(self, path):
        '''Read xarray file'''
        print('Reading xarray file...')
        print(path)
        file_data = xr.load_dataset(path, decode_times=False)
        time_converted, file_data = self.correct_time(file_data)
        
        nino34, oni, self.climatology = compute_nino34(file_data, baseline=(1991, 2020)) #(1860,), (1860,), (12, 10, 50)
        
        # self.sst = self.sst[:, 20:-40] # Remove constant values in north pole
        # file_data_cropped = file_data.sel(latitude=slice(-50, 70))
        file_data_cropped = file_data.sel(latitude=slice(self.lat_range[0], self.lat_range[1]), 
                                          longitude=slice(self.lon_range[0], self.lon_range[1]))
        
        # mask_lon = (file_data_cropped.longitude >= 190) & (file_data_cropped.longitude <= 240)
        # mask_lat = (file_data_cropped.latitude >= -5) & (file_data_cropped.latitude <= 5)
        # mask_2d = np.outer(mask_lat, mask_lon)
        # mask_2d = np.flip(mask_2d, axis=0).astype(int)
        if self.data_type == 'sst':
            sst = np.flip(file_data_cropped.SST.values, axis=1)
        elif self.data_type in ['ohc', 'u925']:
            sst = (file_data_cropped.SST.values)
        else:
            raise ValueError(f'Invalid data type: {self.data_type}')
        # print('mask_2d: ', mask_2d.shape)
        # print('sst: ', sst.shape)
        # print('nino34: ', nino34.shape)
        # print('oni: ', oni.shape)
        # print('-'*100)
        self.sst, self.nino34, self.oni, self.time = self.get_data_split(sst, nino34, oni, time_converted)
        # print('self.sst: ', self.sst.shape)
        # print('self.nino34: ', self.nino34.shape)
        # print('self.oni: ', self.oni.shape)
        # print('self.time: ', self.time.shape)
        # print('-'*100)
        self.sst = self.sst[self.start_idx:]
        self.nino34 = self.nino34[self.start_idx:]
        self.oni = self.oni[self.start_idx:]
        self.time = self.time[self.start_idx:]
        # print('self.sst: ', self.sst.shape)
        # print('self.nino34: ', self.nino34.shape)
        # print('self.oni: ', self.oni.shape)
        # print('self.time: ', self.time.shape)
        # print('-'*100)
        n, h, w = self.sst.shape
        print('n, h, w: ', n, h, w)
        self.mask = torch.from_numpy(~np.isnan(self.sst)).float()
        # self.mask = torch.from_numpy(mask_2d).float().expand(n, h, w)
        self.mask = self._unfold(self.mask).permute(0, 3, 1, 2)
        if self.flatten_input:
            self.mask = self.mask.reshape(-1, self.sst_window_size, h*w)
        else:
            self.mask = self.mask.reshape(-1, self.sst_window_size, h, w)
        self.sst = np.nan_to_num(self.sst, nan=0.0, posinf=0.0, neginf=0.0)
        self.min_max_normalize()
        # print('-'*100)
        # print('self.mask.shape: ', self.mask.shape)
        # print('self.sst.shape: ', self.sst.shape)
        # print('self.time.shape: ', self.time.shape)
        # print('self.time: ', self.time)
        self.years = self._unfold(torch.tensor(self.time.year).long())
        self.months = torch.tensor(self.time.month).long()
        if self.sin_embed:
            self.time_encodings = positional_encoding(self.months, d_model=self.time_dim).float()
            self.time_encodings = self._unfold(self.time_encodings).permute(0, 2, 1)
        else:
            self.time_encodings = self._unfold(self.months)
        self.months = self._unfold(self.months)
        self.sst = torch.from_numpy(self.sst).float()
        self.sst = self._unfold(self.sst).permute(0, 3, 1, 2)
        if self.flatten_input:
            self.sst = self.sst.reshape(-1, self.sst_window_size, h * w)
        else:
            self.sst = self.sst.reshape(-1, self.sst_window_size, h, w)
        self.nino34 = self._unfold(torch.from_numpy(self.nino34).float())
        self.oni = self._unfold(torch.from_numpy(self.oni).float())
        print('-'*100)
        print('self.mask.shape: ', self.mask.shape)
        print('self.sst.shape: ', self.sst.shape)
        print('self.time_encodings.shape: ', self.time_encodings.shape)
        print('self.months.shape: ', self.months.shape)
        print('self.years.shape: ', self.years.shape)
        print('self.nino34.shape: ', self.nino34.shape)
        print('self.oni.shape: ', self.oni.shape)
        print('Read xarray file successfully!')
        print('-'*100)
        return

    
    def get_data_split(self, sst, nino34, oni, time_converted):
        # split_idx = int(len(time_converted) * 0.8)
        test_start, test_end = pd.Timestamp('2002-02-01'), pd.Timestamp('2023-03-01')
        test_idx = (time_converted >= test_start) & (time_converted <= test_end)
        # test_start = pd.Timestamp('2002-02-01')
        # test_idx = (time_converted >= test_start)
        train_idx = (time_converted<test_start)
        if self.mode == 'train':
            # time_array = time_converted[:split_idx]
            # sst_array = sst[:split_idx]
            # nino34_array = nino34[:split_idx]
            # oni_array = oni[:split_idx]
            time_array = time_converted[train_idx]
            sst_array = sst[train_idx]
            nino34_array = nino34[train_idx]
            oni_array = oni[train_idx]
        elif self.mode == 'test':
            # time_array = time_converted[split_idx:]
            # sst_array = sst[split_idx:]
            # nino34_array = nino34[split_idx:]
            # oni_array = oni[split_idx:]
            time_array = time_converted[test_idx]
            sst_array = sst[test_idx]
            nino34_array = nino34[test_idx]
            oni_array = oni[test_idx]
        return sst_array, nino34_array, oni_array, time_array
        
    
    def min_max_normalize(self):
        if self.mode == 'train':
            self.minim = np.min(self.sst, axis=0)
            self.maxim = np.max(self.sst, axis=0)
            self.sst = (self.sst - self.minim) / (self.maxim - self.minim + 1e-8)
        elif self.mode == 'test':
            self.sst = (self.sst - self.minim) / (self.maxim - self.minim + 1e-8)
        return
         
          
    def __getitem__(self, index):
        return (self.sst[index], 
                self.mask[index], 
                self.time_encodings[index], 
                self.months[index],
                self.years[index],
                self.nino34[index],
                self.oni[index])
        
    def __len__(self):
        return len(self.sst)
    

def build_dataloader(cfg, save_dir=None):
    # Load dataset params
    batch_size = cfg.getint('DataFrame', 'batch_size')
    num_workers = cfg.getint('DataFrame', 'num_workers')
    saved_root = cfg.get('User', 'saved_root')
    time_dim = cfg.getint('DataFrame', 'time_dim')
    sin_embed = cfg.getboolean('DataFrame', 'sin_embed')
    dataset_name = cfg.get('DataFrame', 'dataset_name')
    dataset_feature = cfg.get('DataFrame', 'dataset_feature')
    lat_range = cfg.get('DataFrame', 'lat_range')
    lon_range = cfg.get('DataFrame', 'lon_range')
    lat_range = tuple(int(x) for x in lat_range.strip('()').split(','))
    lon_range = tuple(int(x) for x in lon_range.strip('()').split(','))
    sst_window_size = cfg.getint('DataFrame', 'sst_window_size')
    sst_window_step = cfg.getint('DataFrame', 'sst_window_step')
    start_idx = cfg.getint('DataFrame', 'start_month')
    flatten_input = cfg.getboolean('DataFrame', 'flatten_input')
    print('lat_range: ', lat_range)
    print('lon_range: ', lon_range)
    print('-'*100)
    print('start_idx: ', start_idx)
    print('-'*100)
    
    # start_idx = 10
    print('Creating Dataset...')
    train_dataset = ClimateDataset(
        None, mode='train', sin_embed=sin_embed, time_dim=time_dim, 
        dtype=dataset_feature, start_idx=start_idx,
        lat_range=lat_range, lon_range=lon_range,
        sst_window_size=sst_window_size, sst_window_step=sst_window_step, flatten_input=flatten_input)
    test_dataset = ClimateDataset(
        None, mode='test', sin_embed=sin_embed, time_dim=time_dim, 
        dtype=dataset_feature, minim=train_dataset.minim, 
        maxim=train_dataset.maxim, start_idx=start_idx,
        lat_range=lat_range, lon_range=lon_range,
        sst_window_size=sst_window_size, sst_window_step=sst_window_step, flatten_input=flatten_input)
    print(f'Saving min and max in {saved_root}...')
    np.save(os.path.join(save_dir, 'SST_min_'+dataset_name+'.npy'), train_dataset.minim)
    np.save(os.path.join(save_dir, 'SST_max_'+dataset_name+'.npy'), train_dataset.maxim)

    print('train_dataset.climatology: ', train_dataset.climatology)
    print('type(train_dataset.climatology): ', type(train_dataset.climatology))
    # Rename the SST variable to climatology and save
    clim_dataset = train_dataset.climatology
    clim_dataset.attrs["description"] = "Climatology SST dataset for years 1991-2020"
    clim_dataset.to_netcdf(os.path.join(saved_root, 'climatology.nc'))
    
    # Create dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers, 
        pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers, 
        pin_memory=True)
        
    train_num, test_num = train_dataset.__len__(), test_dataset.__len__()
    print('Dataloader created successfully!')
    print('-'*100)
    return train_dataloader, test_dataloader, train_num, test_num, start_idx

