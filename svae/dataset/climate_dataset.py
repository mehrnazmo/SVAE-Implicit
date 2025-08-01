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
set_seed(1234)

def positional_encoding(position, d_model, shift=0.3):
    # Initialize a zero vector
    if isinstance(position, (np.ndarray, pd.Series, list)):
        pos_vector = np.zeros((d_model, len(position)))
    else:
        pos_vector = np.zeros((d_model, ))
    # Compute the positional encodings
    for i in range(d_model):
        if i % 2 == 0:
            # Apply the sin transformation for even indices
            sin_freq = (i//2)*np.pi/(6.0)
            pos_vector[i] = np.sin((position+shift) * sin_freq)
        else:
            # Apply the cos transformation for odd indices
            cos_freq = ((i+1)//2)*np.pi/(6.0)
            pos_vector[i] = np.cos((position+shift) * cos_freq)
    return pos_vector.T

class ClimateDataset:
    '''Loads data files in a dataframe df'''
    
    def __init__(self, data_dir, sin_embed=True, time_series=True,
                 time_dim=200, dtype='sst', ssta=False):
        self.time_series = time_series
        self.data_dir = data_dir
        self.dtype = dtype
        self.sin_embed = sin_embed
        self.time_dim = time_dim
        self.original_data, self.latitude, self.longitude = self.load_data()
        self.months = np.array(self.original_data['M'])
        self.years = np.array(self.original_data['Y'])
        self.T = self.years*12 + self.months
        self.time_encoding = self.positional_encoding(
            self.months, d_model=self.time_dim, base=10000)
        ##Convert to Numpy Array
        if dtype=='sst':
            col = 'ssta' if ssta else 'sst'
            print(f'self.original_data[col].shape'
                  f': {self.original_data[col].shape}')
            if self.time_series:
                self.original_data['T'] = self.T
                self.original_data['sst_time_series'] = self.original_data.apply(
                    lambda row: np.stack(
                        self.original_data[
                            (self.original_data['T'] >= row['T'] - 24) & 
                            (self.original_data['T'] < row['T'])
                        ][col].values
                    ) if len(self.original_data[
                        (self.original_data['T'] >= row['T'] - 24) & 
                        (self.original_data['T'] < row['T'])
                    ][col].values) > 0 else np.zeros((24, 1, 1, 26, 90)), axis=1)
                self.original_data['sst_time_series'] = self.original_data['sst_time_series'].apply(
                    lambda row: row if row.shape[0] ==24 else np.concatenate([np.zeros((24-row.shape[0], *row.shape[1:])), row], axis=0))
                self.original_data = self.original_data[self.original_data['M'] == 10] #for rain prediction
                (self.original_data_train, self.original_data_test
                 ) = self.split_df(self.original_data, split=80, shuffle_flag=True)
                
                print('\nself.original_data_train:')
                print(self.original_data_train.columns)
                print(self.original_data_train.head())
                
                print('\nself.original_data_test:')
                print(self.original_data_test.columns)
                print(self.original_data_test.head())
                self.original_data = np.stack(self.original_data['sst_time_series'], axis=0).squeeze() #(1644, 24, 26, 90)
                self.original_data_train = np.stack(self.original_data_train['sst_time_series'], axis=0).squeeze() #(1314, 24, 26, 90)
                self.original_data_test = np.stack(self.original_data_test['sst_time_series'], axis=0).squeeze() #(330, 24, 26, 90)
                print(f'self.original_data.shape : {self.original_data.shape}')
                print(f'self.original_data_train.shape : {self.original_data_train.shape}')
                print(f'self.original_data_test.shape : {self.original_data_test.shape}')
            else:    
                self.original_data=np.stack(self.original_data[col])
                print(f'self.original_data.shape : {self.original_data.shape}')
                self.original_data = self.original_data[:,0,0,:,:]
        elif dtype=='ppt':
            self.clusters = np.load(os.path.join(os.path.dirname(os.path.dirname(self.data_dir)), "clusters.npy"))
            self.original_data["ppt_cluster_vector"] = self.original_data["ppt"].apply(
                lambda x: self.average_clusters(x, self.clusters, return_image=False)[:-1])
            q1, q2, q3 = self.get_quantiles(self.original_data, cluster=True)
            print(f'PPT Quantiles:\nq1: {q1}\nq2: {q2}\nq3: {q3}')
            self.original_data["ppt_cluster_vector_quantized"] = (
                self.original_data["ppt_cluster_vector"].apply(
                    lambda x: self.get_quantized(x, q1, q2, q3)))
            print('self.original_data:\n')
            print(self.original_data)
            self.original_data=np.stack(self.original_data['ppt_cluster_vector_quantized'])
            print(f'avg ppt cluster vector: {np.mean(self.original_data, axis=0)}')
        else:
            print(f'Data from column {dtype} is not available!')
            exit()
            
        self.mask = self.get_mask()
        
    def split_df(self, df, split=80, shuffle_flag=True):
        assert split <= 100
        assert split >= 0  
        years_copy = df['Y'].unique().copy()
        if shuffle_flag:
            np.random.shuffle(years_copy)
        split_idx = int(len(years_copy) * split / 100.0)
        self.train_years = years_copy[:split_idx]
        self.test_years = years_copy[split_idx:]
        train_df = df[df['Y'].isin(self.train_years)]
        test_df = df[df['Y'].isin(self.test_years)]
        return train_df, test_df
        
    def one_hot(self, x):
        x = x.astype(int)
        one_hot = np.zeros((x.size, x.max()+1))
        one_hot[np.arange(x.size),x] = 1
        return one_hot
    
    def _init_positional_encoding(self, position, d_model, base=10000):
        # Initialize a zero vector
        if isinstance(position, (np.ndarray, pd.Series, list)):
            pos_vector = np.zeros((d_model, len(position)))
        else:
            pos_vector = np.zeros((d_model, ))
        # Compute the positional encodings
        position_cyclic = 2 * np.pi * position / 12.0 #make it cyclic
        for i in range(d_model):
            if i % 2 == 0:
                # Apply the sin transformation for even indices
                pos_vector[i] = np.sin(position_cyclic / (base ** (i / d_model)))
            else:
                # Apply the cos transformation for odd indices
                pos_vector[i] = np.cos(position_cyclic / (
                    base ** ((i - 1) / d_model)))
        return pos_vector.T
    
    def positional_encoding(self, position, d_model, base=10000, shift=0.3):
        return positional_encoding(position, d_model, shift)
             
    def get_filenames(self, directory):
        return sorted(os.listdir(directory))

    def get_path(self, directory, filename):
        return os.path.join(directory, filename)

    def get_basename(self, path):
        return os.path.basename(path)

    def get_month_year(self, file_name):
        '''Get month and year from file name'''
        if self.dtype=='ppt':
            time = pd.Timestamp('-'.join(
                file_name.split('.')[0].split('_')[1:]))
            month, year = time.month, time.year
        elif self.dtype=='sst':
            year, month = (file_name.split('.')[-2][-6:-2], 
                           file_name.split('.')[-2][-2:])
            month, year = int(month), int(year)
        else:
            print(f'dtype {self.dtype} not implemented. Use ppt or sst.')
            month, year = None, None
        return month, year
    
    def average_clusters(self, image, cluster_array, return_image=True):
        """Average the values of each ppt cluster in the image"""
        unique_clusters = np.unique(cluster_array)  # Find unique cluster labels

        if return_image:
            # Initialize an array to store the average values
            averaged_image = np.zeros_like(image, dtype=float)
        else:
            cluster_averages = []

        for cluster_label in unique_clusters:
            # Mask the original image with the current cluster label
            masked_image = np.where(cluster_array == cluster_label, image, 0)

            # Calculate the average value for the current cluster
            cluster_size = np.sum(cluster_array == cluster_label)
            cluster_average = np.sum(masked_image) / cluster_size

            if return_image:
                # Replace pixels in the averaged image with the cluster average
                averaged_image += np.where(
                    cluster_array == cluster_label, cluster_average, 0
                )
            else:
                cluster_averages.append(cluster_average)

        return averaged_image if return_image else np.array(cluster_averages)

    def get_quantiles(self, df, cluster=True):

        vector = df.ppt_cluster_vector if cluster else df.ppt

        ppt_vector = np.stack(vector.apply(lambda x: x.reshape(-1))).T
        q1 = np.quantile(ppt_vector, 0.25, axis=1, keepdims=True).flatten()
        q2 = np.quantile(ppt_vector, 0.5, axis=1, keepdims=True).flatten()
        q3 = np.quantile(ppt_vector, 0.75, axis=1, keepdims=True).flatten()
        return q1, q2, q3
    
    def get_quantized(self, x, q1, q2, q3):
        output = np.zeros_like(x)
        output[x <= q1] = 1
        q2_arr = np.array(x > q1) * np.array(x <= q2)
        q3_arr = np.array(x > q2) * np.array(x <= q3)
        output[q2_arr] = 2
        output[q3_arr] = 3
        output[x > q3] = 4
        return output

    def read_ppt(self, file_data, min_lon=0, max_lon=150, 
                 coarsen_lat=4, coarsen_lon=4):
        file_data = file_data.rename_vars({list(file_data.keys())[1] : 'ppt', 
                                        'longitude':'lon', 'latitude':'lat'})
        
        #Coarsen data to reduce resolution
        file_data = file_data.coarsen(latitude=coarsen_lat, 
                                      longitude=coarsen_lon, 
                                      boundary = "trim").mean()
        # file_data = file_data.sel(longitude=slice(-300, -100))
        ##Slice for west coast
        file_data = file_data.sel(longitude=slice(min_lon, max_lon))
        # xr_data.append()
        columns = list('FYMD')
        columns[-1] = 'ppt'   
        return file_data.ppt.values, columns 
    
    def read_sst(self, file_data):
            # Coarsen data to reduce resolution
            file_data = file_data.coarsen(
                lat = 2, lon = 2, boundary = "trim").mean()
            ##Slice for west coast
            # file_data = file_data.sel(lon=slice(100, 400))
            ## Original slice
            # vals = [np.flip(file_data.sst.values, axis=2)[:,:,3:39,:], 
            #         np.flip(file_data.ssta.values, axis=2)[:,:,3:39,:]]
            vals = [np.flip(file_data.sst.values, axis=2)[:,:,6:32,:], 
                    np.flip(file_data.ssta.values, axis=2)[:,:,6:32,:]]
            ## Might be useful for future
            # vals = [np.flip(file_data.sst.values, axis=2)[:,:,6:36,:], 
            #         np.flip(file_data.ssta.values, axis=2)[:,:,6:36,:]]
            columns = list('FYMDA')
            columns[-2], columns[-1] = 'sst', 'ssta'
            return vals, columns
        
    def read_xr(self, path, min_lon=0, max_lon=150, 
                coarsen_lat=4, coarsen_lon=4):
        '''Read xarray file'''
        file_data = xr.load_dataset(path)
        file_name = self.get_basename(path)
        month, year = self.get_month_year(file_name)
        xr_data = [file_name, year, month]
        
        if self.dtype=='ppt':
            vals, columns = self.read_ppt(
                file_data, min_lon=0, max_lon=150, 
                coarsen_lat=4, coarsen_lon=4)
            longitude, latitude = file_data.longitude.values, file_data.latitude.values
            xr_data.append(vals)
        elif self.dtype=='sst':
            vals, columns = self.read_sst(file_data)
            xr_data.extend(vals) 
            longitude, latitude = file_data.lon.values, file_data.lat.values
        else:
            print(f'dtype {self.dtype} not implemented. Use ppt or sst.')
        return xr_data, longitude, latitude, columns

    def load_data(self):
        '''Load directory files in a pandas dataframe. 
        Assumes all data has the same longitude and latitude values'''
        filenames = self.get_filenames(self.data_dir)
        data_info_list = []
        for i, file_name in tqdm(enumerate(filenames)):
            try:
                path = self.get_path(self.data_dir, file_name)
                if path.endswith('.nc'):
                    xr_data, longitude, latitude, columns = self.read_xr(path)
                    data_info_list.append(xr_data)
                else:
                    print(f'File {file_name} is not a netCDF file.')
                    continue
            except Exception as e:
                print(f"Error loading file {file_name}: {e}")
                continue
        df = pd.DataFrame([p for p in data_info_list], columns=columns)
        return df, latitude, longitude

    def normalize(self, avg=None, std=None, per_pixel=False, threshold=1):
        mask = np.broadcast_to(1 - self.mask, 
                               self.original_data.shape).astype(bool)
        masked_array = np.ma.masked_array(self.original_data, mask=mask)
        if avg is None:
            if per_pixel:
                avg = np.mean(masked_array, axis=0, keepdims=True)
            else:
                avg = np.mean(masked_array, keepdims=True)
        x0 = masked_array - avg
        if std is None:
            if per_pixel:
                std = np.std(x0, axis=0, keepdims=True)
            else:
                std = np.std(x0, keepdims=True)
        self.original_data = (x0 / (std+1e-6))    
        return avg, std
    
    def min_max(self, original_data, minim=None, maxim=None,
                per_pixel=True, minmax_range=2):
        mask = np.broadcast_to(1 - self.mask, 
                               original_data.shape).astype(bool)
        masked_array = np.ma.masked_array(original_data, mask=mask)
        if minim is None and maxim is None:
            if per_pixel:
                if self.time_series:
                    minim = np.min(masked_array, axis=(0, 1), keepdims=True)
                    maxim = np.max(masked_array, axis=(0, 1), keepdims=True)
                else:
                    minim = np.min(masked_array, axis=0, keepdims=True)
                    maxim = np.max(masked_array, axis=0, keepdims=True)
            else:
                minim = np.min(masked_array, keepdims=True)
                maxim = np.max(masked_array, keepdims=True)
        R = maxim - minim
        original_data = (masked_array - minim) / R 
        if minmax_range == 2:
            original_data = 2.0 * original_data - 1.0
        elif minmax_range != 1:
            print(f'Invalid minmax_range value: {minmax_range}. Pick 1 or 2.')
            exit()
        return minim, maxim, original_data
    
    def min_max_normalize(self, minim=None, maxim=None,
                          per_pixel=True, minmax_range=2):
        if self.time_series:
            minim, maxim, original_data_train = self.min_max(
                self.original_data_train, None, None, per_pixel, minmax_range)
            _, _, original_data_test = self.min_max(
                self.original_data_test, minim, maxim, per_pixel, minmax_range)
            self.original_data_train = original_data_train
            self.original_data_test = original_data_test
        else:
            minim, maxim, original_data = self.min_max(
                self.original_data, minim, maxim, per_pixel, minmax_range)
            self.original_data = original_data
        return minim, maxim
        
    
    def percentile_normalize(self, lower, upper, median=None, zero_center=True, 
                             lower_percentile=25, upper_percentile=75):
        mask = np.broadcast_to(1 - self.mask, 
                               self.original_data.shape).astype(bool)
        masked_array = np.ma.masked_array(self.original_data, mask=mask)
        if lower is None and upper is None:
            lower = np.percentile(masked_array, lower_percentile, axis=0, keepdims=True)
            upper = np.percentile(masked_array, upper_percentile, axis=0, keepdims=True)
        if median is None:
            median = np.median(masked_array, axis=0, keepdims=True)
        R = upper - lower
        if zero_center:
            self.original_data = (masked_array - median) / (R + 1e-6)
        else:
            self.original_data = (masked_array - lower) / (R + 1e-6)
        return lower, upper, median
    
    def get_mask(self, index_example=0):
        out = self.original_data[index_example]
        mask = ~np.isnan(out)
        mask = torch.from_numpy(mask).float()
        return mask
    
    def get_time_encoding(self, index_example):
        if self.sin_embed:
            time_encoding_arr = torch.from_numpy(
                self.time_encoding[index_example]).float()
        else:
            time_encoding_arr = torch.from_numpy(
                self.one_hot_months[index_example]).float()
        return time_encoding_arr
                
          
    def __getitem__(self, index):
        out = self.original_data[index]
        # Replace NaN values
        out = np.nan_to_num(out, nan=0.0)
        out = torch.from_numpy(out).float()
        time_encoding = self.get_time_encoding(index)
        if self.dtype=='sst':
            # data = torch.unsqueeze(out, dim=0)
            # mask = torch.unsqueeze(self.mask, dim=0)
            data = out
            mask = self.mask
        else:
            data = out - 1.0 #zeros index (0,1,2,3)
            mask = torch.ones_like(out)
        return (data, 
                mask, 
                time_encoding, 
                self.months[index], 
                self.years[index])
                
        # return (torch.unsqueeze(out.flatten(), dim=0), 
        #         torch.unsqueeze(self.mask.flatten(), dim=0), 
        #         time_encoding, self.months[index])
        
    def __len__(self):
        return len(self.original_data)
    
def get_train_test_dirs(directory, shuffle_flag):
    dir_ext = '_shuffle' if shuffle_flag else ''
    train_dir = os.path.join(directory, 'train'+dir_ext)
    test_dir = os.path.join(directory, 'test'+dir_ext)    
    return train_dir, test_dir

def split_data(directory, split=80, shuffle_flag=False, overwrite=True):
    assert split <= 100
    assert split >= 0
    filenames = sorted(os.listdir(directory))
    # shuffle_idx = np.random.choice(np.arange(1880, 2017), 27, replace=False)
    # print('shuffle_idx: ', shuffle_idx)
    # #ersst.v5.188006.nc
    # train_files, test_files = [], []
    # for f in filenames:
    #     if f.endswith('.nc'):
    #         year = int(f.split('.')[-2][-6:-2])
    #         if year in shuffle_idx:
    #             test_files.append(f)
    #         else:
    #             train_files.append(f)
    # #Shuffle dataframe rows
    # if shuffle_flag:
    #     shuffle(filenames)
    # num_data = len(filenames)
    # split_idx = int(num_data * split / 100.0)
    # print('split_idx: ', split_idx)
    # ## Copy data into train and test directories
    # train_dir, test_dir = get_train_test_dirs(directory, shuffle_flag)
    # os.makedirs(train_dir, exist_ok=True)
    # os.makedirs(test_dir, exist_ok=True)
    # for base_f in train_files:
    #     if os.path.isfile(os.path.join(directory, base_f)):
    #         shutil.copy(os.path.join(directory, base_f), 
    #                     os.path.join(train_dir, base_f))
    # for base_f in test_files:
    #     if os.path.isfile(os.path.join(directory, base_f)):
    #         shutil.copy(os.path.join(directory, base_f), 
    #                     os.path.join(test_dir, base_f))
    # return train_dir, test_dir    
    
    #Shuffle dataframe rows
    if shuffle_flag:
        shuffle(filenames)
    num_data = len(filenames)
    split_idx = int(num_data * split / 100.0)
    print('split_idx: ', split_idx)
    ## Copy data into train and test directories
    train_dir, test_dir = get_train_test_dirs(directory, shuffle_flag)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    # # if overwrite:
    # #     for f in os.listdir(train_dir):
    # #         os.remove(f)
    # #     for f in os.listdir(test_dir):
    # #         os.remove(f)
    for base_f in filenames[:split_idx]:
        if os.path.isfile(os.path.join(directory, base_f)):
            shutil.copy(os.path.join(directory, base_f), 
                        os.path.join(train_dir, base_f))
    for base_f in filenames[split_idx:]:
        if os.path.isfile(os.path.join(directory, base_f)):
            shutil.copy(os.path.join(directory, base_f), 
                        os.path.join(test_dir, base_f))
    return train_dir, test_dir
    
class TimeSeriesDataset():
    def __init__(self, data_arr, years, mask, time_dim=23):
        self.data_arr = data_arr
        self.years = years
        self.mask = mask
        self.time_dim = time_dim
        self.embeddings = self.get_positional_encoding()
        
    def get_positional_encoding(self):
        # months = np.arange(1, 13) # 12 months
        # embeddings = torch.from_numpy(
        #     positional_encoding(months, self.time_dim)
        #     ).reshape(1, 12, -1).repeat(1, 2, 1)
        
        months = np.array([10, 11, 12, 
                           1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
                           1, 2, 3, 4, 5, 6, 7, 8, 9])
        embeddings = torch.from_numpy(positional_encoding(months, self.time_dim))  
        return embeddings
        
    
    def __len__(self):
        return len(self.data_arr)

    def __getitem__(self, index):
        out = self.data_arr[index].reshape(-1, 24, 26*90).squeeze()
        out = torch.from_numpy(np.nan_to_num(out, nan=0.0)).float()
        return (out, self.mask.reshape(-1, 24, 26*90), 
                self.embeddings.squeeze(), 
                self.years[index])

def build_dataloader(cfg, save_dir=None):
    # Load dataset params
    data_dir = cfg.get('User', 'data_dir')
    batch_size = cfg.getint('DataFrame', 'batch_size')
    num_workers = cfg.getint('DataFrame', 'num_workers')
    split = cfg.getboolean('DataFrame', 'split')
    split_percent = cfg.getint('DataFrame', 'split_percent') 
    shuffle_flag = cfg.getboolean('DataFrame', 'shuffle')
    overwrite = cfg.getboolean('DataFrame', 'overwrite')
    saved_root = cfg.get('User', 'saved_root')
    normalize_per_pixel = cfg.getboolean('DataFrame', 'normalize_per_pixel')
    time_dim = cfg.getint('DataFrame', 'time_dim')
    sin_embed = cfg.getboolean('DataFrame', 'sin_embed')
    dataset_name = cfg.get('DataFrame', 'dataset_name')
    dataset_feature = cfg.get('DataFrame', 'dataset_feature')
    normalization_type = cfg.get('DataFrame', 'normalization_type')
    minmax_range = cfg.getint('DataFrame', 'minmax_range')
    zero_center = cfg.getboolean('DataFrame', 'percentile_zero_center')
    lower_percentile = cfg.getint('DataFrame', 'lower_percentile')
    upper_percentile = cfg.getint('DataFrame', 'upper_percentile')
    
    #split data
    if split:
        train_dir, test_dir = split_data(
            data_dir, split=split_percent, 
            shuffle_flag=shuffle_flag, overwrite=overwrite)
    else:
        train_dir, test_dir = get_train_test_dirs(data_dir, shuffle_flag)
    
    time_series = True
    if time_series:
        dataset = ClimateDataset(
            data_dir, sin_embed=sin_embed, time_dim=time_dim, 
            dtype=dataset_feature, time_series=time_series) 
        minim_val, maxim_val = dataset.min_max_normalize(
            minim=None, maxim=None, per_pixel=normalize_per_pixel, 
            minmax_range=minmax_range)  
        if save_dir is None:
            save_dir = saved_root
        if normalization_type == 'minmax':
            minim_val.dump(os.path.join(save_dir, 'SST_min_'+dataset_name+'.pkl'))
            maxim_val.dump(os.path.join(save_dir, 'SST_max_'+dataset_name+'.pkl'))
        print('dataset.original_data_train.shape:', dataset.original_data_train.shape)
        print('dataset.original_data_test.shape:', dataset.original_data_test.shape)
        train_dataset = TimeSeriesDataset(
            dataset.original_data_train, dataset.train_years, dataset.mask)
        test_dataset = TimeSeriesDataset(
            dataset.original_data_test, dataset.test_years, dataset.mask)
        
        # Create dataloader
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, 
            shuffle=shuffle, num_workers=num_workers, 
            pin_memory=True)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, 
            shuffle=shuffle, num_workers=num_workers, 
            pin_memory=True)
            
        train_num, test_num = train_dataset.__len__(), test_dataset.__len__()

        return train_dataloader, test_dataloader, train_num, test_num        




              
    print('===============================')
    print(dataset.original_data.shape)
    print('===============================')
    train_dataset = ClimateDataset(
        train_dir, sin_embed=sin_embed, time_dim=time_dim, dtype=dataset_feature)
    test_dataset = ClimateDataset(
        test_dir, sin_embed=sin_embed, time_dim=time_dim, dtype=dataset_feature)
    
    if normalization_type == 'minmax':
        minim_val, maxim_val = train_dataset.min_max_normalize(
            minim=None, maxim=None, per_pixel=normalize_per_pixel, 
            minmax_range=minmax_range)
        test_dataset.min_max_normalize(
            minim=minim_val, maxim=maxim_val, per_pixel=normalize_per_pixel, 
            minmax_range=minmax_range)
        
    elif normalization_type == 'standard':
        avg, std = train_dataset.normalize(per_pixel=normalize_per_pixel)
        avg, std = test_dataset.normalize(
            avg, std, per_pixel=normalize_per_pixel)
        
    elif normalization_type == 'percentile':
        lower, upper, median = train_dataset.percentile_normalize(
            lower=None, upper=None, median=None, zero_center=zero_center, 
            lower_percentile=lower_percentile, upper_percentile=upper_percentile)
        test_dataset.percentile_normalize(
            lower, upper, median, zero_center=zero_center, 
            lower_percentile=lower_percentile, upper_percentile=upper_percentile)
        
    elif normalization_type == 'none':
        pass
     
    else:
        print(f'Normalization type {normalization_type} not implemented.')
        exit()

    if normalize_per_pixel:
        if save_dir is None:
            save_dir = saved_root
        if normalization_type == 'minmax':
            minim_val.dump(os.path.join(save_dir, 'SST_min_'+dataset_name+'.pkl'))
            maxim_val.dump(os.path.join(save_dir, 'SST_max_'+dataset_name+'.pkl'))
        elif normalization_type == 'standard':
            avg.dump(os.path.join(save_dir, 'SST_avg_'+dataset_name+'.pkl'))
            std.dump(os.path.join(save_dir, 'SST_std_'+dataset_name+'.pkl'))
        elif normalization_type == 'percentile':
            lower.dump(os.path.join(save_dir, 'SST_lower_'+dataset_name+'.pkl'))
            upper.dump(os.path.join(save_dir, 'SST_upper_'+dataset_name+'.pkl'))
            median.dump(os.path.join(save_dir, 'SST_median_'+dataset_name+'.pkl'))
        elif normalization_type == 'none':
            pass
        else:
            print(f'Normalization type {normalization_type} not implemented.')
            exit()
    
    # Test number of workers:
    # workers_to_test = [0, 1, 2, 4, 8, 16, 32] #optimal: 8
    # results = benchmark_num_workers(train_dataset, batch_size=batch_size, num_workers_list=workers_to_test)
    # print(f'results: {results}')
        
    # Create dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=shuffle, num_workers=num_workers, 
        pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, 
        shuffle=shuffle, num_workers=num_workers, 
        pin_memory=True)
        
    train_num, test_num = train_dataset.__len__(), test_dataset.__len__()

    return train_dataloader, test_dataloader, train_num, test_num
    

def benchmark_num_workers(dataset, batch_size, num_workers_list):
    results = {}
    
    for num_workers in num_workers_list:
        start_time = time()
        
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
        
        # Time one complete epoch
        for _ in train_loader:
            pass
            
        end_time = time()
        elapsed_time = end_time - start_time
        
        results[num_workers] = elapsed_time
        print(f"Workers: {num_workers}, Time: {elapsed_time:.2f}s")
    
    # Find optimal number of workers
    optimal_workers = min(results, key=results.get)
    print(f"\nOptimal number of workers: {optimal_workers}")
    
    return results

