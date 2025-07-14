import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('/home/gc/projects/openstl_weather/')
import random
import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from openstl.datasets.utils import create_loader
import tqdm

try:
    import xarray as xr
except ImportError:
    xr = None

d2r = np.pi / 180


def latlon2xyz(lat, lon):
    if type(lat) == torch.Tensor:
        x = -torch.cos(lat)*torch.cos(lon)
        y = -torch.cos(lat)*torch.sin(lon)
        z = torch.sin(lat)

    if type(lat) == np.ndarray:
        x = -np.cos(lat)*np.cos(lon)
        y = -np.cos(lat)*np.sin(lon)
        z = np.sin(lat)
    return x, y, z


def xyz2latlon(x, y, z):
    if type(x) == torch.Tensor:
        lat = torch.arcsin(z)
        lon = torch.atan2(-y, -x)

    if type(x) == np.ndarray:
        lat = np.arcsin(z)
        lon = np.arctan2(-y, -x)
    return lat, lon


data_map = {
    'z': 'geopotential',
    't': 'temperature',
    'tp': 'total_precipitation',
    't2m': '2m_temperature',
    'r': 'relative_humidity',
    's': 'specific_humidity',
    'u10': '10m_u_component_of_wind',
    'u': 'u_component_of_wind',
    'v10': '10m_v_component_of_wind',
    'v': 'v_component_of_wind',
    'tcc': 'total_cloud_cover',
    "lsm": "constants",
    "o": "constants",
    "l": "constants",
}

mv_data_map = {
    **dict.fromkeys(['mv', 'mv4'], ['r', 't', 'u', 'v']),
    'mv5': ['z', 'r', 't', 'u', 'v'],
    'uv10': ['u10', 'v10'],
    'mv12': ['lsm', 'o', 't2m', 'u10', 'v10', 'l', 'z', 'u', 'v', 't', 'r', 's'],
    'mv4my': ['u10', 'v10', 't2m', 'tcc'],
    'mv5my': ['u10', 'v10', 't2m', 'tcc','r'],
    # 'mv5my': ['u10', 'v10', 'tcc', 't2m','r'],
    'mv6my': ['u10', 'v10', 't2m', 'tcc', 'r', 'tp']
}

data_keys_map = {
    'o': 'orography',
    'l': 'lat2d',
    's': 'q'
}


class WeatherBenchDataset(Dataset):
    """Wheather Bench Dataset <http://arxiv.org/abs/2002.00469>`_

    Args:
        data_root (str): Path to the dataset.
        data_name (str|list): Name(s) of the weather modality in Wheather Bench.
        training_time (list): The arrange of years for training.
        idx_in (list): The list of input indices.
        idx_out (list): The list of output indices to predict.
        step (int): Sampling step in the time dimension.
        level (int|list|"all"): Level(s) to use.
        data_split (str): The resolution (degree) of Wheather Bench splits.
        use_augment (bool): Whether to use augmentations (defaults to False).
    """

    def __init__(self, data_root, data_name, training_time,
                 idx_in, idx_out, step=1, levels=['1000'], data_split='5_625',
                 mean=None, std=None,
                 transform_data=None, transform_labels=None, use_augment=False):
        super().__init__()
        self.data_root = data_root
        # self.data_name = data_name
        self.data_split = data_split
        self.training_time = training_time
        self.idx_in = np.array(idx_in)
        self.idx_out = np.array(idx_out)
        self.step = step
        self.data = None
        self.mean = mean
        self.std = std
        self.transform_data = transform_data
        self.transform_labels = transform_labels
        self.use_augment = use_augment

        self.time = None
        self.time_size = self.training_time
        shape = int(32 * 5.625 / float(data_split.replace('_', '.')))
        self.shape = (shape, shape * 2)

        self.data, self.mean, self.std = [], [], []

        if levels == 'all':
            levels = ['50', '250', '500', '600', '700', '850', '925']
        levels = levels if isinstance(levels, list) else [levels]
        levels = [int(level) for level in levels]
        if isinstance(data_name, str) and data_name in mv_data_map:
            data_names = mv_data_map[data_name]
        else:
            data_names = data_name if isinstance(data_name, list) else [data_name]
        if len(data_names) > 1:
            self.data_name = data_names
        else:
            self.data_name = data_names[0]
        for name in tqdm.tqdm(data_names):
            data, mean, std = self._load_data_xarray(data_name=name, levels=levels)
            self.data.append(data)
            self.mean.append(mean)
            self.std.append(std)

        for i, data in enumerate(self.data):
            if data.shape[0] != self.time_size:
                self.data[i] = data.repeat(self.time_size, axis=0)

        self.data = np.concatenate(self.data, axis=1)
        self.mean = np.concatenate(self.mean, axis=1)
        self.std = np.concatenate(self.std, axis=1)
        # print(self.mean.shape)
        # print(self.data.shape)
        # print(self.training_time)
        self.valid_idx = np.array(
            range(-idx_in[0], self.data.shape[0]-idx_out[-1]-1))
        # print(self.valid_idx)
        # raise ValueError('====================================================')

    def _load_data_xarray(self, data_name, levels):
        """Loading full data with xarray"""
        try:
            dataset = xr.open_mfdataset(self.data_root+'/{}/{}*.nc'.format(
                data_map[data_name], data_map[data_name]), combine='by_coords')
        except (AttributeError, ValueError):
            assert False and 'Please install xarray and its dependency (e.g., netcdf4), ' \
                                'pip install xarray==0.19.0,' \
                                'pip install netcdf4 h5netcdf dask'
        except OSError:
            print("OSError: Invalid path {}/{}/*.nc".format(self.data_root, data_map[data_name]))
            assert False

        if 'time' not in dataset.indexes:
            dataset = dataset.expand_dims(dim={"time": 1}, axis=0)
        else:
            dataset = dataset.sel(time=slice(*self.training_time))
            dataset = dataset.isel(time=slice(None, -1, self.step))
            self.time_size = dataset.dims['time']
        # print(levels)
        # raise ValueError('--------------------------------------------')
        if 'level' not in dataset.indexes:
            dataset = dataset.expand_dims(dim={"level": 1}, axis=1)
        else:
            dataset = dataset.sel(level=np.array(levels))

        if data_name in data_keys_map:
            data = dataset.get(data_keys_map[data_name]).values
        else:
            data = dataset.get(data_name).values


        # mean = data.mean().reshape(1, 1, 1, 1)
        # std = data.std().reshape(1, 1, 1, 1)
        # mean = dataset.mean('time').mean(('lat', 'lon')).compute()[data_name].values
        # std = dataset.std('time').mean(('lat', 'lon')).compute()[data_name].values
        mean = np.mean(data, axis=(0, 2, 3), keepdims=True)
        std = np.std(data, axis=(0, 2, 3), keepdims=True)
        data = (data - mean) / std
        # mean = np.repeat(mean, 3, axis=1)
        # std = np.repeat(std, 3, axis=1)
        return data, mean, std

    def _augment_seq(self, seqs, crop_scale=0.96):
        """Augmentations as a video sequence"""
        _, _, h, w = seqs.shape  # original shape, e.g., [4, 1, 128, 256]
        seqs = F.interpolate(seqs, scale_factor=1 / crop_scale, mode='bilinear')
        _, _, ih, iw = seqs.shape
        # Random Crop
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        seqs = seqs[:, :, x:x+h, y:y+w]
        # Random Flip
        if random.randint(0, 1):
            seqs = torch.flip(seqs, dims=(3, ))  # horizontal flip
        return seqs

    def __len__(self):
        return self.valid_idx.shape[0]

    def __getitem__(self, index):
        index = self.valid_idx[index]
        data = torch.tensor(self.data[index+self.idx_in])
        labels = torch.tensor(self.data[index+self.idx_out])
        if self.use_augment:
            len_data = self.idx_in.shape[0]
            seqs = self._augment_seq(torch.cat([data, labels], dim=0), crop_scale=0.96)
            data = seqs[:len_data, ...]
            labels = seqs[len_data:, ...]
        return data, labels


def load_data(batch_size,
              val_batch_size,
              data_root,
              num_workers=4,
              data_split='5_625',
              data_name='t2m',
              train_time=['1979', '2015'],
              val_time=['2016', '2016'],
              test_time=['2017', '2018'],
              idx_in=[-11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0],
              idx_out=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              step=1,
              levels=['1000'],
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False,
              **kwargs):

    assert data_split in ['5_625', '2_8125', '1_40625']
    for suffix in [f'weather_{data_split}deg', f'weather', f'{data_split}deg']:
        if osp.exists(osp.join(data_root, suffix)):
            weather_dataroot = osp.join(data_root, suffix)

    train_set = WeatherBenchDataset(data_root=weather_dataroot,
                                    data_name=data_name, data_split=data_split,
                                    training_time=train_time,
                                    idx_in=idx_in,
                                    idx_out=idx_out,
                                    step=step, levels=levels, use_augment=use_augment)
    vali_set = WeatherBenchDataset(weather_dataroot,
                                    data_name=data_name, data_split=data_split,
                                    training_time=val_time,
                                    idx_in=idx_in,
                                    idx_out=idx_out,
                                    step=step, levels=levels, use_augment=False,
                                    mean=train_set.mean,
                                    std=train_set.std)
    test_set = WeatherBenchDataset(weather_dataroot,
                                    data_name, data_split=data_split,
                                    training_time=test_time,
                                    idx_in=idx_in,
                                    idx_out=idx_out,
                                    step=step, levels=levels, use_augment=False,
                                    mean=train_set.mean,
                                    std=train_set.std)

    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers,
                                     distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_vali = create_loader(test_set, # validation_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)

    return dataloader_train, dataloader_vali, dataloader_test


if __name__ == '__main__':
    from openstl.core import metric
    data_split=['5_625', '1_40625']
    # data_name = 'uv10'
    data_split=['5_625',]
    data_name = 'mv5my'
    level = [150, 500, 850]
    
    for _split in data_split:
        step, level = 24, []
        dataloader_train, _, dataloader_test = \
            load_data(batch_size=1,
                    val_batch_size=1,
                    data_root='/home/gc/projects/openstl_weather/openstl/data',
                    num_workers=4, data_name=data_name,
                    data_split=_split,
                    train_time=['2010', '2015'],
                    val_time=['2016', '2016'],
                    test_time=['2017', '2018'],
                    levels=level,
                    # idx_in=[1+i*6 for i in range(-3, 0)] + [0,],
                    # idx_out=[i*6 + 1 for i in range(4)],
                    # step=6, 
                    # level=level, 
                    use_augment=False)

        print(len(dataloader_train), len(dataloader_test))
        for item in dataloader_train:
            print('train', item[0].shape)
            print(dataloader_train.dataset.data_name)
            # if 'mv' in data_name:
            _, log = metric(item[0].cpu().numpy(), item[1].cpu().numpy(),
                            channel_names=dataloader_train.dataset.data_name, spatial_norm=True)
            print(log)
                
            break
        for item in dataloader_test:
            print('test', item[0].shape)
            break
