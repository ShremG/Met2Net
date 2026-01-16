import sys
sys.path.append('/home/gc/projects/openstl_wind/')
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ERA5Dataset(Dataset):
    def __init__(self, npz_path, seq_len=12, pred_len=12, step_size=12):
        
        loaded = np.load(npz_path, allow_pickle=True)
        self.data = loaded["data"]  # (4, time, lat, lon)
        self.stats = loaded["stats"].item()  # 还原字典数据
        self.step_size = step_size if step_size else (seq_len + pred_len)  # 不重叠窗口

        # 提取最大值和最小值（确保是 NumPy 数组）
        self.max_values = np.array([self.stats[var]["max"] for var in ["msl", "u10", "v10", "t2m"]])[:, np.newaxis, np.newaxis, np.newaxis]
        self.min_values = np.array([self.stats[var]["min"] for var in ["msl", "u10", "v10", "t2m"]])[:, np.newaxis, np.newaxis, np.newaxis]

        self.mean = None
        self.std = None
        self.data_name = ["msl", "u10", "v10", "t2m"]

        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 计算索引范围
        self.total_timesteps = (self.data.shape[1] - (seq_len + pred_len)) // self.step_size + 1
        # 最大最小归一化
        self.data = (self.data - self.min_values) / (self.max_values - self.min_values + 1e-8)
        self.data = np.transpose(self.data, (1, 0, 2, 3)) # 4 time h w -> time 4 h w

        

    def __len__(self):
        return self.total_timesteps

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]  # (4, 12, lat, lon)
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]  # (4, 12, lat, lon)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    

def load_data(batch_size, val_batch_size,input_len=12, pred_len=12, num_workers=4):
    
    

    train_dataset = ERA5Dataset('/home/gc/projects/openstl_wind/data/era5/train_128_128.npz', input_len, pred_len,step_size=6)
    vail_dataset = ERA5Dataset('/home/gc/projects/openstl_wind/data/era5/vail_128_128.npz', input_len, pred_len)
    test_dataset = ERA5Dataset('/home/gc/projects/openstl_wind/data/era5/test_128_128.npz', input_len, pred_len)
    print('---------------------------dataset--------------------------')
    print(train_dataset.__len__())
    print(vail_dataset.__len__())
    print(test_dataset.__len__())


    dataloader_train = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=num_workers)
    dataloader_vali = torch.utils.data.DataLoader(vail_dataset,batch_size=val_batch_size,shuffle=False,drop_last=False,num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(test_dataset,batch_size=val_batch_size,shuffle=False,drop_last=False,num_workers=num_workers)
    
    return dataloader_train, dataloader_vali, dataloader_test




if __name__ == '__main__':
    dataloader_train, dataloader_vali, dataloader_test = load_data(1,1)
    for item in dataloader_train:
        x,y = item
        print(x.shape,y.shape)
        break
