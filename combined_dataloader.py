import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import cv2
from torchvision import transforms

look_back=6
look_forward=4
# data=np.load('../pytorch_lstm/CPV_processed/6_stations/磁县池上光伏电站（清洗后）.npy')
data=np.load('../pytorch_lstm/CPV_processed/涉县台村光伏电站（清洗后）.npy')
month=(data[0]%100000000)//1000000
day=(data[0]%1000000)//10000
hour=(data[0]%10000)//100
min=(data[0]%100)
data_=np.zeros([data.shape[0]+2,data.shape[1]]).astype('float32')
data_[0,:]=month
data_[1,:]=day
data_[2,:]=hour
data_[3,:]=min
data_[4:]=data[2:]
features_dim = data_.shape[0]
dataset = data_.astype('float32')
for i in range(data_.shape[0]):
    std_value = dataset[i].std()
    mean_value = dataset[i].mean()
    dataset[i] = (dataset[i] - mean_value) / std_value


def resize(field,SIZE=256):
    f_new = cv2.resize(field, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    return np.array(f_new, dtype=np.float32)

class station_dataset(Dataset):
    def __init__(self,dataset, swr_dir='/hdd/dataset_power/nc_SWR_15',look_back=look_back ,look_forward=look_forward,transform=None,train=True,only_day=True):
        self.transform=transform
        self.swr_dir = swr_dir
        self.look_back = look_back
        self.look_forward = look_forward
        self.swr_mean = 0  # 0 474
        self.swr_std = 1353  # 1353 386
        data_train = np.zeros([1, 1, 1])
        data_train = data_train.astype('float32')
        data_val = np.zeros([1, 1, 1])
        data_val = data_train.astype('float32')
        for i in range(dataset.shape[1] - look_back - look_forward + 1):
            data_min = dataset[:, i:(i + look_back + look_forward)]
            data_min = data_min[np.newaxis, :, :]
            day_mean = int((data_min[0][1] * day.std() + day.mean()).min())
            if (only_day == False or (data_min[0, 2, :][0] * hour.std() + hour.mean() + (data_min[0, 3, :][0] * min.std() + min.mean()) / 60 > 7.9
                and data_min[0, 2, :][0] * hour.std() + hour.mean() + (data_min[0, 3, :][0] * min.std() + min.mean()) / 60 < 17.1
                and data_min[0, 2, :][-1] * hour.std() + hour.mean() + (data_min[0, 3, :][-1] * min.std() + min.mean()) / 60 < 17.1)):
                if (day_mean % 5 != 0):
                    # if (i % 220 >= 45):
                    if (data_train.shape[1] == 1):
                        data_train = data_min
                    else:
                        data_train = np.insert(data_train, -1, values=data_min, axis=0)
                else:
                    if (data_val.shape[1] == 1):
                        data_val = data_min
                    else:
                        data_val = np.insert(data_val, -1, values=data_min, axis=0)
        if(train == True):
            self.data=data_train
        elif(train == False):
            self.data=data_val

    def __len__(self):
        return self.data.__len__()
    def get_time(self,x):
        month_ = str(round(x[0][0] * month.std() + month.mean())).zfill(2)
        day_ = str(round(x[1][0] * day.std() + day.mean())).zfill(2)
        hour_ = str(round(x[2][0] * hour.std() + hour.mean())).zfill(2)
        min_ = round(x[3][0] * min.std() + min.mean())
        if(min_%10==9 or min_%10==4):
            min_=min_+1
        elif(min_%10==1 or min_%10==6):
            min_=min_-1
        min_=str(min_).zfill(2)
        if(int(month_)<7):
            time_str = '2019' + month_ + day_ + hour_ + min_
        else:
            time_str='2018'+month_+day_+hour_+min_
        return time_str
    def __getitem__(self, index):
        x=self.data[index,:,:self.look_back]
        time_str=self.get_time(x)
        swr_dir=os.path.join(self.swr_dir,time_str[:8])
        swr_dir_list=glob.glob(swr_dir+'/*.npy')
        swr_dir_list.sort()
        for i in range(len(swr_dir_list)):
            if(time_str in swr_dir_list[i]):
                start_index=i-2
                break
        for i in range(6):
            try:
                satellite_path = swr_dir_list[start_index+i]
            except:
                print(time_str)
            swr=resize(np.load(satellite_path)[109:240,111:239]).reshape((1,256,256))#[109:240,111:239]
            swr=(swr-self.swr_mean)/self.swr_std
            if i == 0:
                observation =swr
            else:
                observation = np.concatenate((observation, swr),axis=0)
        for j in range(look_back,look_back+look_forward):#
            satellite_path = swr_dir_list[start_index + j]
            swr=resize(np.load(satellite_path)[109:240,111:239]).reshape((1,256,256))#[109:240,111:239]
            swr=(swr-self.swr_mean)/self.swr_std
            if j == look_back:
                ground_truth = swr
            else:
                ground_truth = np.concatenate((ground_truth, swr),axis=0)
        swr_sample = {
            'observation': observation,
            'ground_truth': ground_truth,
        }
        if self.transform:
            sample = self.transform(swr_sample)
        # x=torch.tensor(x).to(torch.float32)
        # y=self.data[index,-1,self.look_back:self.look_back+self.look_forward]
        # y = torch.tensor(y).to(torch.float32).unsqueeze(0)
        xa = np.concatenate((self.data[index, :8, :self.look_back], self.data[index, -1:, :self.look_back]), axis=0) #history msv+power
        xb = self.data[index, 8:, self.look_back:self.look_back + self.look_forward]  #future nwp+power
        y = self.data[index, -1:, self.look_back:self.look_back + self.look_forward]
        return xa,xb,y,sample
class ToTensor(object):
    def __call__(self, sample):
        sample['ground_truth'] = torch.from_numpy(sample['ground_truth']).type(torch.FloatTensor)
        sample['observation'] = torch.from_numpy(sample['observation']).type(torch.FloatTensor)
        return sample


# transformer = transforms.Compose([ToTensor()])
# dataset_train = station_dataset(dataset,transform=transformer)
# dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=0)
# for step, sample in enumerate(dataloader_train):  # [batch_size,channel_number,H,W]
#     print(sample[0])