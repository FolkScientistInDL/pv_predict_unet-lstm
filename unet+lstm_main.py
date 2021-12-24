import numpy as np
import random
import math
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import time
import argparse

import combined_dataloader
import lstm_model
import UResNext
import parse_result


def parse_args():
    parser = argparse.ArgumentParser(description='Train LSTM_UNET-based PV predictors')
    parser.add_argument('--station_location_txtdir', default='CPV_location.txt',
                        help='the dir to .txt file that indicate the Latitude and longitude position pf each PV station')
    parser.add_argument('--station_datadir', default='CPV_processed_stations_NWP+MSV/',
                        help='the dir to folder to store time sequence data of each PV station')
    parser.add_argument('--swr_dir', default='/hdd/dataset_power/nc_SWR_15_30',
                        help='the dir to folder to store swr data')
    parser.add_argument('--history_step', type=int, default=6, help='history_steps')
    parser.add_argument('--predicted_step', type=int, default=4, help='predicted_step')
    parser.add_argument(
        '--Train_Unet',
        action='store_true',
        help='whether to do pretraining on UNET model')
    parser.add_argument(
        '--NWP_OFF',
        action='store_true',
        help='whether to turn off NWP')
    parser.add_argument(
        '--LMD_OFF',
        action='store_true',
        help='whether to turn off LMD')
    parser.add_argument(
        '--SWR_OFF',
        action='store_true',
        help='whether to turn off SWR')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--train_epoches', type=int, default=50, help='train_epoches')
    args = parser.parse_args()
    return args
args = parse_args()

Train_Unet = args.Train_Unet
NWP_OFF = args.NWP_OFF
MSV_OFF = args.LMD_OFF
SWR_OFF = args.SWR_OFF
combined_dataloader.look_back = args.history_step
combined_dataloader.look_forward = args.predicted_step

def seed_torch(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_torch(args.seed)

def get_swr_index(station_lon=114.09231, station_lag=36.43593,size=128):
    '''
    transfer station_lon,station_lag to x,y position on feature map
    '''
    y_index = (42.333 - station_lag + 0.05) / (6.583 + 0.05) * size - 1
    x_index = (station_lon - 113.45 + 0.05) / (6.383 + 0.05) * size - 1
    return y_index,x_index

class ToTensor(object):
    def __call__(self, sample):
        sample['ground_truth'] = torch.from_numpy(sample['ground_truth']).type(torch.FloatTensor)
        sample['observation'] = torch.from_numpy(sample['observation']).type(torch.FloatTensor)
        return sample
transformer = transforms.Compose([ToTensor()])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    with open(args.station_location_txtdir, "r",encoding='gbk') as f:
        stations=f.readlines()
except:
    with open(args.station_location_txtdir, "r",encoding='utf8') as f:
        stations=f.readlines()
if __name__ == "__main__":
    for i in range(stations.__len__()):  # Training on all the stations listed in txtfile
        station=stations[i].split('\t')
        station_name=station[0]
        lon = eval(station[1])
        lag = eval(station[2])
        print('training on '+station_name)

        # transfer station_lon,station_lag to x,y position on feature map
        station_x,station_y=get_swr_index(lon,lag)
        if(station_y<2.5):
            station_y=2.5
        elif(station_y>124.5):
            station_y=124.5
        if (station_x < 2.5):
            station_x = 2.5
        elif (station_x > 124.5):
            station_x = 124.5

        # load the time sequence data, perform normalization and time embedding
        data = np.load(args.station_datadir+station_name)
        month = (data[0] % 100000000) // 1000000
        day = (data[0] % 1000000) // 10000
        hour = (data[0] % 10000) // 100
        min = (data[0] % 100)
        data_ = np.zeros([data.shape[0] + 2, data.shape[1]]).astype('float32')
        data_[0, :] = month
        data_[1, :] = day
        data_[2, :] = hour
        data_[3, :] = min
        data_[4:] = data[2:]
        features_dim = data_.shape[0]
        dataset = data_.astype('float32')
        for i in range(data_.shape[0]):
            std_value = dataset[i].std()
            mean_value = dataset[i].mean()
            if (std_value == 0):
                dataset[i] = 0
            else:
                dataset[i] = (dataset[i] - mean_value) / std_value
        if (MSV_OFF):
            dataset[4:8, :] = 0
        if (NWP_OFF):
            dataset[8:-1, :] = 0
        # Divide the data set into trainset/valset
        dataset_train = combined_dataloader.station_dataset(dataset,
                                                            swr_dir=args.swr_dir,
                                                            transform=transformer, train=True)
        dataset_val = combined_dataloader.station_dataset(dataset,
                                                          swr_dir=args.swr_dir, transform=transformer,
                                                          train=False)
        dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=8)
        dataloader_val = DataLoader(dataset_val, batch_size=16, shuffle=False, num_workers=8)
        # LSTM model initialization
        encoder = lstm_model.Encoder(9, 64, 2, drop_prob=0.5).to(device)
        decoder = lstm_model.Decoder(5, 64, 2, 10, drop_prob=0.5, atten=True).to(device)
        enc_optimizer = torch.optim.AdamW(encoder.parameters(), lr=0.001)
        dec_optimizer = torch.optim.AdamW(decoder.parameters(), lr=0.001)
        enc_scheduler = torch.optim.lr_scheduler.StepLR(enc_optimizer, step_size=1, gamma=0.95)
        dec_scheduler = torch.optim.lr_scheduler.StepLR(dec_optimizer, step_size=1, gamma=0.95)
        loss_fn = nn.MSELoss()
        # UNET model initialization
        unet_model = UResNext.UneXt50(input_layers=combined_dataloader.look_back, output_layers=combined_dataloader.look_forward).to(device)
        if (Train_Unet == False):
            unet_model = torch.load('unet_pretrain.pt')
        model_split = UResNext.split_layers(unet_model)
        unet_optimizer = torch.optim.AdamW(unet_model.parameters(), lr=1e-4, weight_decay=1e-2)
        unet_scheduler = torch.optim.lr_scheduler.StepLR(unet_optimizer, step_size=1, gamma=0.95)
        unet_loss_func = nn.MSELoss()
        # CatHead model (the model that connect UNET and LSTM) initialization
        CatHead = UResNext.CatHead(station_x=station_x, station_y=station_y, size=6).to(device)
        CatHead_optimizer = torch.optim.AdamW(CatHead.parameters(), lr=1e-2, weight_decay=1e-2)
        CatHead_scheduler = torch.optim.lr_scheduler.StepLR(CatHead_optimizer, step_size=1, gamma=0.95)

        best_test_loss = 999
        best_test_mae = 999
        best_test_mape = 999
        minimal_swr_test_loss = 999
        best_metric = 999

        for e in range(args.train_epoches):
            # Training
            time_start = time.time()
            loss_sum_unet_train = 0
            loss_sum_train = 0
            loss_sum_test = 0
            train_loss_list=[]
            unet_model.train()
            encoder.train()
            decoder.train()
            CatHead.train()
            for step, sample in enumerate(dataloader_train):  # [batch_size,channel_number,H,W]
                var_xa = sample[0].to(device).transpose(1, 2)
                var_xb = sample[1].to(device).transpose(1, 2)
                var_y = sample[2].to(device).transpose(1, 2)
                ground_truth = sample[3]['ground_truth'].to(device)
                observation = sample[3]['observation'].to(device)
                l_sum = 0.0
                enc_optimizer.zero_grad()
                dec_optimizer.zero_grad()
                unet_optimizer.zero_grad()
                CatHead_optimizer.zero_grad()
                # UNET infer
                if(SWR_OFF==False):
                    if(Train_Unet==True):
                        ims, feature_swr = unet_model(observation)
                    else:
                        with torch.no_grad():
                            ims,feature_swr = unet_model(observation)
                    losst = 0
                    ratio = 0
                    for ix in range(ground_truth.size()[0]):
                        losst += loss_fn(ims[ix, :, :, :], ground_truth[ix, :, :, :])
                    ratio = 1 / dataloader_train.batch_size
                    loss = losst * ratio
                    loss_item = loss.item()
                    last_loss = loss
                    loss_sum_unet_train += loss_item
                    if (Train_Unet == True):
                        loss.backward(retain_graph=True)

                    #CatHead infer
                    feature_swr=CatHead(feature_swr)
                elif(SWR_OFF):
                    feature_swr = torch.zeros([var_xa.shape[0], 2, 64]).cuda()
                # LSTM infer
                l = lstm_model.batch_loss(encoder, decoder,var_xa, var_xb, loss_fn,feature_swr=feature_swr)  # criterion
                train_loss_list.append(l.cpu().data.item())
                # Freeze the UNET model weight when updating the LSTM model
                for model_item in model_split[0]:
                    model_item.requires_grad = False
                l.backward()
                for model_item in model_split[0]:
                    model_item.requires_grad = True

                unet_optimizer.step()
                enc_optimizer.step()
                dec_optimizer.step()
                CatHead_optimizer.step()
            enc_scheduler.step()
            dec_scheduler.step()
            unet_scheduler.step()
            CatHead_scheduler.step()

            unet_model.eval()
            encoder.eval()
            decoder.eval()
            CatHead.eval()
            test_loss_list = [[]for i in range(combined_dataloader.look_forward)]
            test_mae_list = [[]for i in range(combined_dataloader.look_forward)]
            test_mape_list = [[]for i in range(combined_dataloader.look_forward)]
            loss_sum_unet_test = 0
            loss_sum_train = 0
            loss_sum_test = 0
            for step, sample in enumerate(dataloader_val): #eval
                var_xa = sample[0].to(device).transpose(1, 2)
                var_xb = sample[1].to(device).transpose(1, 2)
                var_y = sample[2].to(device).transpose(1, 2)
                ground_truth = sample[3]['ground_truth'].to(device)
                observation = sample[3]['observation'].to(device)
                with torch.no_grad():
                    #UNET infer
                    if (SWR_OFF == False):
                        test_ims,feature_swr = unet_model(observation)
                        losst = 0
                        ratio = 0
                        for ix in range(ground_truth.size()[0]):
                            losst += loss_fn(test_ims[ix, :, :, :], ground_truth[ix, :, :, :])
                        ratio = 1 / dataloader_val.batch_size
                        loss = losst * ratio
                        loss_item = loss.item()
                        last_loss = loss
                        loss_sum_unet_test += loss_item

                        # CatHead infer
                        feature_swr = CatHead(feature_swr)
                    elif (SWR_OFF):
                        feature_swr = torch.zeros([var_xa.shape[0], 2, 64]).cuda()
                    # LSTM infer
                    out = lstm_model.predict(encoder, decoder, var_xa, var_xb, combined_dataloader.look_forward,feature_swr=feature_swr)
                    for i in range(combined_dataloader.look_forward):
                        test_loss = loss_fn(out[:,i,:], var_y[:,i,:]).cpu().data.item()
                        test_mae = lstm_model.masked_mae(out[:,i,:], var_y[:,i,:]).cpu().data.item()*std_value
                        test_mape=lstm_model.masked_mape(out[:,i,:]*std_value+mean_value,var_y[:,i,:]*std_value+mean_value).cpu().data.item()
                        test_loss_list[i].append(test_loss)
                        test_mae_list[i].append(test_mae)
                        test_mape_list[i].append(test_mape)

            if(minimal_swr_test_loss>=loss_sum_unet_test):
                minimal_swr_test_loss=loss_sum_unet_test
                if(Train_Unet==True):
                    torch.save(unet_model,'unet_pretrain.pt')
            '''
            define the metric of the model
            2 1 4 : save the model that balance 3 metric
            To save the model that optimized for only one metric, set the other to weight to zero
            '''
            metric = std_value * math.sqrt(np.array(test_loss_list).mean()) * 2 + np.array(test_mae_list).mean() * 1 + np.array(test_mape_list).mean() *4
            if (best_metric >= metric):
                best_test_loss=np.zeros(combined_dataloader.look_forward)
                best_test_mae = np.zeros(combined_dataloader.look_forward)
                best_test_mape = np.zeros(combined_dataloader.look_forward)
                best_metric = metric
                for i in range(combined_dataloader.look_forward):
                    best_test_loss[i] = np.array(test_loss_list)[i].mean()
                    best_test_mae[i] = np.array(test_mae_list)[i].mean()
                    best_test_mape[i] = np.array(test_mape_list)[i].mean()
                torch.save(encoder, 'LSTM_model_weights/best_encoder_' + station_name + '.pt')
                torch.save(decoder, 'LSTM_model_weights/best_decoder_' + station_name + '.pt')
                torch.save(CatHead, 'LSTM_model_weights/best_CatHead_' + station_name + '.pt')
            time_end = time.time()
            time_c = time_end - time_start
            if (e + 1) % 1 == 0:
                for i in range(combined_dataloader.look_forward):
                    print(str(((i+1)*15))+'min ','LSTM:Epoch: {}, Train_Loss: {:.5f}, Val_Loss: {:.5f}, Best_Val_RMSE: {:.5f}, Best_Val_MAE: {:.5f}, Best_Val_MAPE: {:.5f}, Time Cost: {:.3f}s'.format(e + 1, np.array(train_loss_list).mean(), np.array(test_loss_list).mean(), std_value*math.sqrt(best_test_loss[i]),best_test_mae[i],best_test_mape[i],time_c))
                print(f'UNET:Epoch {e + 1:03d} | Train loss: {loss_sum_unet_train / (step + 1)}', end="")
                print(f' | Test loss: {loss_sum_unet_test / (step + 1)}', end="")
                print(f' | minimal_test_loss: {math.sqrt(minimal_swr_test_loss/ (step + 1)) * dataset_train.swr_std }')
        txt_name='PV_predict_Unet+Lstm'
        txt_name=txt_name
        if(SWR_OFF):
            txt_name=txt_name+'SWR_OFF'
        if(MSV_OFF):
            txt_name=txt_name+'MSV_OFF'
        if (NWP_OFF):
            txt_name = txt_name + 'NWP_OFF'
        txt_name=txt_name+'.txt'
        with open(txt_name, "a+") as f:
            for i in range(combined_dataloader.look_forward):
                f.write('PV_predict:\t'+str(((i+1)*15))+'min '+station_name+'Best_Val_RMSE: {:.5f}, Best_Val_MAE: {:.5f}, Best_Val_MAPE: {:.5f}'.format(std_value*math.sqrt(best_test_loss[i]),best_test_mae[i],best_test_mape[i])+'\n')
        torch.cuda.empty_cache()
    # generate experiment report
    parse_result.write_csv(file_name = txt_name[:-4],steps = combined_dataloader.look_forward)