import csv
import pandas as pd

def write_csv(file_name='', steps=4):
    with open(file_name+'.txt',"r",encoding='utf8') as f:
        stations_result = f.readlines()
    f = open(file_name+'.csv','w+',encoding='gb2312',newline='')
    csv_writer = csv.writer(f)
    time_steps=steps
    tab_list=['station names']
    for i in range(time_steps):
        tab_list.append(str((i+1)*15)+'min RMSE')
    for i in range(time_steps):
        tab_list.append(str((i+1)*15)+'min MAE')
    for i in range(time_steps):
        tab_list.append(str((i+1)*15)+'min MAPE')
    csv_writer.writerow(tab_list)
    for i in range(stations_result.__len__()//time_steps):
        station=stations_result[i*time_steps]
        station_name=station.split('（')[0][station.find('min')+3:]
        station_min=[]
        station_RMSE=[]
        station_MAE=[]
        station_MAPE = []
        for j in range(time_steps):
            station = stations_result[i * time_steps+j]
            station_min.append(station[:station.find('min')])
            station_RMSE.append(station[station.find('MSE')+5:station.find('MSE')+12])
            station_MAE.append(station[station.find('MAE')+5:station.find('MAE')+12])
            station_MAPE.append(station[station.find('MAPE') + 6:station.find('MAPE') + 13])
        station_list=[station_name]
        for k in range(time_steps):
            station_list.append(station_RMSE[k])
        for k in range(time_steps):
            station_list.append(station_MAE[k])
        for k in range(time_steps):
            station_list.append(station_MAPE[k])
        csv_writer.writerow(station_list)
    f.close()
    avg=['mean']
    df=pd.read_csv(f.name,encoding='gb2312')
    for i in range(time_steps*3):
        avg.append(round(df.iloc[:,i+1].mean(),6))
    f = open(f.name,'a',encoding='gb2312',newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(avg)
    f.close()