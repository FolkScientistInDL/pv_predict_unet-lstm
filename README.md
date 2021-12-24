# pv_predict_unet-lstm
Code for "Intra-hour Photovoltaic Generation Forecasting based on Multi-source Data and Deep Learning Methods." IEEE Transactions on Sustainable Energy.

Folder structure of the project:

```
pv_predict_unet-lstm
├── ...
├── location.txt
├── unet_pretain.pt
├── station_npy_dir
│   ├── station_data_xxx.npy
│   ├── station_data_xxx.npy
│   ├── station_data_xxx.npy
│   ├── ...
├── swr_dir
│   ├── xxx_swr.npy
│   ├── xxx_swr.npy
│   ├── xxx_swr.npy
│   ├── ...
├── ...
```

Training Example： 

	python unet+lstm_main.py --station_location_txtdir /path/to/txt.txt --station_datadir /path/to/npydir --swr_dir /path/to/swr/dir path/to/npy

