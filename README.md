# pv_predict_unet-lstm
Code for "Intra-hour Photovoltaic Generation Forecasting based on Multi-source Data and Deep Learning Methods." IEEE Transactions on Sustainable Energy.



Download the pre-train model from  https://drive.google.com/file/d/1Ux5ohVJceRteBfd9NpJv02dXn9kPikVi/view?usp=sharing and then put it in the root path of this project. 

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

# Cite Paper:

Plain text version:

T. Yao et al., "Intra-Hour Photovoltaic Generation Forecasting Based on Multi-Source Data and Deep Learning Methods," in IEEE Transactions on Sustainable Energy, vol. 13, no. 1, pp. 607-618, Jan. 2022, doi: 10.1109/TSTE.2021.3123337.

or BibTeX version:

@ARTICLE{9591417,  author={Yao, Tiechui and Wang, Jue and Wu, Haoyan and Zhang, Pei and Li, Shigang and Xu, Ke and Liu, Xiaoyan and Chi, Xuebin},  journal={IEEE Transactions on Sustainable Energy},   title={Intra-Hour Photovoltaic Generation Forecasting Based on Multi-Source Data and Deep Learning Methods},   year={2022},  volume={13},  number={1},  pages={607-618},  doi={10.1109/TSTE.2021.3123337}}

