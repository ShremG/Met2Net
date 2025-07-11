dataset_parameters = {
    'bair': {
        'in_shape': [4, 3, 64, 64],
        'pre_seq_length': 4,
        'aft_seq_length': 12,
        'total_length': 16,
        'metrics': ['mse', 'mae', 'ssim', 'psnr', 'lpips'],
    },
    'mfmnist': {
        'in_shape': [10, 1, 64, 64],
        'pre_seq_length': 10,
        'aft_seq_length': 10,
        'total_length': 20,
        'data_name': 'fmnist',
        'metrics': ['mse', 'mae', 'ssim', 'psnr'],
    },
    'mmnist': {
        'in_shape': [10, 1, 64, 64],
        'pre_seq_length': 10,
        'aft_seq_length': 10,
        'total_length': 20,
        'data_name': 'mnist',
        'metrics': ['mse', 'mae', 'ssim', 'psnr'],
    },
    'mmnist_cifar': {
        'in_shape': [10, 3, 64, 64],
        'pre_seq_length': 10,
        'aft_seq_length': 10,
        'total_length': 20,
        'data_name': 'mnist_cifar',
        'metrics': ['mse', 'mae', 'ssim', 'psnr'],
    },
    'noisymmnist': {
        'in_shape': [10, 1, 64, 64],
        'pre_seq_length': 10,
        'aft_seq_length': 10,
        'total_length': 20,
        'data_name': 'noisymmnist',
        'metrics': ['mse', 'mae', 'ssim', 'psnr'],
    },
    'taxibj': {
        'in_shape': [4, 2, 32, 32],
        'pre_seq_length': 4,
        'aft_seq_length': 4,
        'total_length': 8,
        'metrics': ['mse', 'mae', 'ssim', 'psnr'],
    },
    'human': {
        'in_shape': [4, 3, 256, 256],
        'pre_seq_length': 4,
        'aft_seq_length': 4,
        'total_length': 8,
        'metrics': ['mse', 'mae', 'ssim', 'psnr', 'lpips'],
    },
    **dict.fromkeys(['kth20', 'kth'], {
        'in_shape': [10, 1, 128, 128],
        'pre_seq_length': 10,
        'aft_seq_length': 20,
        'total_length': 30,
        'metrics': ['mse', 'mae', 'ssim', 'psnr', 'lpips'],
    }),
    'kth40': {
        'in_shape': [10, 1, 128, 128],
        'pre_seq_length': 10,
        'aft_seq_length': 40,
        'total_length': 50,
        'metrics': ['mse', 'mae', 'ssim', 'psnr', 'lpips'],
    },
    'kitticaltech': {
        'in_shape': [10, 3, 128, 160],
        'pre_seq_length': 10,
        'aft_seq_length': 1,
        'total_length': 11,
        'metrics': ['mse', 'mae', 'ssim', 'psnr', 'lpips'],
    },
    **dict.fromkeys(['kinetics400', 'kinetics'], {
        'in_shape': [4, 3, 256, 256],
        'pre_seq_length': 4,
        'aft_seq_length': 4,
        'total_length': 8,
        'data_name': 'kinetics400',
        'metrics': ['mse', 'mae', 'ssim', 'psnr', 'lpips'],
    }),
    'kinetics600': {
        'in_shape': [4, 3, 256, 256],
        'pre_seq_length': 4,
        'aft_seq_length': 4,
        'total_length': 8,
        'data_name': 'kinetics600',
        'metrics': ['mse', 'mae', 'ssim', 'psnr', 'lpips'],
    },
    **dict.fromkeys(['weather', 'weather_t2m_5_625'], {  # 2m_temperature
        'in_shape': [12, 1, 32, 64],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 't2m',
        'train_time': ['2010', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'metrics': ['mse', 'rmse', 'mae'],
    }),
    'weather_mv4my_12_5_625': {  # multi-variant weather bench, 12h -> 12h
        'in_shape': [12, 4, 32, 64],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 32,
        'data_name': 'mv4my',
        'train_time': ['2010', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_mv5my_12_5_625': {  # multi-variant weather bench, 12h -> 12h
        'in_shape': [12, 5, 32, 64],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 'mv5my',
        'train_time': ['2010', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_mv6my_12_5_625': {  # multi-variant weather bench, 12h -> 12h
        'in_shape': [12, 6, 32, 64],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 'mv6my',
        'train_time': ['2010', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_mv8my_12_5_625': {  # multi-variant weather bench, 12h -> 12h
        'in_shape': [12, 8, 32, 64],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 'mv8my',
        'train_time': ['2010', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_mv9my_12_5_625': {  # multi-variant weather bench, 12h -> 12h
        'in_shape': [12, 9, 32, 64],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 'mv9my',
        'train_time': ['2010', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_mv5my_12_1_40625': {  # multi-variant weather bench, 12h -> 12h
        'in_shape': [12, 5, 64, 128], # 128 256
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 'mv5my',
        'train_time': ['2010', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_mv_4_28_s6_5_625': {  # multi-variant weather bench, 4->28 (7 days)
        'in_shape': [4, 12, 32, 64],
        'pre_seq_length': 4,
        'aft_seq_length': 28,
        'total_length': 32,
        'data_name': 'mv',
        'train_time': ['1979', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'idx_in': [1+i*6 for i in range(-3, 0)] + [0,],
        'idx_out': [i*6 + 1 for i in range(28)],
        'step': 6,
        'levels': [150, 500, 850],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_mv_4_4_s6_5_625': {  # multi-variant weather bench, 4->4 (1 day)
        'in_shape': [4, 12, 32, 64],
        'pre_seq_length': 4,
        'aft_seq_length': 4,
        'total_length': 8,
        'data_name': 'mv',
        'train_time': ['1979', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'idx_in': [1+i*6 for i in range(-3, 0)] + [0,],
        'idx_out': [i*6 + 1 for i in range(4)],
        'step': 6,
        'levels': [150, 500, 850],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_r_5_625': {  # relative_humidity
        'in_shape': [12, 1, 32, 64],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 'r',
        'train_time': ['2010', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_uv10_5_625': {  # u10+v10, component_of_wind
        'in_shape': [12, 2, 32, 64],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 'uv10',
        'train_time': ['2010', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_tcc_5_625': {  # total_cloud_cover
        'in_shape': [12, 1, 32, 64],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 'tcc',
        'train_time': ['2010', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_t2m_1_40625': {  # relative_humidity
        'in_shape': [12, 1, 64, 128],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 't2m',
        'train_time': ['2010', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_r_1_40625': {  # relative_humidity
        'in_shape': [12, 1, 64, 128],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 'r',
        'levels':['1000'],
        'train_time': ['2010', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_uv10_1_40625': {  # u10+v10, component_of_wind
        'in_shape': [12, 2, 64, 128],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 'uv10',
        'train_time': ['2010', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_tcc_1_40625': {  # total_cloud_cover
        'in_shape': [12, 1, 64, 128],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 'tcc',
        'train_time': ['2010', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'sevir_vis':{
        'in_shape': [13, 1, 768, 768],
        'pre_seq_length': 13,
        'aft_seq_length': 12,
        'total_length': 25,
        'data_name': 'vis', 
        'metrics': ['mse', 'mae', 'pod', 'sucr', 'csi', 'lpips'],
    },
    'sevir_ir069':{
        'in_shape': [13, 1, 192, 192],
        'pre_seq_length': 13,
        'aft_seq_length': 12,
        'total_length': 25,
        'data_name': 'ir069',
        'metrics': ['mse', 'mae', 'pod', 'sucr', 'csi', 'lpips'],
    },
    'sevir_ir107':{
        'in_shape': [13, 1, 192, 192],
        'pre_seq_length': 13,
        'aft_seq_length': 12,
        'total_length': 25,
        'data_name': 'ir107',
        'metrics': ['mse', 'mae', 'pod', 'sucr', 'csi', 'lpips'],
    },
    'sevir_vil':{
        'in_shape': [13, 1, 384, 384],
        'pre_seq_length': 13,
        'aft_seq_length': 12,
        'total_lenght': 25,
        'data_name': 'vil', 
        'metrics': ['mse', 'mae', 'pod', 'sucr', 'csi', 'lpips'],
    },
}