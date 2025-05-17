export CUDA_VISIBLE_DEVICES="1"
# nohup python tools/train.py -d weather_mv5my_12_5_625 -c configs/weather/mv5my_5_625/its.py --ex_name weather_mv5_5_625/debug > train_1.log &
# nohup python tools/train.py -d weather_mv8my_12_5_625 -c configs/weather/mv8my_5_625/TAU.py --ex_name weather_mv8_5_625/tau -b 16  > train_0.log &
# nohup python tools/train.py -d weather_mv8my_12_5_625 -c configs/weather/mv8my_5_625/its.py --ex_name weather_mv8_5_625/tau_32_256_lr5e-4_b8 -b 8 > train_1.log &
nohup python tools/train.py -d weather_mv6my_12_5_625 -c configs/weather/mv6my_5_625/its.py --ex_name weather_mv6_5_625/tau_32_256_lr1e-4_b8 -b 8 > train_1.log &
# nohup python tools/train.py -d weather_mv9my_12_5_625 -c configs/weather/mv9my_5_625/its.py --ex_name weather_mv9_5_625/tau_32_256_lr2.5e-4_b4 -b 4 > train_0.log &