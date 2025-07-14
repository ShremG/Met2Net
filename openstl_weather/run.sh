export CUDA_VISIBLE_DEVICES="1"
nohup python tools/train.py -d weather_mv5my_12_5_625 -c configs/weather/mv5my_5_625/its.py --ex_name weatherbench/weather_mv5_5_625/its > train_1.log &
