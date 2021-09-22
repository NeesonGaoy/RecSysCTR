###
 # @Author: sysu.gaoyong
 # @Email: ygaoneeson@gmail.com
 # @Date: 2021-09-22 11:45:57
 # @LastEditTime: 2021-09-22 17:47:44
 # @LastEditors: sysu.gaoyong
 # @FilePath: /RecSysCTR/run_base.sh
 # Copyright (c) 2011 Neeson.GaoYong All rights reserved.
### 
source activate py3.7


python main_run.py \
--config_path "/Users/gaoyong/Desktop/RecSysCTR/config/" \
--data_config "data_config/criteo678w.yaml" \
--model_config "model_config/DCN.yaml" \
--train_config "train_config/criteo678w_dcn.yaml" \
--batch_size 256 \
--lr 5e-5 \
--log_path "/Users/gaoyong/Desktop/RecSysCTR/logs/"\
--log_name "criteo678w_dcn_v1.log"
