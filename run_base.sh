###
 # @Author: sysu.gaoyong
 # @Email: ygaoneeson@gmail.com
 # @Date: 2021-09-22 11:45:57
 # @LastEditTime: 2021-09-22 11:45:57
 # @LastEditors: sysu.gaoyong
 # @FilePath: /RecSysCTR/run_base.sh
 # Copyright (c) 2011 Neeson.GaoYong All rights reserved.
### 
source activate py3.7

# python main_run_quant.py \
# --data_name criteo20w \
# --emb_type normal \
# --emb_dim 64 \
# --model_name dcn \
# --batch_size 256 \
# --l2_sparse 5e-4 \
# --l2_dense 5e-4 \
# --lr 5e-5


# IRazorDNN
python main_run_quant.py \
--data_name criteo20w \
--emb_type normal \
--emb_dim 64 \
--model_name IRazorDNN \
--batch_size 256 \
--l2_sparse 5e-4 \
--l2_dense 5e-4 \
--lr 5e-5

