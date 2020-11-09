#!/bin/bash

# cd /home/yihaoxia/Desktop/fast_fiber_trimming_github/code

python fast_trimming.py --base_dir ../data/cst \
                    --batch_size 100 --group_perc 1 --cache_flag 0 \
                    --cpu_num 4 --lengthD_perc 0.8 --lengthE_perc 0.01 --sigma 8 \
                    --distance_termination 3 --res_dir ../data/cst/res_test_20201108