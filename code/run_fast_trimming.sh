#!/bin/bash

python fast_trimming.py --base_dir ../data/cst \
                    --batch_size 100 --group_perc 1 --cache_flag 0 \
                    --cpu_num 4 --length_min_perc 0.8 --length_max_perc 0.01 --sigma 8 \
                    --distance_termination 3 --res_dir ../data/cst/res_test
                    
