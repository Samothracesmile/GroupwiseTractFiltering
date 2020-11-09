'''
Date: July 30, 2020


'''
import os
import glob
import time
import json

import numpy as np
import argparse
import multiprocessing

from sklearn import neighbors
from scipy import spatial
from scipy.spatial import distance_matrix
from itertools import chain
from datetime import datetime
import cpuinfo
# import matplotlib.pyplot as plt
import copy

from fibersup import *

####################################################### functions ####################################################### 

def fastknn(ref_trk, target_trk, downsampling_rate=10, k_pts=500, k_streamlines=10, batch_size=100,mode_flag=1):

    if batch_size >= len(ref_trk):
        ref_streamlines = ref_trk.downsampleTract(downsampling_rate)
    else:
        random.seed(0)
        random_intlist = random.sample(range(len(ref_trk)), batch_size)
        ref_streamlines = [ref_trk.streamlines[i][::downsampling_rate,:] for i in random_intlist]    
        
    target_streamlines = target_trk.downsampleTract(downsampling_rate)

    ref_streamlines_flat = np.vstack(ref_streamlines)
    target_streamlines_flat = np.vstack(target_streamlines)

    # using scipy cKDTree
    tree = spatial.cKDTree(ref_streamlines_flat, leafsize=100) # on one thread
    cdist, cind = tree.query(target_streamlines_flat, k=k_pts) 

    # use concatenate for pt2streamline idex in reference streamlines
    pt2streamline = np.concatenate([i*np.ones(len(streamline),dtype=int) for i,streamline in enumerate(ref_streamlines)]) #30.2 ms ± 2.4 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)

    # the demone
    if batch_size < len(ref_trk):
        pt2streamline = np.array([random_intlist[i] for i in pt2streamline])
    
    kNN_streamlines_index = pt2streamline[cind]
    
    # extract the concatenate index in target streamlines
    length_list = [len(streamline) for streamline in target_streamlines]
    streamlins_num = len(length_list)
    c_length_list = [0] + [sum(length_list[:i+1]) for i in range(streamlins_num)]
    streamline_indexs = [[c_length_list[i],c_length_list[i+1]] for i in range(streamlins_num)]
    
    pwdist_mat_list = []
    for s_index in range(len(target_streamlines)):
        kNN_pt2streamline_index = kNN_streamlines_index[streamline_indexs[s_index][0]:streamline_indexs[s_index][1],:]

        streamline_kNN_pt2streamline_index = np.concatenate([np.unique(pt_kNN_pt2streamline_index) for pt_kNN_pt2streamline_index in kNN_pt2streamline_index])
        streamline2streamline_index, streamline2streamline_counts = np.unique(streamline_kNN_pt2streamline_index, return_counts = True)

        # select k streamline from reference streamlines according to count hits
        ktop_streamline_index_count = np.argsort(streamline2streamline_counts)[-k_streamlines:] # large count first 

        kNN_streamline_index_count = streamline2streamline_index[ktop_streamline_index_count] #
        pwdist_mat_list.append(kNN_streamline_index_count)
        
    return np.vstack(pwdist_mat_list)


 
def merge_ref_dmc(cache_ref_dmc_list,new_ref_dmc_list):
    merged_ref_list = []
    for sl_cache_ref_dmc,sl_new_ref_dmc in zip(cache_ref_dmc_list,new_ref_dmc_list):
    
        merged_ref_list.append(np.array([cache_col if np.mean(cache_col) < np.mean(new_col) else new_col for cache_col,new_col in zip(sl_cache_ref_dmc.T,sl_new_ref_dmc.T)]).T)
        
    return merged_ref_list



def tract_estimate_affsum(que_tract):
    
    def group_aff_dmc(ref_dmc_list):
        if group_num < len(ref_boxs):
            comN_ref_dmc_list = []
            for ref_dmc in ref_dmc_list:
                ref_sum_distance_list = [np.sum(ref_dmc[:,ref_num*ref_sub_index:ref_num*(ref_sub_index+1)]) for ref_sub_index in range(len(ref_boxs))]
                comSubIndexs = np.argsort(ref_sum_distance_list)[:group_num]
                comN_ref_dmc = np.concatenate([ref_dmc[:,ref_num*ref_sub_index:ref_num*(ref_sub_index+1)] for ref_sub_index in comSubIndexs],axis=1)
                comN_ref_dmc_list.append(comN_ref_dmc)
            return comN_ref_dmc_list
        else:
            return ref_dmc_list
          
    
    que_bundle = que_tract.streamlines
    ref_boxs = []
    
    ####################################################################
    # step1 construct reference boxs (in merged_ref_box) 
    # containing the reference set 
    # (with multiple subject information) for each streamlines in que_tract
    
    for ref_tract in org_warped_tract_list:
        if ref_tract.fname != que_tract.fname:
            ref_bundle = ref_tract.streamlines

            ref_streamline_index =  fastknn(ref_tract, que_tract, downsampling_rate=downsampling_rate, batch_size=batch_size, k_pts=k_pts, k_streamlines=k_streamlines)
            ref_box = [[ref_bundle[ref_index] for ref_index in que_str_index[:ref_num]] for que_str_index in ref_streamline_index] 

            ref_boxs.append(ref_box)

    merged_ref_box = [list(chain.from_iterable(x)) for x in zip(*ref_boxs)]

    ####################################################################
    # step2 calculate the pointwise distance for each point 
    # Implementation 1: comparing the distance with previous calculated distance (for random sample objects)
    # Implementation 2: only selecting the top N subject's reference to depict the local commonness
    
    ref_dmc_list = []

    for qs1,streamline_ref_box in zip(que_tract.streamlines, merged_ref_box):
        # cdist
        ref_dmc = np.vstack([np.min(spatial.distance.cdist(qs1,rs,'euclidean'),axis=1) for rs in streamline_ref_box])

        ref_dmc_list.append(ref_dmc.T)

    # Implementation 1
    if que_tract.cache_ref_dmc:
        ref_dmc_list = merge_ref_dmc(que_tract.cache_ref_dmc,ref_dmc_list)

    if cache_flag:
        que_tract.cache_ref_dmc = ref_dmc_list
    
    # Implementation 2
    comN_ref_dmc_list = group_aff_dmc(ref_dmc_list)
    
    # update the que_tract's affsums 
    ref_affsum_list = []
    for ref_dmc_T in comN_ref_dmc_list:
        ref_dmc = ref_dmc_T.T
        ref_affsum = np.mean(np.exp(-1*ref_dmc**2/(sigma**2)), axis=0) + 10**-15
        ref_affsum_list.append(ref_affsum)
        
    que_tract.affsums = ref_affsum_list
    
    return que_tract

######################################################## Parameters ##############################################

parser = argparse.ArgumentParser()

# data set setting
parser.add_argument('--base_dir', type=str, default='../data_general/CST')

# common streamline matching setting
parser.add_argument('--downsampling_rate', type=int, default=3, help='Downsampling rate of point on streamlines')
parser.add_argument('--k_pts', type=int, default=50, help='Number of NN pts')
parser.add_argument('--k_streamlines', type=int, default=3, help='Number of NN streamlines')

# optional parameters
parser.add_argument('--cache_flag', type=int, default=0, help='cache the distance calculation in each iteration')
parser.add_argument('--batch_size', type=int, default=200, help='Size of random selected streamlines in each bundle')

# trimming setting
parser.add_argument('--sigma', type=int, default=8, help='sigma')
parser.add_argument('--length_min_perc', type=float, default=0.6, help='Min Length Percentage Requirement')
parser.add_argument('--length_max_perc', type=float, default=0.1, help='Max Length Percentage Requirement')
parser.add_argument('--group_perc', type=float, default=1, help='Group size')

# for termination
parser.add_argument('--max_iteration', type=int, default=20, help='Max iteration number')
parser.add_argument('--distance_termination', type=float, default=3, help='Termination distance')

# computation setting
parser.add_argument('--cpu_num', type=int, default=6, help='Number of cpu threads')

# results
parser.add_argument('--res_dir', type=str, default='')


args = parser.parse_args()
print('Args Setting:', args)

# Parameter transfer
base_dir = args.base_dir

downsampling_rate = args.downsampling_rate
k_pts = args.k_pts
k_streamlines = args.k_streamlines

cache_flag = args.cache_flag
batch_size = args.batch_size

sigma = args.sigma
length_min_perc = args.length_min_perc
length_max_perc = args.length_max_perc
group_perc = args.group_perc


max_iteration = args.max_iteration
distance_termination = args.distance_termination

cpu_num = args.cpu_num
res_dir = args.res_dir

# timing
ostime = time.time()


######################################### Load the data set #########################################
warped_dir = base_dir + '/warped/*.trk'
trk_list = sorted(glob.glob(warped_dir))
warped_tract_list = [Tract(trk) for trk in trk_list if len(Tract(trk)) > 50]
org_warped_tract_list = [Tract(trk) for trk in trk_list if len(Tract(trk)) > 50]

org_unwarped_dir = base_dir + '/unwarped/*.trk'
unwarped_tract_fname_list = sorted(glob.glob(org_unwarped_dir))
unwarped_tract_list = [Tract(trk) for trk in unwarped_tract_fname_list if len(Tract(trk)) > 50]

print(30*'*',"\nLoaded cst data set! ")
print('Total number of subjects {}'.format(len(warped_tract_list)))

ref_num = k_streamlines  # set the reference number = k NN streamlines

print(30*'*',"\nTrimming ... ")


######################################################## Iterative Trimming ###################################################

# copy for untrimmed
for warped_tract, org_warped_tract, unwarped_tract in zip(warped_tract_list, org_warped_tract_list, unwarped_tract_list):

    warped_tract.streamlines_untrimmed = copy.deepcopy(warped_tract.streamlines)
    org_warped_tract.streamlines_untrimmed = copy.deepcopy(org_warped_tract.streamlines)
    unwarped_tract.streamlines_untrimmed = copy.deepcopy(unwarped_tract.streamlines)

# set the lengthD according to lengthD percentage
mean_point_on_streamline = np.mean([len(sl) for unwarped_tract in unwarped_tract_list for sl in unwarped_tract.streamlines])

lengthD = length_min_perc*mean_point_on_streamline
lengthE = max(round(length_max_perc*mean_point_on_streamline),1)
group_num = min(int(group_perc*len(unwarped_tract_list)),len(unwarped_tract_list)-1)

# set pool for multiprocessing
pool = multiprocessing.Pool(processes=cpu_num) 

# Trimming Iterations
max_mean_pt_distance_record = [] # debug
streamline_number_record = [] # debug

for itera in range(1,max_iteration):
    # Trimming Iterations
    stime = time.time()

    # nonparallel implementation
    streamline_number_record.append(np.array([len(warped_tract) for warped_tract in warped_tract_list]))
    
    # affinity sum estimation
    if cpu_num > 1:
        # parallel
        warped_tract_list = pool.map(tract_estimate_affsum, (que_tract for que_tract in warped_tract_list))
        for que_tract,unwarped_que_tract in zip(warped_tract_list,unwarped_tract_list):
            unwarped_que_tract.affsums = que_tract.affsums
    else:
        for que_tract,unwarped_que_tract in zip(warped_tract_list,unwarped_tract_list):
            que_tract = tract_estimate_affsum(que_tract)
            unwarped_que_tract.affsums = que_tract.affsums
            
    max_mean_pt_distance = max([max([np.mean(np.sqrt(-np.log(tract_affsum)*sigma**2)) for tract_affsum in unwarped_que_tract.affsums]) for unwarped_que_tract in unwarped_tract_list])  
    
    max_mean_pt_distance_record.append(max_mean_pt_distance) # debug

    print('After iteration {0}, max mean point distance reduces to {1:.4f}'.format(str(itera-1),max_mean_pt_distance))

    if max_mean_pt_distance < distance_termination:
        break

    # Termination creteria not meet, process continue 
    All_fttaffsum = np.concatenate([np.concatenate(que_tract.affsums) for que_tract in warped_tract_list if que_tract.affsums])
    Tha = max(np.mean(All_fttaffsum) - 2*np.std(All_fttaffsum),0.01)   

    print(f'Mean affsum: {np.mean(All_fttaffsum):.4f}')
    print(f'Std affsum: {np.std(All_fttaffsum):.4f}')

    print(f'Total number of points: ', All_fttaffsum.shape[0])
    print(f'Nnumber of trimmed points: ', All_fttaffsum[All_fttaffsum < Tha].shape[0])   

    # Fiber filtering
    for que_tract,unwarped_que_tract in zip(warped_tract_list,unwarped_tract_list):
        que_tract.streamline_label_update(Tha, lengthD=lengthD, lengthE=lengthE)
        unwarped_que_tract.streamline_label_update(Tha, lengthD=lengthD, lengthE=lengthE)

    print('Iteration {} takes: '.format(str(itera)), time.time() - stime)
    print(30*'*')

# print('streamline_number_record: ',streamline_number_record)


################################################# save the finnal results ################################################
for unwarped_que_tract in unwarped_tract_list:

    # extract fname pattern
    que_tract_fname = os.path.basename(unwarped_que_tract.fname)

    # save filtering results
    save_fname = os.path.join(res_dir, que_tract_fname.replace('.trk', '_res.trk'))
    # print(save_fname)
    save_tract(unwarped_que_tract.streamlines, save_fname, unwarped_que_tract.hdr)   

    # save untrimmed results
    save_fname = save_fname.replace('.trk','_untrimmed.trk')
    # print(save_fname)
    save_tract(unwarped_que_tract.streamlines_untrimmed, save_fname, unwarped_que_tract.hdr) 


################################################ save the experiement log ######################################################
log_file = os.path.join(res_dir, "args.txt")

with open(log_file, 'w') as f:
    json.dump(args.__dict__, f, indent=2)


    for iter_num, (distance, sl_num) in enumerate(zip(max_mean_pt_distance_record,streamline_number_record)):
        str_divider = '\n',30*'*','\n'
        f.write('{}Iter {} '.format(str_divider,iter_num + 1))
        f.write('distance {} '.format(distance))
        # f.write('sl_num mean {0}, std {1} \n'.format(np.mean(sl_num),np.std(sl_num)))

    f.write('Total time: {}.'.format(time.time() - ostime))


# f = open(os.path.dirname(save_fname)+"/exlog.txt","a+")

# f.write('The total number of cpus {}\n'.format(cpu_num))
# # f.write(cpuinfo.get_cpu_info()['brand'])
# f.write(cpuinfo.get_cpu_info()['brand_raw'])

# f.write('\n')
# f.write('cache_flag {}\n'.format(cache_flag))
# f.write('downsampling_rate {}\n'.format(downsampling_rate))
# f.write('k_pts {}\n'.format(k_pts))
# f.write('sigma {}\n'.format(sigma))
# f.write('k_streamlines {}\n'.format(k_streamlines))
# f.write('ref_num {}\n'.format(ref_num))
# f.write('batch_size {}\n'.format(batch_size))

# f.write('length_min_perc {}\n'.format(length_min_perc))
# f.write('lengthD {}\n'.format(lengthD))

# f.write('lengthE {}\n'.format(lengthE))
# f.write('length_max_perc {}\n'.format(length_max_perc))

# f.write('group_num {}\n'.format(group_num))
# f.write('group_perc {}\n'.format(group_perc))


# f.write('max_iteration {}\n'.format(max_iteration))
# f.write('distance_termination {}\n'.format(distance_termination))

# for iter_num, (distance, sl_num) in enumerate(zip(max_mean_pt_distance_record,streamline_number_record)):
# #     f.write('iter_num {} '.format(iter_num))
# #     f.write('distance {} '.format(distance))
# #     f.write('sl_num {} \n'.format(sl_num))
#     f.write('iter_num {} '.format(iter_num + 1))
#     f.write('distance {} '.format(distance))
#     f.write('sl_num mean {0}, std {1} \n'.format(np.mean(sl_num),np.std(sl_num)))

# f.write('Total time: {}.'.format(time.time() - ostime))

# f.close()