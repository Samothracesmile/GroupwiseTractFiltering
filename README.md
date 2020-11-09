# GroupwiseTractFiltering

This repository constains the implementation of track filtering method presented in: Xia, Yihao, and Yonggang Shi. "**Groupwise track filtering via iterative message passing and pruning**." NeuroImage 221 (2020): 117147.

https://www.sciencedirect.com/science/article/pii/S1053811920306339



## 1. Code setup and Requirement
Python packages such as numpy, scipy, nibabel are required for this framework. The code are tested with recent versions of numpy==1.16.4, scipy==1.2.1, nibabel==2.3.3 in python==3.6.4. Users may install these packages by running following command in terminal:

```
pip install -r requirements.txt
```

## 2. Data Preparation
The inputs of this tract filtering framework are fiber bundles (TrackVis .trk files) and their counterparts warped in a common space. 

### Example Data
The examples provided along with the framework are 20 left cortiospinal tracts from the the Human Connectome Project (HCP) data. Each tract containing 500 streamlines which are reconstructed by using the probabilistic tractography tool in MRTrix3. Tracts are non-linearly warped to MNI space by using Advanced Normalization Tools (ANTs). The fiber bundles in subject space and the warped counterparts are located at **./data/cst/unwarped** and **./data/cst/warped** respectively. 

## 3. Running the code
To extract the group-wise consistent sub-bundle structures of provided examples with default settings, user can run following code:

```
cd ./code
bash run_fast_trimming.sh
```

The filtering results will be save in folder **./data/cst/res_test**

Users can indicate specific parameters via command-line arguments:

* **data path setting**
  * --base_dir: directory for input data
  * --res_dir: directory for filtering results

* **common streamline matching setting**
  * --downsampling_rate: downsampling rate of points on streamlines.
  * --k_streamlines: reference set size

* **trimming setting**
  * --batch_size: batch size for computational simplification of streamline to streamline distance. The bundle subsampling rate(r) = bath size / average streamline number in the bundle
  * --sigma: distance scale parameter
  * --length_min_perc: minimum length requirement ensuring the extracted sub-bundle structures are anatomical meaningful.
  * --length_max_perc: maximum length limitation indicating the tolerance of local inconsistency.
  * --group_perc: affinity rate, the affinity parameter(K) = affinity rate * the total number of subjects in the dataset

* **termination setting**
  * --max_iteration: maximum iteration number
  * --distance_termination: proximity parameter for termination

* **computation setting**
  * --cpu_num: cup threads using for parallel computing


