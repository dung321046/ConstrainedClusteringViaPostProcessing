# Code for ECAI 2020: Constrained Clustering Via Post Processing
 
## Introduction


The source for IDEC and DCC implementation is here [Deep Constrained Clustering](https://github.com/blueocean92/deep_constrained_clustering).
## Installation


#### Step 1: Clone the Code from Github
```
git clone git@github.com:dung321046/ConstrainedClusteringViaPostProcessing.git
cd ConstrainedClusteringViaPostProcessing/
```
### Step 2: Install Requirements

Install conda 

Install libs

```
conda install --yes --file requirements.txt 
```

Install gurobipy

```
conda config --add channels http://conda.anaconda.org/gurobi
conda install gurobi
```

## Running Constrained Clustering Experiments

```
export PYTHONPATH="./"
```
#### Step 1: Generate dataset

Generate pairwise and cluster-size contraints

```
python generate_testsets/generate-pw-csize.py --data $DATA
```



#### Step 2: Run models
```
python models/kmeans-post.py
```

#### Step 2: Run reports

