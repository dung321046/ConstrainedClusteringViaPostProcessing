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

## Running pairwise, cluster-size and attribute-level costraints experiments

### Step 0: Setup 
```
export PYTHONPATH="./"
```

Download:

[IDEC weights](https://drive.google.com/drive/folders/1hJ7Dvwo_4GYgslaqL7-TlHzQGaV8Kp2j?usp=sharing)

#### Step 1: Generate dataset

Generate pairwise and cluster-size contraints

```
python generate_testsets/generate-pw-csize.py --data $DATA
```



#### Step 2: Run models
```
python models/encode-kmeans-post.py --data $DATA
python models/pw-csize-ilp.py --data $DATA --csize True
```


## Running Fairness Experiments

