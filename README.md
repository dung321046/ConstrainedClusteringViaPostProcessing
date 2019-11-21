# Code for ECAI 2020: Constrained Clustering Via Post Processing
 
## Introduction

Constrained clustering has been studied almost twenty years ago. However, there are still room for research.

This work proposed the novel approach by converting the output of many clustering algorithm to a cluster fractional allocation matrix (CFAM).  Then, takes it as an optimization problem with requirement to statisfy (combined) different types of  constraints.


Moreover, we also shows practical application namely the attribute-level constraint and the (combine) fairness constraint.  
  

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

More details for IDEC and DCC is here [Deep Constrained Clustering](https://github.com/blueocean92/deep_constrained_clustering).

#### Step 1: Generate dataset

Generate pairwise and cluster-size contraints

```
python generate_testsets/generate-pw-csize.py --data $DATA
```



#### Step 2: Run models
```
python models/encode-kmeans-post.py --data $DATA
python models/pw-csize-ilp.py --data $DATA --csize True
python models/attribute-level-ilp.py
```


## Running Fairness Experiments

#### Step 1: Generate neighbor set for individual fairness constraints

```
python fairness/generate_neighbor.py 
```

### Step 2: Run individual and/or group fairness

```
python fairness/run_individual_fairness.py
python fairness/run_group_fairness.py
python fairness/run_combine_fairness.py
```