# Contrastive Graph Neural Network Explanation

This is the source code for the paper _Contrastive Graph Neural Network Explanation_

## Required libraries
You can install the required libraries by running:
```shell
pip install -r requirements.txt
```

## Recreating the benchmark dataset
```shell
python gen_dataset.py
```

##Training a GNN model
We use GNNExplainer source code for training the model and generating the explanation.
 
## Explaining the model
You can run different explanation method via `explain.py` script
```shell
python explain.py sensitivity | occlusion | random
```
## Evaluation