# Contrastive Graph Neural Network Explanation

This is the source code for the paper _Contrastive Graph Neural Network Explanation_

## Required libraries
You can install the required libraries by running:
```shell
pip install -r requirements.txt
```

## Recreating the benchmark dataset
Run the following command to recreate the datasets
```shell
python gen_dataset.py DATASET_NAME
```
`DATASET_NAME` can be one of the followings:
* `CYCLIQ`: Same as the dataset mentioned in the paper
* `CYCLIQ-MULTI`: 2 additional classes compared to `CYCLIQ`. One class is base trees without any attachment and the other is base trees with both cycles and cliques.
* `TRISQ`: 1000 base trees with attached triangles and 1000 base trees with attached squares (cycles of length 4)
* `HOUSE_CLIQ`: 1000 base trees with attached cliques of size 5 and 1000 base trees with attached [house graphs](https://mathworld.wolfram.com/HouseGraph.html)

You can add a `--sample-size` option to change the number of samples for each label (default is 1000).


## Training a GNN model
We use GNNExplainer source code for training the model, making node embeddings, and generating the explanation.  
Any file or folder with the prefix of `gnnexplainer` is ported from [GNNExplainer repo](https://github.com/RexYing/gnn-model-explainer) with small modifications to make it compatible with our framework.  
To train a model, you can run the following command: 
`python gnnexplainer_train.py --bmname=DATASET_NAME --epochs=20`
The following command will output a model in `ckpt` folder which you will use in subsequent commands.

## Generating node embeddings and GNNExplainer output
Run the following command for generating node embeddings for all graphs as well as GNNExplainer explanation
`python gnnexplainer_main.py --bmname=DATASET_NAME --graph-mode --explain-all`
This will create a new folder with name `embeddings-DATASET_NAME` and explanations in `explanations/gnnexplainer` folder.
 
## Running other explanation methods
You can run all the other explanation method via `explain.py` script
```shell
python explain.py contrast | sensitivity | occlusion | random
```
Use `--help` option to see all the available options for each command. For example `python explain.py contrast --help`

## Evaluation
Run the following command to see the accuracy of explanations
```shell
python evaluate.py DATASET_PATH EXPLAIN_PATH
```

## Complete example
Here is the complete list of commands needed to reproduce the paper results:
```sh
python gen_dataset.py CYCLIQ

python gnnexplainer_train.py --bmname=CYCLIQ --epochs=100
python gnnexplainer_main.py --bmname=CYCLIQ --graph-mode --explain-all

python explain.py random data/CYCLIQ/ explanations/random
python explain.py sensitivity data/CYCLIQ/ ckpt/HOUSE_CLIQ_base_h20_o20.pth.tar explanations/sensitivity
python explain.py occlusion data/CYCLIQ/ ckpt/HOUSE_CLIQ_base_h20_o20.pth.tar explanations/occlusion
python explain.py contrast data/CYCLIQ/ embeddings-HOUSE_CLIQ explanations/contrast

python evaluate.py data/CYCLIQ/ explanations/gnnexplainer/
python evaluate.py data/CYCLIQ/ explanations/random/
python evaluate.py data/CYCLIQ/ explanations/sensitivity/
python evaluate.py data/CYCLIQ/ explanations/occlusion/
python evaluate.py data/CYCLIQ/ explanations/contrast/
```

## Visualizing Explanations
You can run the `Visualize.ipynb` notebook for visualizing each method explanation