# Optimal Time Complexity Algorithms for Computing General Random Walk Graph Kernels on Sparse Graphs

General Random Walk kernels (RWK) take O($N^6$) time complexity and special instances of this kernel take O($N^3$) time. In this work we present graph voyagers (GVoys): the first O($N$) algorithm for the unbiased approximation of general RWKs, for labelled and unlabelled graphs. GVoys first use the graph random features (Choromanski 2023, Reid et al. 2024) to reduce the time complexity to O($N^2$) and then simulate dependent walks on graphs $G_1$ and $G_2$ by introducing extra random variables to sample walks on the product graph but without storing it in memory, further reducing time complexity as well as storage complexity  (see Figure below).


<p align="center">
<img src="https://github.com/arijitthegame/efficient_random_walk_kernel/blob/main/sim-double-walk.jpg"  width="800px"/>
</p>

## Installation
```bash
git clone https://github.com/arijitthegame/efficient_random_walk_kernel.git
cd efficient_random_walk_kernel
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
pip3 install -e . --user
```

## Getting Started
All code resides in the `src` folder.  We have provided the following : 
- `baseline_methods` : Various methods to compute the RWK kernel mainly code copied from various repos like GraKel
- `gvoys` : Implementation of the proposed GVoy method
- `graph_classification` : example usage of GVoy for graph classifcation on MUTAG but the dataset can be changed to use any dataset from TUDataset


## Citation
If you find our work useful, please cite : 

```bibtex
@inproceedings{
choromanski2025optimal,
title={Optimal Time Complexity Algorithms for Computing General Random Walk Graph Kernels on Sparse Graphs},
author={Krzysztof Marcin Choromanski and Isaac Reid and Kumar Avinava Dubey and Arijit Sehanobish},
booktitle={The 28th International Conference on Artificial Intelligence and Statistics},
year={2025},
url={https://openreview.net/forum?id=NF1WK6BTRZ}
}

```