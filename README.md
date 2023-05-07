Directed Message Passing Based on Attention for Prediction of Molecular Properties
==================================================================================

Authors: Gong CHEN, Yvon MADAY

D-GATs follow the common framework of MPNNs and explore a bond-level message passing algorithm completely relying on scaled dot-product attention mechanism, which outperforms SOTA baselines on 13/15 molecular property prediction tasks on the MoleculeNet benchmark.

<p align="center"><img src="figure/framework.png" width=80%></p>

Dependencies
------------

To run the code, it requires jupyter, rdkit and pytorch.

We advice the following environment:
```
conda create -n D_GATs python=3.8
conda activate D_GATs
conda install -c anaconda jupyter
conda install -c rdkit rdkit
conda install -c conda-forge pytorch-gpu
```

Utilization
-----------

If you want to pre-train a model, the configuration is specified in config/pretrain_config.json. You only need to use jupyter notebook to open Pre-training.ipynb and to run the code.

If you want to finetune a model, the configuration is specified in config/config.json. You only need to use jupyter notebook to open finetune.ipynb and to run the code.

D-GATs' pre-training strategy
-----------------------------

According to our message passing algorithm, the atom states are updated by directed bond states but independent to the update of bond states. Therefore, the successful recovery of atom features relies on the ability to correctly recover the masked bond features. Hence, we only need to recover atom features in the pre-training stage.

<p align="center"><img src="figure/masked.png" width=80%></p>

D-GATs' performance
-------------------
Results on classification tasks:

<p align="center"><img src="figure/auc.png" width=80%></p>

Results on regression tasks:
<p align="center"><img src="figure/reg.png" width=80%></p>

Citation
--------

Please kindly cite this paper if you use the code.
```
not published yet
```

License
-------

This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/GongCHEN-1995/D-GATs/blob/main/LICENSE) for additional details.
