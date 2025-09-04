# Conditional diffusion-based protein trajectory sampling

### Abstract

Understanding how protein structures dictate their diverse biological functions remains one of the central and enduring challenges in structural biology. The development of AlphaFold and ESMAtlas marks a significant advance in protein science, enabling the reliable prediction of protein structure directly from amino acid sequence. This advance in structure prediction underscores the need for complementary methods that can explore conformational space and enable efficient sampling of dynamic trajectories. Here, we present TSS-Pro, a conditional generative diffusion framework that enables efficient sampling of protein conformational trajectory space. TSS-Pro takes the initial frame as conditional input and generates protein conformational trajectories. It supports two sampling strategies: (1) Consecutive sampling, where each trajectory segment is generated step-by-step by conditioning on the final frame of the previously predicted segment, enabling temporally coherent propagation of structural transitions; (2) Parallel sampling, where multiple trajectory branches are independently generated from initial conditions to enhance conformational diversity. We validate TSS-Pro on three representative systems of increasing complexity: alanine dipeptide, ubiquitin, and Drosophila cryptochrome (dCRY). TSS-Pro reproduces the free energy landscape of alanine dipeptide. In the case of ubiquitin, consecutive sampling with TSS-Pro overcomes local minima and uncovers distinct conformational states of the C-terminal region. For the large protein dCRY, TSS-Pro achieves high efficiency through parallel trajectory sampling, enabling conformational and dynamic exploration typically accessible only through extensive simulations. TSS-Pro paves the way for high-throughput exploration of protein trajectories and conformational landscapes for large and complex systems.


### TSS-Pro Architecture

![figure](./Figures/architecture.svg)

## System requires

The software package can be installed and runned on Linux, Windows, and MacOS (x86_64)

Dependency of Python and Python packages: 

(versions that has been previously tested on are also listed below, other versions should work the same)

```bash
python == 3.9
numpy == 1.26.1
scipy == 1.11.4
torch == 1.13.1
tqdm == 4.66.1
```
The required python packages with the latest versions will be automatically installed if these python packages are not already present in your local Python environment.

## Installation from sources

The source code can be installed with a local clone:

The most time-consuming step is the installation of PyTorch (especially cuda version) and the whole installation procedure takes around 5 mins to complete at a local desktop.

```bash
git clone https://github.com/xuhuihuang/ts-dar.git
```

```bash
python -m pip install ./ts-dar
```

## Quick start

### Note

Our python package name is currently tsdart.

### Start with jupyter notebook

Check these two files for the demo:

```
./ts-dar/example/muller-example.ipynb
```

```
./ts-dar/example/quadruple-well-example.ipynb
```

### Start with python script (Linux)

The whole training procedure of the following demo on i9-10900k cpu takes around 30mins to complete at a local desktop.

```sh
python ./ts-dar/scripts/train_tsdart.py \
    --seed 1 \
    --device 'cpu' \
    --lag_time 10 \
    --encoder_sizes 2 20 20 20 10 2 \
    --feat_dim 2 \
    --n_states 2 \
    --beta 0.01 \
    --gamma 1 \
    --proto_update_factor 0.5 \
    --scaling_temperature 0.1 \
    --learning_rate 0.001 \
    --pretrain 10 \
    --n_epochs 20 \
    --train_split 0.9 \
    --train_batch_size 1000 \
    --data_directory ./ts-dar/data/quadruple-well \
    --saving_directory . 
```

Or
```
sh ./ts-dar/scripts/train_tsdart.sh
```

## Compiling Document Environment
Once you have already installed ts-dar in your conda environment. 
```bash
python -m pip install -U sphinx
pip install sphinx-rtd-theme
pip install nbconvert nbformat
pip install sphinx-design
cd docs
make html
```
(Warnings can be ignored!)
You can also visit our [documentation online](https://bojunliu0818.github.io/ts-dart-doc/html/index.html)

## More instructions 

TS-DAR refers to the preprint [10.26434/chemrxiv-2024-r8gjv](https://chemrxiv.org/engage/chemrxiv/article-details/65adf0b966c1381729fb4c11).

We already added the example of Muller potential reported in this preprint to the repo. 

To reproduce the results of the other datasets reported in this preprint, please refer to [Zenodo](https://zenodo.org/records/13835580), where we have uploaded all of our training results and raw simulation data. Or you can directly contact bliu293@wisc.edu.

## Reference

Our codebase builds heavily on
- [https://github.com/deeplearning-wisc/cider](https://github.com/deeplearning-wisc/cider)
- [https://github.com/deeptime-ml/deeptime](https://github.com/deeptime-ml/deeptime)

Thanks for open-sourcing!

[Go to Top](#Abstract)
