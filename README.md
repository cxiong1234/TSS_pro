# Conditional diffusion-based protein trajectory sampling

### Abstract

Understanding how protein structures dictate their diverse biological functions remains one of the central and enduring challenges in structural biology. The development of AlphaFold and ESMAtlas marks a significant advance in protein science, enabling the reliable prediction of protein structure directly from amino acid sequence. This advance in structure prediction underscores the need for complementary methods that can explore conformational space and enable efficient sampling of dynamic trajectories. Here, we present TSS-Pro, a conditional generative diffusion framework that enables efficient sampling of protein conformational trajectory space. TSS-Pro takes the initial frame as conditional input and generates protein conformational trajectories. It supports two sampling strategies: (1) Consecutive sampling, where each trajectory segment is generated step-by-step by conditioning on the final frame of the previously predicted segment, enabling temporally coherent propagation of structural transitions; (2) Parallel sampling, where multiple trajectory branches are independently generated from initial conditions to enhance conformational diversity. We validate TSS-Pro on three representative systems of increasing complexity: alanine dipeptide, ubiquitin, and Drosophila cryptochrome (dCRY). TSS-Pro reproduces the free energy landscape of alanine dipeptide. In the case of ubiquitin, consecutive sampling with TSS-Pro overcomes local minima and uncovers distinct conformational states of the C-terminal region. For the large protein dCRY, TSS-Pro achieves high efficiency through parallel trajectory sampling, enabling conformational and dynamic exploration typically accessible only through extensive simulations. TSS-Pro paves the way for high-throughput exploration of protein trajectories and conformational landscapes for large and complex systems.


### TSS-Pro Architecture

![figure](./Figures/architecture.svg)

## Running System

The code was running in the Ubuntu 22.04.5 LTS with the Nvidia A100-SXM4-80GB, provided by the O’Donnell Data Science and Research Computing Institute (ODSRCI) at Southern Methodist University.


## Installation of environment

1. Create the environment from the YAML:
   ```
   conda env create -f diffusionProtein.yml
   ```
2. Activate the env:
   ```
   conda activate diffusionProtein
   ```
3. Install the [PyRosetta](http://pyrosetta.org/downloads) package:
   ```
   pip install pyrosetta-installer
   python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'
   ```
## Demo
Training The model:
   ```
   python training.py --tensor_file=<PATH_TO_YOUR_PT_TRAINING_FILE> --batch-size=128
   ```

Consecutive sampling:
   ```
   python long_traj_cond_sample.py --ckpt==<PATH_TO_YOUR_CKPT_CHECKPOINT_FILE> --max_sampling_cycle=1000 --ref-path=<PATH_TO_YOUR_REF_FRAME>
   ```


## Reference
Please refer to ChemRxiv:

