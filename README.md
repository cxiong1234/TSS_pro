

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

