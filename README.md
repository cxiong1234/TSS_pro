

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
Training The model (for demo training data `conditioned_traj_dataset_5snapshots_wrapped.pt`, please  download from google drive https://drive.google.com/file/d/1uAUwlbuW4GeOLCWwT2vvxmWUoxOW5P02/view?usp=sharing):
   ```
   python training.py --tensor_file=<your training data .pt tensorfile> --batch-size=100 
   ```

Consecutive sampling (before sample, download our ubiquitin_5step.ckpt model file in the zenodo: https://zenodo.org/records/17064407):
   ```
   python long_traj_cond_sample.py --ckpt=./demoInput/ubiquitin_5step.ckpt --max_sampling_cycle=1000 --ref-path=./demoInput/ref.npy
   ```


## Reference
Please refer to ChemRxiv:

