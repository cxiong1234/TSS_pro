import pyrosetta
from pyrosetta import pose_from_file
from pyrosetta.rosetta.core.scoring import get_score_function,ScoreType
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.scoring.constraints import CoordinateConstraint, ConstraintSet
from pyrosetta.rosetta.core.id import AtomID
from pyrosetta.rosetta.protocols.constraint_movers import AddConstraintsToCurrentConformationMover
from pyrosetta.rosetta.core.scoring.func import HarmonicFunc
import os

dir = "./"


def refine_conformations(pdb_file,refine_pdb_file):
    pyrosetta.init()
    pose = pose_from_file(pdb_file)
    scorefxn = get_score_function(True)
    scorefxn.set_weight(ScoreType.coordinate_constraint, 0.5)
    constraint_set = ConstraintSet()
    for i in range(1, pose.total_residue() + 1):
        for atom_name in ['N', 'CA', 'C']: 
            if pose.residue(i).has(atom_name):
                atom_id = AtomID(pose.residue(i).atom_index(atom_name), i)
                xyz = pose.xyz(atom_id)
                harmonic_func = HarmonicFunc(0.0, 3.5)
                constraint = CoordinateConstraint(atom_id, atom_id, xyz, harmonic_func)
                constraint_set.add_constraint(constraint)

    pose.constraint_set(constraint_set)
    constraint_mover = AddConstraintsToCurrentConformationMover()
    constraint_mover.apply(pose)
    fast_relax = FastRelax()
    fast_relax.set_scorefxn(scorefxn)
    fast_relax.max_iter(1000)
    fast_relax.ramp_down_constraints(True)
    fast_relax.apply(pose)
    pose.dump_pdb(f'{refine_pdb_file}')

# refine_conformations(os.path.join(dir,"pred_struct.pdb"), os.path.join(dir, "refined.pdb"))