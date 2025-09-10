from biotite.structure.io.pdb import PDBFile
import biotite.structure as struc
import numpy as np
import warnings; warnings.filterwarnings('ignore')
import torch
import os
import sys

def get_angle_from_pdb(pdb):

    pdb = PDBFile.read(pdb)

    if pdb.get_model_count() > 1:
        raise ValueError("Multiple models are not supported")

    structure = pdb.get_structure(model=1)

    # dihedrals
    phi, psi, omega = struc.dihedral_backbone(structure)
    # print("phi shape=",phi.shape)
    # print("psi shape=",psi.shape)
    # print("omega shape=",omega.shape)
    # angles
    backbone = structure[struc.filter_backbone(structure)]
    n = len(backbone)
    # print("---------------------------------")
    # print("backbone length=", n)

    #print(backbone)

    triplet_indices = np.array([
        np.arange(n - 2),
        np.arange(1, n - 1),
        np.arange(2, n)
    ]).T

    theta1 = struc.index_angle(backbone, triplet_indices[range(0, n - 2, 3)])
    theta2 = struc.index_angle(backbone, triplet_indices[range(1, n - 2, 3)])
    theta3 = struc.index_angle(backbone, triplet_indices[range(2, n - 2, 3)])
    print("theta1 type", type(theta1))
    npy = np.array([
        phi,
        psi,
        omega,
        theta1,
        np.hstack([theta2, np.nan]),  # theta2 is not defined for the last residue
        np.hstack([theta3, np.nan]),  # theta3 is not defined for the last residue
    ]).T


    return npy

def get_delta_angle_from_pdb(pdb,reference):
    reference_angle=np.load(reference)
    pdb = PDBFile.read(pdb)

    if pdb.get_model_count() > 1:
        raise ValueError("Multiple models are not supported")

    structure = pdb.get_structure(model=1)

    # dihedrals
    phi, psi, omega = struc.dihedral_backbone(structure)

    # angles
    backbone = structure[struc.filter_backbone(structure)]
    n = len(backbone)

    triplet_indices = np.array([
        np.arange(n - 2),
        np.arange(1, n - 1),
        np.arange(2, n)
    ]).T

    theta1 = struc.index_angle(backbone, triplet_indices[range(0, n - 2, 3)])
    theta2 = struc.index_angle(backbone, triplet_indices[range(1, n - 2, 3)])
    theta3 = struc.index_angle(backbone, triplet_indices[range(2, n - 2, 3)])

    npy = np.array([
        phi,
        psi,
        omega,
        theta1,
        np.hstack([theta2, np.nan]),  # theta2 is not defined for the last residue
        np.hstack([theta3, np.nan]),  # theta3 is not defined for the last residue
    ]).T

    delta_npy=npy-reference_angle
    return delta_npy

def place_fourth_atom(a, b, c, bond_angle, torsion, bond_length):
    # Place atom D with respect to atom C at origin.
    d = np.array(
        [
            bond_length * np.cos(np.pi - bond_angle),
            bond_length * np.cos(torsion) * np.sin(bond_angle),
            bond_length * np.sin(torsion) * np.sin(bond_angle),
        ]
    ).T

    # Transform atom D to the correct frame.
    bc = c - b
    bc /= np.linalg.norm(bc)  # Unit vector from B to C.

    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n)  # Normal vector of the plane defined by a, b, c.

    M = np.array([bc, np.cross(n, bc), n]).T
    return M @ d + c



def angles2coord(angles, n_ca=1.46, ca_c=1.54, c_n=1.33):
    """Given L x 6 angle matrix,
    reconstruct the Cartesian coordinates of atoms.
    Returns L x 3 coordinate matrix.

    Implements NeRF (Natural Extension Reference Frame) algorithm.
    """
    if isinstance(angles, torch.Tensor):
        phi, psi, omega, theta1, theta2, theta3 = angles.T.numpy()
    else:
        phi, psi, omega, theta1, theta2, theta3 = angles.T


    torsions = np.stack([psi[:-1], omega[:-1], phi[1:]], axis=-1).flatten()
    bond_angles = np.stack([theta2[:-1], theta3[:-1], theta1[1:]], axis=-1).flatten()

    #
    # Place the first three atoms.
    #
    # The first atom (N) is placed at origin.
    a = np.zeros(3)
    # The second atom (Ca) is placed on the x-axis.
    b = np.array([1, 0, 0]) * n_ca
    # The third atom (C) is placed on the xy-plane with bond angle theta1[0]
    c = np.array([np.cos(np.pi - theta1[0]), np.sin(np.pi - theta1[0]), 0]) * ca_c + b

    # Iteratively place the fourth atom based on the last three atoms.

    coords = [a, b, c]
    # cycle through [n, ca, c, n, ca, c, ...]
    for i, bond_length in enumerate([c_n, n_ca, ca_c] * (len(angles) - 1)):
        torsion, bond_angle = torsions[i], bond_angles[i]
        d = place_fourth_atom(a, b, c, bond_angle, torsion, bond_length)
        coords.append(d)

        a, b, c = b, c, d

    return np.array(coords)



def coordinate_to_pdb(coordinate,index,dirpath):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    external_folder_path = os.path.join(current_directory, dirpath)
    print(external_folder_path)

    num_residues = len(coordinate) // 3
    structure = struc.AtomArray(len(coordinate))
    structure.coord = coordinate
    structure.atom_name = ["N", "CA", "C"] * (num_residues)
    structure.res_name = ['MET', 'MET', 'MET', 'GLN', 'GLN', 'GLN', 'ILE', 'ILE', 'ILE', 'PHE', 'PHE', 'PHE', 'VAL', 'VAL', 'VAL', 'LYS', 'LYS', 'LYS', 'THR', 'THR', 'THR', 'LEU', 'LEU', 'LEU', 'THR', 'THR', 'THR', 'GLY', 'GLY', 'GLY', 'LYS', 'LYS', 'LYS', 'THR', 'THR', 'THR', 'ILE', 'ILE', 'ILE', 'THR', 'THR', 'THR', 'LEU', 'LEU', 'LEU', 'GLU', 'GLU', 'GLU', 'VAL', 'VAL', 'VAL',
                        'GLU', 'GLU', 'GLU', 'PRO', 'PRO', 'PRO', 'SER', 'SER', 'SER', 'ASP', 'ASP', 'ASP', 'THR', 'THR', 'THR', 'ILE', 'ILE', 'ILE', 'GLU', 'GLU', 'GLU', 'ASN', 'ASN', 'ASN', 'VAL', 'VAL', 'VAL', 'LYS', 'LYS', 'LYS', 'ALA', 'ALA', 'ALA', 'LYS', 'LYS', 'LYS', 'ILE', 'ILE', 'ILE', 'GLN', 'GLN', 'GLN', 'ASP', 'ASP', 'ASP', 'LYS', 'LYS', 'LYS', 'GLU', 'GLU', 'GLU', 
                        'GLY', 'GLY', 'GLY', 'ILE', 'ILE', 'ILE', 'PRO', 'PRO', 'PRO', 'PRO', 'PRO', 'PRO', 'ASP', 'ASP', 'ASP', 'GLN', 'GLN', 'GLN', 'GLN', 'GLN', 'GLN', 'ARG', 'ARG', 'ARG', 'LEU', 'LEU', 'LEU', 'ILE', 'ILE', 'ILE', 'PHE', 'PHE', 'PHE', 'ALA', 'ALA', 'ALA', 'GLY', 'GLY', 'GLY', 'LYS', 'LYS', 'LYS', 'GLN', 'GLN', 'GLN', 'LEU', 'LEU', 'LEU', 'GLU', 'GLU', 'GLU',
                            'ASP', 'ASP', 'ASP', 'GLY', 'GLY', 'GLY', 'ARG', 'ARG', 'ARG', 'THR', 'THR', 'THR', 'LEU', 'LEU', 'LEU', 'SER', 'SER', 'SER', 'ASP', 'ASP', 'ASP', 'TYR', 'TYR', 'TYR', 'ASN', 'ASN', 'ASN', 'ILE', 'ILE', 'ILE', 'GLN', 'GLN', 'GLN', 'LYS', 'LYS', 'LYS', 'GLU', 'GLU', 'GLU', 'SER', 'SER', 'SER', 'THR', 'THR', 'THR', 'LEU', 'LEU', 'LEU', 'HIS', 'HIS', 'HIS', 
                            'LEU', 'LEU', 'LEU', 'VAL', 'VAL', 'VAL', 'LEU', 'LEU', 'LEU', 'ARG', 'ARG', 'ARG', 'LEU', 'LEU', 'LEU', 'ARG', 'ARG', 'ARG', 'GLY', 'GLY', 'GLY', 'GLY', 'GLY', 'GLY']

    structure.res_id = np.repeat(range(1, num_residues + 1), 3)
    pdb = PDBFile()
    pdb.set_structure(structure)
    pdb.write(external_folder_path+f"/rosetta_input_{index}.pdb")
    # pdb.write(external_folder_path+f"/{index}.pdb")
#

# #data=torch.load('denoise_delta_599_T_0.01.pt')


# # # build the final conformations
# batchsize=100
# processed_data = np.zeros((batchsize, 147* 3, 3))
# # 对每个 batch 和每个序列进行处理
# for i in range(batchsize):
#     processed_data[i]=angles2coord(data[i,1000])

# i=0
# for i in range(batchsize):
#     coordinate_to_pdb(processed_data[i],i,dirpath='denoise_process_10_delta_angle_last_frame_0.01')


# #build the sample process
# one_conformation=data[67]   # which conformation's sample process is building
# frames=1001
# processed_data = np.zeros((frames, 147* 3, 3))
# for i in range(frames):
#     processed_data[i]=angles2coord(one_conformation[i])

# i=0
# for i in range(frames):
#     coordinate_to_pdb(processed_data[i],i,dirpath='denoise_process_10_delta_angle_process_0.01')