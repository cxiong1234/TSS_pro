import argparse
import torch
import math
import numpy as np
from tqdm import tqdm
import os
from model import FoldingDiff
from util import wrap
from get_angle_from_pdb_func import angles2coord, coordinate_to_pdb, get_angle_from_pdb
from refine_pyrosetta import refine_conformations
import MDAnalysis as mda
from MDAnalysis.coordinates.PDB import PDBWriter

# ubiquitin mu 5 snapshots
DEFAULT_MU_REF = [
    -1.242405652999878, 1.1974568367004397, 0.8838186860084534,
    1.960501790046692, 2.029496431350708, 2.1438329219818115
]

def parse_argument(*args_list):
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default="")
    parser.add_argument('--timepoints', type=int, default=1000)
    parser.add_argument('--num-residues', type=int, default=76)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--mu-ref', type=float, nargs='+', default=DEFAULT_MU_REF)
    parser.add_argument('--ref-path', type=str, default="")
    #parser.add_argument('--output', type=str, default='sample_trajectory_conditional.pt')
    parser.add_argument('--magicCOEF', type=float, default=0.1)
    parser.add_argument('--max_sampling_cycle', type=int, default=10000)
    parser.add_argument('--continueCycle', type=int, default=None)
    return parser.parse_args(args_list) if args_list else parser.parse_args()

def main():

    ### def function for single sampling 
    def sampling_code(reference_abs_torch, cycle_idx): 

        
        reference_input = wrap(reference_abs_torch - mu_ref)  ## (1, 537, 6)
        reference_input = reference_input.repeat(args.batch_size, 1, 1)  # (B, 537, 6)

        trajectory = []
        with torch.no_grad():
            x = wrap(torch.randn(args.batch_size, args.num_residues, 24))
            trajectory.append(x.unsqueeze(1))

            for t in tqdm(range(T, 0, -1), desc='sampling'):
                sigma_t = math.sqrt((1 - alpha_bar[t - 1]) / (1 - alpha_bar[t]) * beta[t])
                z = torch.randn_like(x) * sigma_t * args.magicCOEF if t > 1 else torch.zeros_like(x)

                # t_tensor = torch.tensor([t]).long().expand(args.batch_size).cuda()
                t_tensor = torch.full((args.batch_size, 1), t, dtype=torch.long).cuda()
                out = model(x.cuda(), t_tensor, condition=reference_input.cuda()).cpu()   ## (B, N, 54)

                x = 1 / math.sqrt(alpha[t]) * (x - beta[t] / math.sqrt(1 - alpha_bar[t]) * out) + z
                x = wrap(x)
                trajectory.append(x.unsqueeze(1))

        result = wrap(torch.cat(trajectory, dim=1))
        last_frame_output = reference_abs_torch + result[0, -1, :, 18:24]
        torch.save(result, os.path.join(save_dir, f"{cycle_idx}_sample_trajectory_conditional.pt"))
        # torch.save(last_frame_output, f"{cycle_idx}_last_frame_output.pt")
        return result, last_frame_output
    ## after function done

    torch.cuda.set_device(0)  # or the proper device if more than one

    ## create a new folder to save the .pt sampled data
    # Create the folder if it doesn't exist
    save_dir = os.path.join(os.getcwd(), "100ns_all_gen_traj_data")
    os.makedirs(save_dir, exist_ok=True)


    args = parse_argument()
    T = args.timepoints
    
    model = FoldingDiff()
    state_dict = torch.load(args.ckpt, map_location=torch.device('cpu'))['state_dict']
    model.load_state_dict(state_dict)
    model.cuda().eval()

    s = 8e-3
    t = torch.arange(T + 1)
    f_t = torch.cos((t / T + s) / (1 + s) * math.pi / 2.0).square()
    alpha_bar = f_t / f_t[0]
    beta = torch.cat([torch.tensor([1e-20]), torch.clip(1 - alpha_bar[1:] / alpha_bar[:-1], min=1e-5, max=1 - 1e-5)])
    alpha = 1 - beta

    mu_ref = torch.tensor(args.mu_ref, dtype=torch.float32)  # shape (6,)
    mu_ref = mu_ref.view(1, 1, 6).expand(1, args.num_residues, 6)  # shape (1, 76, 6)
    
    reference = torch.from_numpy(np.nan_to_num(np.load(args.ref_path)[30003])).float()  if args.continueCycle == None else torch.load(os.path.join(save_dir, f"condition_{args.continueCycle}.pt")) ## (537, 6)
    prev_last_frame = None
    
    if not args.continueCycle:
        torch.save(reference, os.path.join(save_dir, "condition_0.pt"))

    start_idx = args.continueCycle if  args.continueCycle else 0
    # sampling number of cycles
    for idx in range(start_idx, args.max_sampling_cycle):
        _, prev_last_frame = sampling_code(reference if prev_last_frame is None else prev_last_frame, idx)
        

        ## prev_last_frame -> rosetta_input_<idx>.pdb
        coordinate_to_pdb(angles2coord(prev_last_frame), idx, save_dir)
        refine_conformations(os.path.join(save_dir, f"rosetta_input_{idx}.pdb"), os.path.join(save_dir, f"rosetta_output_{idx}.pdb"))
        u = mda.Universe(os.path.join(save_dir, f"rosetta_output_{idx}.pdb"))
            # Select atoms: name CA or C or N
        selection = u.select_atoms("name CA or name C or name N")

        # Change residue names to GLY
        for res in selection.residues:
            res.resname = "GLY"

        tmp_pdb_path = os.path.join(save_dir, "tmp_Modified_out.pdb")
        # Save to output PDB
        with PDBWriter(tmp_pdb_path) as writer:
            writer.write(selection)

        # Get angles from the PDB file
        angles = get_angle_from_pdb(os.path.join(save_dir, "tmp_Modified_out.pdb"))
        os.remove(tmp_pdb_path)        
        prev_last_frame = torch.from_numpy(np.nan_to_num(angles)).float()
        torch.save(prev_last_frame, os.path.join(save_dir, f"condition_{idx+1}.pt"))
    

if __name__ == '__main__':
    main()
