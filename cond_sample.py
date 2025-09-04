import argparse
import torch
import math
import numpy as np
from tqdm import tqdm

from model import FoldingDiff
from util import wrap

DEFAULT_MU_REF = [
    -1.242405652999878, 1.1974568367004397, 0.8838186860084534,
    1.960501790046692, 2.029496431350708, 2.1438329219818115
]

def parse_argument(*args_list):
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default="")
    parser.add_argument('--timepoints', type=int, default=1000)
    parser.add_argument('--num-residues', type=int, default=76)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--mu-ref', type=float, nargs='+', default=DEFAULT_MU_REF)
    parser.add_argument('--ref-path', type=str, default="")
    parser.add_argument('--output', type=str, default='sample_trajectory_conditional.pt')
    parser.add_argument('--magicCOEF', type=float, default=0.1)
    return parser.parse_args(args_list) if args_list else parser.parse_args()

def main():
    args = parse_argument()
    T = args.timepoints
    mu_ref = torch.tensor(args.mu_ref, dtype=torch.float32)  # shape (6,)
    mu_ref = mu_ref.view(1, 1, 6).expand(1, args.num_residues, 6)  # shape (1, 76, 6)

    reference = torch.from_numpy(np.nan_to_num(np.load(args.ref_path)[0])).float()
    reference = wrap(reference - mu_ref)
    reference = reference.repeat(args.batch_size, 1, 1)  # (B, 76, 6)

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

    trajectory = []
    with torch.no_grad():
        x = wrap(torch.randn(args.batch_size, args.num_residues, 24))
        trajectory.append(x.unsqueeze(1))

        for t in tqdm(range(T, 0, -1), desc='sampling'):
            sigma_t = math.sqrt((1 - alpha_bar[t - 1]) / (1 - alpha_bar[t]) * beta[t])
            z = torch.randn_like(x) * sigma_t * args.magicCOEF if t > 1 else torch.zeros_like(x)

            t_tensor = torch.full((args.batch_size, 1), t, dtype=torch.long).cuda()
            out = model(x.cuda(), t_tensor, condition=reference.cuda()).cpu()

            x = 1 / math.sqrt(alpha[t]) * (x - beta[t] / math.sqrt(1 - alpha_bar[t]) * out) + z
            x = wrap(x)
            trajectory.append(x.unsqueeze(1))

    result = wrap(torch.cat(trajectory, dim=1))
    torch.save(result, args.output)

if __name__ == '__main__':
    main()
