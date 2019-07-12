import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

import util


def main():
    """
    Load trained generator model and save generated sampels.
    <n_row> x <n_row> grid image is saved into <output_dir>
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("gen_model_file", type=Path)
    parser.add_argument("gen_params_file", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--n_row", "-n", type=int, default=10)
    parser.add_argument("--batchsize", "-b", type=int, default=10)
    args = parser.parse_args()

    # restore model snapshot
    gen_model = torch.load(args.gen_model_file)
    gen_model.load_state_dict(torch.load(args.gen_params_file))
    gen_model = gen_model.to(util.current_device())

    # generate and save samples
    args.output_dir.mkdir(exist_ok=True, parents=True)
    latent_vars = gen_model.latent_vars
    for var_name, var in tqdm(latent_vars.items()):
        img: np.array
        if var.kind == "z":
            img = util.gen_random_images(gen_model, args.n_row)
        else:
            if var.prob_name == "categorical":
                img = util.gen_images_discrete(gen_model, var_name)
            else:
                img = util.gen_images_continuous(gen_model, var_name, args.n_row)

        save_image(img, args.output_dir / f"{var_name}.jpg")


if __name__ == "__main__":
    main()
