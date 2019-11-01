# This script precomputes an offline dataset for faster training
import os
import errno
import pickle
from tqdm import tqdm
import torch

from lib.dataset.dataset import MotionVectorDataset


# run as python -m lib.dataset.tools.precompute_dataset from root dir
if __name__ == "__main__":

    # configure desired dataset settings here
    batch_size = 8
    codec = "mpeg4"
    static_only = True

    modes = ["train", "val"]  # which datasets to generate
    input_folder = "data"  # where to look for the input dataset, relative to root dir
    output_folder = "data_precomputed" # where to save the precomputed samples, relative to root dir

    items = ["motion_vectors", "boxes_prev", "boxes", "velocities",
        "num_boxes_mask"]

    for mode in modes:

        for item in items:
            try:
                os.makedirs(os.path.join(output_folder, mode, item))
            except FileExistsError:
                msg = ("Looks like the output directory "
                f"'{output_folder}' already contains data. Manually move or "
                "delete this directory before proceeding.")
                raise FileExistsError(msg)

        dataset = MotionVectorDataset(root_dir=input_folder,
            batch_size=batch_size, codec=codec, static_only=static_only,
            visu=False, mode=mode)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
            shuffle=False, num_workers=0)

        print("Mode {} of {}".format(mode, modes))
        pbar = tqdm(total=len(dataloader))
        for step, sample in enumerate(dataloader):
            # save data into output folder
            for sample_name, data in sample.items():
               output_file = os.path.join(output_folder, mode, sample_name, "{:08d}.pkl".format(step))
               pickle.dump(data, open(output_file, "wb"))

            pbar.update()

        pbar.close()
