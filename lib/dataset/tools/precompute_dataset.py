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
    batch_size = 4
    codec = "mpeg4"
    modes = ["train", "val"]  # which datasets to generate
    input_folder = "data"  # where to look for the input dataset, relative to root dir
    output_folder = "data_precomputed" # where to save the precomputed samples, relative to root dir

    items = ["motion_vectors", "boxes_prev", "velocities", "num_boxes_mask",
        "det_boxes_prev"]

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
            batch_size=batch_size, codec=codec, visu=False, mode=mode)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
            shuffle=False, num_workers=0)

        print("Mode {} of {}".format(mode, modes))
        pbar = tqdm(total=len(dataloader))
        for step, (motion_vectors, boxes_prev, velocities, num_boxes_mask,
            det_boxes_prev) in enumerate(dataloader):

            data = {
                "motion_vectors": motion_vectors,
                "boxes_prev": boxes_prev,
                "velocities": velocities,
                "num_boxes_mask": num_boxes_mask,
                "det_boxes_prev": det_boxes_prev
            }

            # save data into output folder
            for item in items:
               output_file = os.path.join(output_folder, mode, item, "{:08d}.pkl".format(step))
               pickle.dump(data[item], open(output_file, "wb"))

            pbar.update()

        pbar.close()
