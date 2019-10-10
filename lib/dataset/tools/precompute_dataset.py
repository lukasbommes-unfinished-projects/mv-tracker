# This script precomputes an offline dataset for faster training
import os
import errno
import pickle
from tqdm import tqdm
import torch

from lib.dataset.dataset import MotionVectorDataset
from lib.dataset.stats import Stats
from lib.transforms.transforms import standardize_motion_vectors, \
    standardize_velocities, scale_image


# run as python -m lib.dataset.tools.precompute_dataset from root dir
if __name__ == "__main__":

    # configure desired dataset settings here
    batch_size = 8
    codec = "mpeg4"
    modes = ["train", "val"]  # which datasets to generate
    input_folder = "data"  # where to look for the input dataset, relative to root dir
    output_folder = "data_precomputed" # where to save the precomputed samples, relative to root dir
    stats = Stats()

    items = ["motion_vectors", "boxes_prev", "velocities", "num_boxes_mask",
        "motion_vector_scale", "det_boxes_prev"]

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

            # standardize velocities
            velocities = standardize_velocities(velocities,
                mean=stats.velocities["mean"],
                std=stats.velocities["std"])

            # standardize motion vectors
            motion_vectors = standardize_motion_vectors(motion_vectors,
                mean=stats.motion_vectors["mean"],
                std=stats.motion_vectors["std"])

            # resize spatial dimensions of motion vectors
            motion_vectors, motion_vector_scale = scale_image(motion_vectors,
                short_side_min_len=600, long_side_max_len=1000)

            # swap channel order of motion vectors from BGR to RGB
            motion_vectors = motion_vectors[..., [2, 1, 0]]

            # swap motion vector axes so that shape is (B, C, H, W) instead of (B, H, W, C)
            motion_vectors = motion_vectors.permute(0, 3, 1, 2)

            motion_vector_scale = torch.tensor(motion_vector_scale)
            motion_vector_scale = motion_vector_scale.repeat(batch_size).view(-1, 1)

            data = {
                "motion_vectors": motion_vectors,
                "boxes_prev": boxes_prev,
                "velocities": velocities,
                "num_boxes_mask": num_boxes_mask,
                "motion_vector_scale": motion_vector_scale,
                "det_boxes_prev": det_boxes_prev
            }

            # save data into output folder
            for item in items:
               output_file = os.path.join(output_folder, mode, item, "{:08d}.pkl".format(step))
               pickle.dump(data[item], open(output_file, "wb"))

            pbar.update()

        pbar.close()