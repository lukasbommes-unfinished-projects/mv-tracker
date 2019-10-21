import pickle

from video_cap import VideoCap




            cap = VideoCap()
            ret = cap.open(video_file)
            if not ret:
                raise RuntimeError("Could not open the video file")

            pbar = tqdm(total=num_frames)
            for frame_idx in range(0, num_frames):
                ret, frame, motion_vectors, frame_type, _ = cap.read()
                if not ret:
                    break

                pbar.update()

                data_item = {
                    "frame_idx": frame_idx,
                    "sequence": sequence,
                    "frame_type": frame_type,
                }

                data.append(data_item)

            cap.release()
            pbar.close()

        pickle.dump(data, open(os.path.join("preprocessed", codec, mode, "data.pkl"), 'wb'))
        pickle.dump(lengths[mode], open(os.path.join("preprocessed", codec, mode, "lengths.pkl"), 'wb'))
