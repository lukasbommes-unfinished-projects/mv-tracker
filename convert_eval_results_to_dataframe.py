import os
import pickle

import pandas as pd


eval_dir = "eval_output/custom"  # the directory containing evaluation results
outfile = "parsed_eval_results.csv" # where to write the parsed dataframe

if __name__ == "__main__":

    # get all paths that have files in their leaves
    path_info = [(path, files) for path, dirs, files in list(os.walk(eval_dir)) if len(files) > 0]

    # remove all the paths which already contain a mot_metrics.log file
    path_info = [(path, files) for path, files in path_info if 'mot_metrics.log' in files]

    # create empty data frame
    df_variables = [
        "scale",
        "codec",
        "tracker_type",
        "mvs_mode",
        "weights_file",
        "vector_type",
        "iou_thres",
        "conf_thres",
        "state_thres",
        "det_interval"
    ]
    df_mot_metrics = [
        "IDF1",
        "IDP",
        "IDR",
        "Rcll",
        "Prcn",
        "GT",
        "MT",
        "PT",
        "ML",
        "FP",
        "FN",
        "IDs",
        "FM",
        "MOTA",
        "MOTP",
        "predict_fps_mean",
        "predict_fps_std",
        "total_fps_mean",
        "total_fps_std",
        "inference_fps_mean",
        "inference_fps_std"
    ]

    data = []

    for eval_path, files in path_info:

        eval_path_items = str.split(eval_path, "/")

        # strip off the root directory
        eval_path_items = eval_path_items[len(str.split(eval_dir, "/")):]

        # parse settings from file name
        scale = eval_path_items[0]
        codec = eval_path_items[1]
        tracker_type = eval_path_items[2]

        if tracker_type == "baseline":
            mvs_mode = None
            weights_file = None
            if codec == "mpeg4":
                vector_type = None
                iou_thres = eval_path_items[3]
                conf_thres = eval_path_items[4]
                state_thres = eval_path_items[5]
                det_interval = eval_path_items[6]
            elif codec == "h264":
                vector_type = eval_path_items[3]
                iou_thres = eval_path_items[4]
                conf_thres = eval_path_items[5]
                state_thres = eval_path_items[6]
                det_interval = eval_path_items[7]

        elif tracker_type == "deep":
            mvs_mode = eval_path_items[3]
            weights_file = eval_path_items[4]
            if codec == "mpeg4":
                vector_type = None
                iou_thres = eval_path_items[5]
                conf_thres = eval_path_items[6]
                state_thres = eval_path_items[7]
                det_interval = eval_path_items[8]
            elif codec == "h264":
                vector_type = eval_path_items[5]
                iou_thres = eval_path_items[6]
                conf_thres = eval_path_items[7]
                state_thres = eval_path_items[8]
                det_interval = eval_path_items[9]

        # post process the parsed settings
        scale = float(scale[6:])
        iou_thres = float(iou_thres[10:])
        conf_thres = float(conf_thres[11:])
        state_thres = state_thres[12:]
        det_interval = int(det_interval[13:])

        # load mot metrics file
        mot_metrics_file = os.path.join(eval_path, "mot_metrics.pkl")
        mot_metrics = pickle.load(open(mot_metrics_file, "rb"))

        # load FPS from file
        fps_csv_file = os.path.join(eval_path, "time_perf.log")
        fps_csv_data = pd.read_csv(fps_csv_file, delimiter=",")
        predict_fps_mean = fps_csv_data.iloc[-2, 0]
        predict_fps_std = fps_csv_data.iloc[-2, 1]
        total_fps_mean = fps_csv_data.iloc[-2, 4]
        total_fps_std = fps_csv_data.iloc[-2, 5]
        inference_fps_mean = fps_csv_data.iloc[-2, 6]
        inference_fps_std = fps_csv_data.iloc[-2, 7]

        # pack everything into a single row
        row = [scale, codec, tracker_type, mvs_mode, weights_file,
               vector_type, iou_thres, conf_thres, state_thres, det_interval]
        row.extend(list(mot_metrics.loc["OVERALL", :]))
        row.extend([predict_fps_mean, predict_fps_std, total_fps_mean,
            total_fps_std, inference_fps_mean, inference_fps_std])

        data.append(row)

    df = pd.DataFrame(data, columns=df_variables+df_mot_metrics)

    df.to_csv(outfile)
    print("Wrote output file to {}".format(outfile))
