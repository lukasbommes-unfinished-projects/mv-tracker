# mpeg4 upsampled, training data, only static cameras (Note: Theese metrics still include the num_boxes_mask bug)
class StatsMpeg4UpsampledStatic():
    velocities = {
        "mean": [0.0035801701807530333, 0.0002152514312298825, 0.00021574009192311123, 8.994028635911342e-05],
        "std": [0.09732869575370293, 0.012557833992140246, 0.01932378335492126, 0.004216800692202473]
    }
    motion_vectors = {
        "mean": [0.0, 0.014817102005829075, 0.0705440781107341],
        "std": [1.0, 0.060623864350822454,  0.4022695243698158]
    }

# mpeg4 upsampled, training data, full dataset, batch size
class StatsMpeg4UpsampledFull():
    velocities = {
        "mean": [-0.018599934971495233, 0.0025663658130240146, 0.002417661236454852, 0.002733427047478466],
        "std": [0.21269956851634267, 0.036000079453724264, 0.044600952658130015, 0.014932423372624959]
    }
    motion_vectors = {
        "mean": [0.0, 0.139698425141241, -0.16339078281237884],
        "std": [1.0, 0.9256943895550922,  2.821564673026061]
    }

# mpeg4 dense, no keyframes, training data, only static cameras
# codec = mpeg4, mvs_mode = dense, static_only = True, exclude_keyframes = True, scales = [1.0, 0.75, 0.5]
class StatsMpeg4DenseStatic():
    velocities = {
        "mean": [0.003505179233179393, 0.00018602562223680547, 0.00024045374204044416, 8.181187497684377e-05],
        "std": [0.09751977367068247, 0.012586229218958307, 0.019328811763204536, 0.00421376344570489]
    }
    motion_vectors = {
        "mean": [0.0, 0.013189169794109501, 0.05825615665845193],
        "std": [1.0, 0.06031575367251622, 0.374547766431143]
    }


# # h264, training data
# class Stats():
#     velocities = {
#         "mean": [],
#         "std": []
#     }
#     motion_vectors = {
#         "mean": [0.0, 0.3219420202390504, -0.3864056486553166],
#         "std": [1.0, 1.277147814669969, 4.76270068707976]
#     }
#
# # mpeg4, training data
# class Stats():
#     velocities = {
#         "mean": [],
#         "std": []
#     }
#     motion_vectors = {
#         "mean": [0.0, 0.1770176594258104, -0.12560456383521534],
#         "std": [1.0, 0.7420489598781672, 1.8279847980299613]
#     }
