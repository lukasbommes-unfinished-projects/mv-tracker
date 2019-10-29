# mpeg4 upsampled, training data, only static cameras (Note: Theese metrics still include the num_boxes_mask bug)
class StatsMpeg4UpsampledStatic():
    velocities = {
        "mean": [0.0035801701807530333, 0.0002152514312298825, 0.00021574009192311123, 8.994028635911342e-05],
        "std": [0.09732869575370293, 0.012557833992140246, 0.01932378335492126, 0.004216800692202473]
    }
    motion_vectors = {
        "mean": [0.0, 0.014817102005829075, 0.0705440781107341], # [0.0, mvs_mean_y, mvs_mean_x]
        "std": [1.0, 0.060623864350822454,  0.4022695243698158]  # [1.0, mvs_std_y, mvs_std_x]
    }

# mpeg4 upsampled, training data, full dataset
class StatsMpeg4UpsampledFull():
    velocities = {
        "mean": [-0.018599934971495233, 0.0025663658130240146, 0.002417661236454852, 0.002733427047478466],
        "std": [0.21269956851634267, 0.036000079453724264, 0.044600952658130015, 0.014932423372624959]
    }
    motion_vectors = {
        "mean": [0.0, 0.139698425141241, -0.16339078281237884],
        "std": [1.0, 0.9256943895550922,  2.821564673026061]
    }

######################### NEW DATASET (excludes keyframes) #########################

# mpeg4 upsampled, no keyframes, training data, only static cameras
# codec = mpeg4, mvs_mode = upsampled, static_only = True, exclude_keyframes = True, scales = [1.0]
class StatsMpeg4UpsampledStaticSinglescale():
    velocities = {
        "mean": [0.003505179250665827, 0.00018602565541264086, 0.0002404537348835983, 8.181186571028626e-05],
        "std": [0.09751977443928779, 0.012586229216352347, 0.0193288116599051, 0.0042137634660418125]
    }
    motion_vectors = {
        "mean": [0.0, 0.016144431789511565, 0.07686348226290671],
        "std": [1.0, 0.06311146108698115, 0.41932218398916854]
    }


# mpeg4 upsampled, no keyframes, training data, full dataset
# codec = mpeg4, mvs_mode = upsampled, static_only = False, exclude_keyframes = True, scales = [1.0]
class StatsMpeg4UpsampledFullSinglescale():
    velocities = {
        "mean": [-0.018749966234393636, 0.002650610891189643, 0.0024707428297439603, 0.002743245574765845],
        "std": [0.213048754134384, 0.036284140941397, 0.04492437750683111, 0.014939794555400754]
    }
    motion_vectors = {
        "mean": [0.0, 0.152116229012106, -0.17859823792827198],
        "std": [1.0, 0.965357249505038, 2.9459254858101116]
    }


# mpeg4 dense, no keyframes, training data, only static cameras
# codec = mpeg4, mvs_mode = dense, static_only = True, exclude_keyframes = True, scales = [1.0, 0.75, 0.5]
class StatsMpeg4DenseStaticMultiscale():
    velocities = {
        "mean": [0.003505179233179393, 0.00018602562223680547, 0.00024045374204044416, 8.181187497684377e-05],
        "std": [0.09751977367068247, 0.012586229218958307, 0.019328811763204536, 0.00421376344570489]
    }
    motion_vectors = {
        "mean": [0.0, 0.013189169794109501, 0.05825615665845193],
        "std": [1.0, 0.06031575367251622, 0.374547766431143]
    }

# mpeg4 dense, no keyframes, training data, only static cameras
# codec = mpeg4, mvs_mode = dense, static_only = True, exclude_keyframes = True, scales = [1.0]
class StatsMpeg4DenseStaticSinglescale():
    velocities = {
        "mean": [0.003505179250665827, 0.00018602565541264086, 0.0002404537348835983, 8.181186571028626e-05],
        "std": [0.09751977443928779, 0.012586229216352347, 0.0193288116599051, 0.0042137634660418125]
    }
    motion_vectors = {
        "mean": [0.0, 0.016107642943830696, 0.07541612903921065],
        "std": [1.0, 0.06298750897933865, 0.42173798690940506]
    }
