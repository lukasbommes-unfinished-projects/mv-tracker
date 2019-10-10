class Config(object):
    SCALING_FACTOR = 1.0
    DETECTOR_PATH = "models/detector/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb"  # detector frozen inferenze graph (*.pb)
    DETECTOR_BOX_SIZE_THRES = None #(0.25*1920, 0.6*1080) # discard detection boxes larger than this threshold
    DETECTOR_INTERVAL = 20
    TRACKER_WEIGHTS_FILE = "models/tracker/09_10_2019.pth"
    TRACKER_IOU_THRES = 0.05
