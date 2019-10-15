import motmetrics as mm
import numpy as np

# Create an accumulator that will be updated during each frame
acc = mm.MOTAccumulator(auto_id=True)

# Call update once for per frame. For now, assume distances between
# frame objects / hypotheses are given.
acc.update(
    ['a', 'b'],                 # Ground truth objects in this frame
    [1, 2, 3],                  # Detector hypotheses in this frame
    [
        [0.1, np.nan, 0.3],     # Distances from object 'a' to hypotheses 1, 2, 3
        [0.5,  0.2,   0.3]      # Distances from object 'b' to hypotheses 1, 2, 3
    ]
)
