import time
import torch
import numpy as np

from lib.models.pnet_dense import PropagationNetwork

m = PropagationNetwork()
#input = torch.normal(torch.zeros(3,1080,1920),torch.ones(3,1080,1920)) # 10 FPS
#input = torch.normal(torch.zeros(3,540,960),torch.ones(3,540,960))  # 37 FPS
#input = torch.normal(torch.zeros(3,270,480),torch.ones(3,270,480))  # 80 FPS

#input = torch.normal(torch.zeros(3,68,120),torch.ones(3,68,120))  # 115 FPS
#input = torch.normal(torch.zeros(3,34,60),torch.ones(3,34,60))  # 111 FPS
input = torch.normal(torch.zeros(3,17,30),torch.ones(3,17,30)) # 115 FPS

input = input.unsqueeze(0)
boxes_prev = torch.Tensor([[0,10,10,50,50],[0,15,15,55,55],[0,50,50,60,60]])

#boxes_prev = boxes_prev * 0.5

m = m.to("cuda:0")
input = input.to("cuda:0")
boxes_prev = boxes_prev.to("cuda:0")

dts = []
for i in range(100):
    if i > 0:
        t0 = time.process_time()
        m(input, None, boxes_prev)
        dts.append(time.process_time() - t0)

print(np.mean(dts))
print(1/np.mean(dts))


# RESULT:
# for larger scale the inference time depends on the size of the input
# for the small sized used in the dense network no change is observable

# DISCUSSION:
# Probably some setup code governs the inference time for the small input
