# test the inference speed of the model in the GPU

import torch
from nets.ImprovedNeckYolo import YoloBody
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "logs/fuse/fuse_BN.pth"
anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
num_classes = 1
# net = YoloBody(anchors_mask, num_classes).to(device)
# net.load_state_dict(torch.load(model_path))
# net.eval()
net=torch.load(model_path, map_location="cuda").to(device)
net.eval()
print(net)

x = torch.randn(1, 3, 416, 416).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

t = []
for iteration in range(100):
    with torch.no_grad():
        starter.record()
        out = net(x)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
    t.append(curr_time)

print('timings mean:%s ms' % np.mean(t))
