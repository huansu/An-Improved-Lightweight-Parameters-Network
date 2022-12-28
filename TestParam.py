# measure the FLOPs and Parameters of the model

import torch
from nets.ImprovedNeckYolo import YoloBody
# from ptflops import get_model_complexity_info


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_classes = 1

    x = torch.randn((1, 3, 416, 416)).to(device)
    net = YoloBody(anchors_mask, num_classes).to(device)
    # #
    # flops, params = get_model_complexity_info(net, (3, 416, 416), as_strings=True, print_per_layer_stat=True)
    # print('flops: ', flops, 'params: ', params)
    from torchstat import stat

    stat(net, (3, 416, 416))
