# After the training, the model will be fused

import torch
import torch.nn as nn

from nets.ImprovedNeckYolo import YoloBody


class DummyModule(nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()

    def forward(self, x):
        return x


def fuse(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    beta = bn.weight
    gamma = bn.bias

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean)/var_sqrt * beta + gamma

    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        bias=True,
        padding_mode=conv.padding_mode
    )
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv


def fuse_module(m):
    children = list(m.named_children())
    conv = None
    conv_name = None

    for name, child in children:
        if isinstance(child, nn.BatchNorm2d) and conv:
            bc = fuse(conv, child)
            m._modules[conv_name] = bc
            # m._modules[name] = DummyModule()
            del m._modules[name]
            conv = None
        elif isinstance(child, nn.Conv2d):
            conv = child
            conv_name = name
        else:
            fuse_module(child)


def validate(net, input_, cuda=True):
    net.eval()
    if cuda:
        net.cuda()
    if cuda:
        torch.cuda.synchronize()
    fuse_module(net)
    # print(net)
    # save the net and weight
    torch.save(net, "logs/fuse/fuse_BN.pth")
    return None


if __name__ == '__main__':
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_classes = 1
    net = YoloBody(anchors_mask, num_classes)
    net.load_state_dict(torch.load("model_data/ChangedNeck.pth"))

    net.eval()
    validate(net, torch.randn(1, 3, 416, 416), True)

