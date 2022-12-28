# show the Grad-CAM thermodynamic diagram

import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import normalize, resize, to_tensor, to_pil_image
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
from nets.yolo import YoloBody
import torch

device = "cuda" if torch.cuda.is_available() else 'cpu'
anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
num_classes = 1
net = YoloBody(anchors_mask, num_classes).to(device)
net.load_state_dict(torch.load("logs/baseline97.37/ep300-loss0.016-val_loss0.012.pth"))

hook_a = []
def _hook_a(module,inp,out):
    global hook_a
    hook_a.append(out)
    return None

hook_g = []
def _hook_g(module,inp,out):
    global hook_g
    hook_g.append(out[0])

submodule_dict = dict(net.named_modules())
target_layer_1 = submodule_dict['yolo_head1']
target_layer_2 = submodule_dict['yolo_head2']
target_layer_3 = submodule_dict['yolo_head3']
hook1 = []
hook2 = []

# 注册hook
hooka_1 = target_layer_1.register_forward_hook(_hook_a)
hook1.append(hooka_1)
hookg_1 = target_layer_1.register_backward_hook(_hook_g)
hook2.append(hookg_1)

hooka_2 = target_layer_2.register_forward_hook(_hook_a)
hook1.append(hooka_2)
hookg_2 = target_layer_2.register_backward_hook(_hook_g)
hook2.append(hookg_2)

hooka_3 = target_layer_3.register_forward_hook(_hook_a)
hook1.append(hooka_3)
hookg_3 = target_layer_3.register_backward_hook(_hook_g)
hook2.append(hookg_3)

# the path of the input image
img_path = 'VOCdevkit/VOC2007/JPEGImages/Dream_1353.jpg'
img = Image.open(img_path, mode='r').convert('RGB')
img_tensor = normalize(to_tensor(resize(img, (416, 416))),
                           [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).cuda()

scores = net(img_tensor.unsqueeze(0))

# the index of your classes in the vector
class_idx = 5

hook_a.reverse()
for i in range(len(scores)):
    loss = scores[i][:,5].sum()
    loss.backward(retain_graph=True)
    print(loss)
    hook1[i].remove()
    hook2[i].remove()

    # Grad-CAM
    # weights = hook_g[i].squeeze(0).mean(dim=(1,2))

    # Grad-CAM++
    grad_2 = hook_g[i].pow(2)
    grad_3 = grad_2 * hook_g[i]
    denom = 2 * grad_2 + (grad_3 * hook_a[i]).sum(dim=(2, 3), keepdim=True)
    nan_mask = grad_2 > 0
    grad_2[nan_mask].div_(denom[nan_mask])
    weights = grad_2.squeeze_(0).mul_(torch.relu(hook_g[i].squeeze(0))).sum(dim=(1, 2))

    cam = (weights.view(*weights.shape, 1, 1) * hook_a[i].squeeze(0)).sum(0)
    cam = F.relu(cam)
    cam.sub_(cam.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1))
    cam.div_(cam.flatten(start_dim=-2).max(-1).values.unsqueeze(-1).unsqueeze(-1))
    cam = cam.data.cpu().numpy()

    heatmap = to_pil_image(cam, mode='F')
    overlay = heatmap.resize(img.size, resample=Image.BICUBIC)
    cmap = cm.get_cmap('jet')
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
    alpha = .65
    result = (alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8)
    plt.imshow(result)
    plt.axis('off')
    # plt.show()
    plt.savefig('./CAM/standard_head'+str(i)+'_CAM'+".jpg",pad_inches=0.0)
