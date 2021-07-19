import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

checkpoint = torch.load(
    '/home/yuqingz/autonomous_driving/baseline/OpenPCDet/checkpoints/pointrcnn_7870.pth')

torch.min(checkpoint['model_state']['point_head.cls_layers.6.weight'])
torch.max(checkpoint['model_state']['point_head.cls_layers.6.weight'])

torch.min(checkpoint['model_state']['point_head.cls_layers.6.bias'])
torch.max(checkpoint['model_state']['point_head.cls_layers.6.bias'])

del checkpoint['model_state']['point_head.cls_layers.6.weight']
del checkpoint['model_state']['point_head.cls_layers.6.bias']

w = torch.empty(6, 256)
torch.nn.init.uniform_(w, a=-1.0, b=1.0)
b = torch.empty(6)
torch.nn.init.uniform_(b, a=-1.0, b=1.0)

checkpoint['model_state']['point_head.cls_layers.6.weight'] = w
checkpoint['model_state']['point_head.cls_layers.6.bias'] = b

torch.save(checkpoint, '/home/yuqingz/autonomous_driving/baseline/OpenPCDet/checkpoints/pointrcnn_7870_rmlast.pth')
