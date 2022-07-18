import os

print('Select GPU number:')
gpu = int(input())
print('Selected GPU ' + str(gpu))

print('Enter batch size (default=20):')
b = str(input())
if len(b.strip()) == 0:
    b = 20
b = int(b)
print('batch_size ' + str(b))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % gpu

from time import time
import numpy as np

import torch

nn = torch.nn

layers = [
            nn.Conv2d(3,128,5), nn.ReLU(),
            nn.Conv2d(128,128,5), nn.ReLU(),
            nn.Conv2d(128,128,5), nn.ReLU(),
            nn.Conv2d(128,128,5), nn.ReLU(),
            nn.Conv2d(128,128,5), nn.ReLU(),
            nn.Conv2d(128,128,5), nn.ReLU(),
            nn.Conv2d(128,128,5), nn.ReLU(),
            nn.Conv2d(128,128,5), nn.ReLU(),
            nn.Conv2d(128,128,5), nn.ReLU(),
            nn.Conv2d(128,128,5), nn.ReLU(),
            nn.Conv2d(128,128,5), nn.ReLU(),
            nn.Conv2d(128,128,5), nn.ReLU(),
            nn.Conv2d(128,128,5), nn.ReLU(),
            nn.Conv2d(128,128,5), nn.ReLU(),
            nn.Conv2d(128,128,5), nn.ReLU(),
            nn.Conv2d(128,128,5), nn.ReLU(),
            nn.Conv2d(128,128,5), nn.ReLU(),
            nn.Conv2d(128,128,5), nn.ReLU(),
            nn.Conv2d(128,128,5), nn.ReLU(),
            nn.Conv2d(128,128,5), nn.ReLU(),
            nn.Conv2d(128,128,5), nn.ReLU(),
            nn.Conv2d(128,128,5), nn.ReLU(),
            nn.Conv2d(128,128,5), nn.ReLU(),
        ]
layers = [layer.cuda() for layer in layers]

def forward(x):
    net = layers[0](x)
    num = np.random.randint(0, len(layers)-1)
    for i in range(num):
        net = layers[i+1](net)
    return net

while True:
    x = torch.rand(b, 3, 224, 224)
    x = x.cuda()
    x = torch.tensor(x, requires_grad=True)
    res = torch.mean(forward(x))
    res.backward()
    #x.grad.cpu().numpy()