import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import math
import numpy as np

from resnet_test1 import resnet56 as net

#device = torch.device("cuda:0")
model = net()#.to(device)

model_name = os.path.join("C://유민형//개인 연구//gpu_test//", "resnet56_test.pth")
checkpoint = torch.load(model_name)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(model.conv1.beta.min(), model.conv1.beta.max())
print("="*20)