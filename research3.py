import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import math
import numpy as np

from preact_resnet import resnet56 as net

device = torch.device("cuda:0")
model = net()#.to(device)

model_name = os.path.join("C://유민형//개인 연구//gpu_test//", "resnet56_preact.pth")
checkpoint = torch.load(model_name)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
'''
a = torch.randn((1,3,32,32)).to(device)
b = model(a)

'''
#print(model.bn1.running_mean.min(), model.bn1.running_mean.max())
#print("="*20)
for i in range(9):
    print("-"*10)
    print(model.layer1[i].bn1.running_mean.min(), model.layer1[i].bn1.running_mean.max())
    print(model.layer1[i].bn2.running_mean.min(), model.layer1[i].bn2.running_mean.max())

print("="*20)
for i in range(9):
    print("-"*10)
    print(model.layer2[i].bn1.running_mean.min(), model.layer2[i].bn1.running_mean.max())
    print(model.layer2[i].bn2.running_mean.min(), model.layer2[i].bn2.running_mean.max())

print("="*20)
for i in range(9):
    print("-"*10)
    print(model.layer3[i].bn1.running_mean.min(), model.layer3[i].bn1.running_mean.max())
    print(model.layer3[i].bn2.running_mean.min(), model.layer3[i].bn2.running_mean.max())



print("#"*40)
#print(model.bn1.bias.min(), model.bn1.bias.max())
#print("="*20)
for i in range(9):
    print("-"*10)
    print(model.layer1[i].bn1.bias.min(), model.layer1[i].bn1.bias.max())
    print(model.layer1[i].bn2.bias.min(), model.layer1[i].bn2.bias.max())

print("="*20)
for i in range(9):
    print("-"*10)
    print(model.layer2[i].bn1.bias.min(), model.layer2[i].bn1.bias.max())
    print(model.layer2[i].bn2.bias.min(), model.layer2[i].bn2.bias.max())

print("="*20)
for i in range(9):
    print("-"*10)
    print(model.layer3[i].bn1.bias.min(), model.layer3[i].bn1.bias.max())
    print(model.layer3[i].bn2.bias.min(), model.layer3[i].bn2.bias.max())



print("#"*40)
#a = np.sqrt(model.bn1.weight.detach().numpy())
#b = np.sqrt(model.bn1.running_var.detach().numpy())
#c = a/b
#print(c.min(), c.max())

#print("="*20)

print("="*20)
for i in range(9):
    print("-"*10)
    a = model.layer1[i].bn1.weight.detach().numpy()
    b = np.sqrt(model.layer1[i].bn1.running_var.detach().numpy())
    c = a/b
    print(c.min(), c.max(), c.mean())

    d = model.layer1[i].bn2.weight.detach().numpy()
    e = np.sqrt(model.layer1[i].bn2.running_var.detach().numpy())
    f = d/e
    print(f.min(), f.max(), f.mean())

print("="*20)
for i in range(9):
    print("-"*10)
    a = model.layer2[i].bn1.weight.detach().numpy()
    b = np.sqrt(model.layer2[i].bn1.running_var.detach().numpy())
    c = a/b
    print(c.min(), c.max(), c.mean())

    d = model.layer2[i].bn2.weight.detach().numpy()
    e = np.sqrt(model.layer2[i].bn2.running_var.detach().numpy())
    f = d/e
    print(f.min(), f.max(), f.mean())

print("="*20)
for i in range(9):
    print("-"*10)
    a = model.layer3[i].bn1.weight.detach().numpy()
    b = np.sqrt(model.layer3[i].bn1.running_var.detach().numpy())
    c = a/b
    print(c.min(), c.max(), c.mean())

    d = model.layer3[i].bn2.weight.detach().numpy()
    e = np.sqrt(model.layer3[i].bn2.running_var.detach().numpy())
    f = d/e
    print(f.min(), f.max(), f.mean())
