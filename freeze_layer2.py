import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim

from preact_resnet_test67 import resnet56 as net

train_loader = DataLoader(
                datasets.CIFAR10(
                        "./data/CIFAR10",
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                ),
                        ),
                batch_size=128, shuffle=True)#, pin_memory=True)


test_loader = DataLoader(
                datasets.CIFAR10(
                        './data/CIFAR10',
                        train=False,
                        download=True,
                        transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                ),
                        ),
                batch_size=64, shuffle=False)#, pin_memory=True)


device = torch.device("cuda:0")
model = net().to(device)

path = "/data/ymh/gpu_test/resnet56_LN.pth"
checkpoint = torch.load(path)

from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in checkpoint['model_state_dict'].items():
    key_list = k.split(".")
    if 'alpha2' in key_list or 'beta2' in key_list:
        new_state_dict[k] = v
        print(k)

model.load_state_dict(new_state_dict, strict=False)
#model.load_state_dict(checkpoint['model_state_dict'], strict=False)


fix_index = []
index = 0
for key in model.state_dict():
    print(index, key)
    key_list = key.split(".")
    if 'alpha2' in key_list or 'beta2' in key_list:
        print(index, key_list)
        fix_index.append(index)
    
    index +=1
    
param_list = list(model.parameters())
for index in fix_index:
    param_list[index].requires_grad = False

print(len(param_list))

optimizer = optim.Adam(model.parameters(), lr=0.01)
#optimizer = optim.SGD(model.parameters(), lr=0.1)
#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.000001)
#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,150], gamma=0.1)
criterion = nn.CrossEntropyLoss()

model.train()
best_acc = 0
for epoch in range(300):
    runnning_loss = 0
    for x, y in train_loader:
        optimizer.zero_grad()
               
        output = model(x.float().to(device))
        loss = criterion(output, y.long().to(device))
        loss.backward()
        optimizer.step()
        runnning_loss += loss.item()
        #print(loss.item())
        
    runnning_loss /= len(train_loader)
    print("[Epoch:%d] [Loss:%f]" % ((epoch+1), runnning_loss), end=" ")
    
    accuracy = 0
    with torch.no_grad():
        model.eval()
        correct = 0
        for x, y in test_loader:
            output = model(x.float().to(device))
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(y.long().to(device).view_as(pred)).sum().item()
                
        accuracy = correct / len(test_loader.dataset)

    if accuracy >= best_acc:
        print("[Accuracy:%f] **Best**" % accuracy)
        best_acc = accuracy
    else:
        print("[Accuracy:%f]" % accuracy)






