import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim

from torch.optim.swa_utils import AveragedModel, SWALR
from resnet_test1 import resnet56 as net

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
'''
pretrained_path = "c://results/resnet56_pretrained.pth" # your path
if device == "cuda:0":
    checkpoint = torch.load(pretrained_path)
else:
    checkpoint = torch.load(pretrained_path, map_location=lambda storage, location: 'cpu')
model.load_state_dict(checkpoint['model_state_dict'])
'''
optimizer = optim.Adam(model.parameters(), lr=0.01)
#optimizer = optim.SGD(model.parameters(), lr=0.1)
#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,150], gamma=0.1)
criterion = nn.CrossEntropyLoss()

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
swa_model = AveragedModel(model)
swa_start = 200
swa_scheduler = SWALR(optimizer, swa_lr=0.001)

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
    
    if epoch > swa_start:
        swa_model.update_parameters(model)
        swa_scheduler.step()
    else:
        scheduler.step()
    
    runnning_loss /= len(train_loader)
    print("[Epoch:%d] [Loss:%f]" % ((epoch+1), runnning_loss), end=" ")
    
    accuracy = 0
    with torch.no_grad():
        model.eval()
        correct = 0
        for x, y in test_loader:
            if epoch > swa_start:
                output = swa_model(x.float().to(device))
            else:
                output = model(x.float().to(device))
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(y.long().to(device).view_as(pred)).sum().item()
                
        accuracy = correct / len(test_loader.dataset)
        '''
        if save:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': losses[epoch].item()}, "/home/cscoi/MH/resnet56_real4.pth")
            save_again = False
            print("[Accuracy:%f]" % accuracy)
            print("Saved early")
            break
        ''' 
        if accuracy >= best_acc:
            print("[Accuracy:%f] **Best**" % accuracy)
            best_acc = accuracy
        else:
            print("[Accuracy:%f]" % accuracy)
        model.train()
        
    #scheduler.step()