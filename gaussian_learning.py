import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import numpy as np

from preact_resnet_simclr import resnet56, Encoder

########################################################################################################################
class NTXentLoss(nn.Module):
    def __init__(self, device, batch_size, temperature=0.5, use_cosine_similarity=True):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.labels = torch.zeros(2 * self.batch_size).long().to(self.device)

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        print("mask is created.")
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        #labels = torch.zeros(2 * self.batch_size).long().to(self.device)
        loss = self.criterion(logits, self.labels)

        return loss / (2 * self.batch_size)
########################################################################################################################
device = torch.device("cuda:0")
model = resnet56().to(device)
encoding = Encoder().to(device)

params = list(model.parameters()) + list(encoding.parameters())
optimizer = optim.Adam(model.parameters(), lr=0.01)


contrastive_loss = NTXentLoss(device, 512)

for epoch in range(50000):
    running_loss = 0
    for i in range(390):
        x1 = torch.from_numpy(np.random.normal(0,0.5,(128,3,32,32)))
        x2 = torch.from_numpy(np.random.normal(0,0.5,(128,3,32,32)))
        optimizer.zero_grad()
        
        x1 = x1.float().to(device)
        x2 = x2.float().to(device)
        
        out1 = encoding(model(x1))
        out2 = encoding(model(x2))
        
        loss = contrastive_loss(out1, out2)
        loss.backward()
        
        optimizer.step()
        running_loss += loss.item()
        #print(loss.item())
        
    running_loss /= 390
    print("[Epoch:%d] [loss:%f]" % (epoch+1, running_loss))

    model_name = os.path.join(save_path, "self_sup.pth")
    torch.save({'model_state_dict': model.state_dict()}, model_name)