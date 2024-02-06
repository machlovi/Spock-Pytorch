import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLoss(nn.Module):
    def __init__(self, name=None):
        super(CustomLoss, self).__init__()
        self.name = name

class CosineSimilarityLoss(CustomLoss):
    def forward(self, x, y):
        return torch.mean(1.0 / (1.01 + F.cosine_similarity(x, y)))

class IntraClassLoss(CustomLoss):
    def forward(self, cosine_values):
        return 1.0 / (torch.std(cosine_values) + 0.001)

class BinaryCrossEntropyLoss(CustomLoss):
    def forward(self, attention_score, hidden4):
        return nn.BCEWithLogitsLoss()(attention_score, hidden4)

class LossValues:
    def __init__(self):
        self.offensive_normal_loss = []
        self.toxic_intra_loss = []
        self.non_toxic_intra_loss = []
        self.attention_loss = []

    def update(self, offensive_normal_loss, toxic_intra_loss, non_toxic_intra_loss, attention_loss):
        self.offensive_normal_loss.append(offensive_normal_loss.item())
        self.toxic_intra_loss.append(toxic_intra_loss.item())
        self.non_toxic_intra_loss.append(non_toxic_intra_loss.item())
        self.attention_loss.append(attention_loss.item())

    def get_values(self):
        return {
            'offensive_normal_loss': self.offensive_normal_loss,
            'toxic_intra_loss': self.toxic_intra_loss,
            'non_toxic_intra_loss': self.non_toxic_intra_loss,
            'attention_loss': self.attention_loss
        }
