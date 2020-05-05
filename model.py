import torch
import torch.nn as nn
from torchvision.models import vgg16


class NIAA(nn.Module):
    """Neural Image Aesthetic Assessment model"""

    def __init__(self, base_model: nn.module = vgg16(pretrained=False), num_classes: int = 10):
        super(NIAA, self).__init__()
        self.num_classes = num_classes
        self.scores = torch.IntTensor(list(range(1, num_classes + 1)))  # [1..10]
        self.features = base_model.features  # vgg16.features
        # fmt: off
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(in_features=25088,
            out_features=num_classes), nn.Softmax())
        # fmt: on

        def _forwardSingle(self, x, score: bool):  # for AVA
            out = self.features(x)
            out = out.view(out.size(0), -1)
            out = self.classifier(out)

            if score == True:  # return score 1..10
                return out.dot(scores)
            elif score == False:  # return distribution like NIMA
                return out

        def _forwardSiamese(self, x1, x2):  #  for pexels
            out1 = self._forwardSingle(x1, score=True)
            out2 = self._forwardSingle(x2, score=True)
            return (out1, out2)

        def forward(self, x1, x2, mode: str):
            if mode == "distribution":
                return _forwardSingle(x1, score=False)
            elif mode == "score":
                return _forwardSingle(x1, score=True)
            elif mode == "siamese":
                return _forwardSiamese(x1, x2)
            else:
                raise ValueError(f"unsupported mode")


class Earth_Movers_Distance_Loss(nn.Module):
    """Earth Mover's Distance"""

    def __init__(self):
        super(Earth_Movers_Distance_Loss, self).__init__()

    def _single_emd_loss(self, p, q, r=2):
        """
        Earth Mover's Distance of one sample

        Args:
            p: true distribution of shape num_classes × 1
            q: estimated distribution of shape num_classes × 1
            r: norm parameter
        """
        assert p.shape == q.shape, "Length of the two distribution must be the same"
        length = p.shape[0]
        emd_loss = 0.0
        for i in range(1, length + 1):
            emd_loss += torch.abs(sum(p[:i] - q[:i])) ** r
        return (emd_loss / length) ** (1.0 / r)

    def forward(self, p, q, r=2):
        """
        Earth Mover's Distance on a batch

        Args:
            p: true distribution of shape mini_batch_size × num_classes × 1
            q: estimated distribution of shape mini_batch_size × num_classes × 1
            r: norm parameters
        """
        assert p.shape == q.shape, "Shape of the two distribution batches must be the same."
        mini_batch_size = p.shape[0]
        loss_vector = []
        for i in range(mini_batch_size):
            loss_vector.append(_single_emd_loss(p[i], q[i], r=r))
        return sum(loss_vector) / mini_batch_size


class Distance_Loss(nn.Module):
    """Distance_Loss"""

    def __init__(self):
        super(Distance_Loss, self).__init__()

    def _single_distance_loss(self, x1: float, x2: float, epsilon: float = 0.1):
        """x1 better than x2 by at least epsilon == 0"""
        return torch.max((x2 - x1) + epsilon, 0)

    def forward(self, x1, x2, epsilon: float = 0.1):
        """x1 better than x2 by at least epsilon == 0"""
        assert x1.shape == x2.shape, "Shape of the two inputs must be the same."
        mini_batch_size = x1.shape[0]
        loss_vector = []
        for i in range(mini_batch_size):
            loss_vector.append(_single_distance_loss(x1[i], x2[i], epsilon=epsilon))
        return sum(loss_vector) / mini_batch_size
