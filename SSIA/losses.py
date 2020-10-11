import logging
from typing import List

import torch
import torch.nn
from torch import cuda


class SoftMarginRankingLoss(torch.nn.Module):
    """SoftMarginRankingLoss reimplementation of the MarginRankingLoss, but with a Softplus"""

    def __init__(self, margin: float):
        super(SoftMarginRankingLoss, self).__init__()
        self.sp = torch.nn.Softplus(beta=10)
        self.margin = margin

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """(x1 better than x2 by at least the margin) -> 0"""
        # target currently does nothing! It's just here for API reasons
        assert x1.shape == x2.shape, "Shape of the two inputs must be the same."
        mini_batch_size = x1.shape[0]
        result = self.sp((x2 - x1) + self.margin)
        return sum(result) / mini_batch_size


class EfficientRankingLoss(torch.nn.Module):
    def __init__(self, margin: float, softplus: bool = False):
        super(EfficientRankingLoss, self).__init__()
        self.softplus = softplus
        if softplus:
            self.mrloss = SoftMarginRankingLoss(margin)
        else:
            self.mrloss = torch.nn.MarginRankingLoss(margin)
        self.one = torch.Tensor([1]).to(torch.device("cuda" if cuda.is_available() else "cpu"))

    def forward(self, original, x, polarity: str, score: str) -> torch.Tensor:
        loss: List[torch.Tensor] = []
        for idx1, change1 in enumerate(x.keys()):
            logging.debug(f"score\toriginal\t{change1}")
            loss.append(self.mrloss(original[score], x[change1][score], self.one))
            for idx2, change2 in enumerate(x.keys()):
                if idx1 >= idx2:
                    continue
                if polarity == "pos":
                    logging.debug(f"score\t{change1}\t{change2}")
                    loss.append(self.mrloss(x[change1][score], x[change2][score], self.one))
                elif polarity == "neg":
                    logging.debug(f"score\t{change2}\t{change1}")
                    loss.append(self.mrloss(x[change2][score], x[change1][score], self.one))
        return sum(loss)


def h(z: torch.Tensor, T: int = 50):
    """loss balancing function: https://arxiv.org/pdf/2002.04792.pdf"""
    return torch.exp(z / T)
