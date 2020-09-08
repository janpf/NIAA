import torch
import logging


class PerfectLoss(torch.nn.Module):
    """Perfect Image gets score 1"""

    def __init__(self):
        super(PerfectLoss, self).__init__()
        self.mse = torch.nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x: torch.Tensor):
        """if the image is unedited force a score of 1"""
        mini_batch_size = x.shape[0]
        return self.mse(torch.squeeze(x).float(), torch.ones(mini_batch_size).to(self.device))  #  pull it to 1


class SoftMarginRankingLoss(torch.nn.Module):
    """SoftMarginRankingLoss reimplementation of the MarginRankingLoss, but with a Softplus"""

    def __init__(self):
        super(SoftMarginRankingLoss, self).__init__()
        self.sp = torch.nn.Softplus(beta=10)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, margin: float) -> torch.Tensor:
        """(x1 better than x2 by at least the margin) -> 0"""
        assert x1.shape == x2.shape, "Shape of the two inputs must be the same."
        mini_batch_size = x1.shape[0]
        result = self.sp((x2.float() - x1.float()) + margin)
        return sum(result) / mini_batch_size


class EfficientRankingLoss(torch.nn.Module):
    def __init__(self):
        super(EfficientRankingLoss, self).__init__()
        self.smrloss = SoftMarginRankingLoss()

    def forward(self, original, x, polarity: str, score: str, margin: float) -> torch.Tensor:
        loss = []
        for idx1, change1 in enumerate(x.keys()):
            logging.debug(f"score\toriginal\t{change1}")
            loss.append(self.smrloss(original[score], x[change1][score], margin))
            for idx2, change2 in enumerate(x.keys()):
                if idx1 >= idx2:
                    continue
                if polarity == "pos":
                    logging.debug(f"score\t{change1}\t{change2}")
                    loss.append(self.smrloss(x[change1][score], x[change2][score], margin))
                elif polarity == "neg":
                    logging.debug(f"score\t{change2}\t{change1}")
                    loss.append(self.smrloss(x[change2][score], x[change1][score], margin))
        return sum(loss)


def h(z: torch.Tensor, T: int = 50):
    """loss balancing function: https://arxiv.org/pdf/2002.04792.pdf"""
    return torch.exp(z / T)
