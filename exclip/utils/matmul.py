import torch
from torch import Tensor


def split_mm(
    A: Tensor, B: Tensor, splits: int = 2, device: torch.device = torch.device("cuda:0")
):
    As = torch.split(A, A.shape[0] // splits)
    Bs = torch.split(B, B.shape[1] // splits, dim=1)
    rows = []
    for a in As:
        a = a.to(device)
        row = []
        for b in Bs:
            b = b.to(device)
            c = a @ b
            c = c.cpu()
            row.append(c)
        row = torch.cat(row, dim=1)
        rows.append(row)
    C = torch.cat(rows, dim=0)
    return C
