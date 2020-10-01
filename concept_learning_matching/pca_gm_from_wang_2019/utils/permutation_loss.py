import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """
    Cross entropy loss between two permutations.
    """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, pred_perm, gt_perm, all_matches):
        batch_num = pred_perm.shape[0]

        pred_perm = pred_perm.to(dtype=torch.float32)
        gt_perm = gt_perm.to(dtype=torch.float32)


        assert torch.all((pred_perm >= 0) * (pred_perm <= 1))
        assert torch.all((gt_perm >= 0) * (gt_perm <= 1))

        loss = torch.tensor(0.).to(pred_perm.device)
        n_sum = torch.zeros_like(loss)
        if len(all_matches) == 0:
            loss.requires_grad = True
            return loss
        for b in range(batch_num):
            for idx, val in enumerate(all_matches):

                loss += F.binary_cross_entropy(
                    pred_perm[b, val[0], val[1]],
                    gt_perm[b, val[0], val[1]]
                )
            # loss += F.binary_cross_entropy(
            #     pred_perm[b, :pred_ns[b], :gt_ns[b]],
            #     gt_perm[b, :pred_ns[b], :gt_ns[b]],
            #     reduction='sum')
            # n_sum += pred_ns[b].to(n_sum.dtype).to(pred_perm.device)

        return loss / len(all_matches)
