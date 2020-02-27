"""
This code is based on https://github.com/Philip-Bachman/amdim-public/blob/master/costs.py
"""

import torch
import torch.nn as nn


def tanh_clip(x, clip_val=10.):
    """
    soft clip values to the range [-clip_val, +clip_val]
    """
    if clip_val is not None:
        x_clip = clip_val * torch.tanh((1. / clip_val) * x)
    else:
        x_clip = x
    return x_clip


class LossMultiNCE(nn.Module):
    def __init__(self, tclip=10.):
        super(LossMultiNCE, self).__init__()
        self.tclip = tclip

    def forward(self, r_src, r_tgt, pos_matrix, neg_matrix):
        """
            Compute the NCE scores for predicting r_src->r_trg.
            Input:
              r_src    : (n_batch, n_rkhs, n_dims, n_dims)
              r_trg    : (n_keys, n_rkhs, n_dims, n_dims)
              pos_matrix : (n_batch, n_keys)
              neg_matrix : (n_batch, n_keys)
            Output:
              query_to_key_loss  : scalar
              contrast_norm_loss : scalar
        """
        assert r_src.size(1) == r_tgt.size(1)
        assert r_src.size(2) == r_src.size(3) == r_tgt.size(2) == r_tgt.size(3)

        n_batch = r_src.size(0)
        n_keys = r_tgt.size(0)
        n_rkhs = r_src.size(1)
        n_dims = r_src.size(2)

        # (n_batch * n_dims * n_dims, n_rkhs)
        r_src = r_src.permute(0, 2, 3, 1).reshape(-1, n_rkhs)

        # (n_rkhs, n_keys * n_dims * n_dims)
        r_tgt = r_tgt.permute(1, 0, 2, 3).reshape(n_rkhs, -1)

        # reshape pos_matrix and neg_matrix for ease-of-use
        # (n_batch, n_dims * n_dims, n_keys, n_dims * n_dims)
        pos_matrix = pos_matrix.unsqueeze(2).unsqueeze(3) \
            .expand(-1, -1, n_dims * n_dims, n_dims * n_dims).permute(0, 2, 1, 3).float()
        neg_matrix = neg_matrix.unsqueeze(2).unsqueeze(3) \
            .expand(-1, -1, n_dims * n_dims, n_dims * n_dims).permute(0, 2, 1, 3).float()

        # compute src->trg raw scores for batch
        # (n_batch * n_dims * n_dims, n_keys * n_dims * n_dims)
        raw_scores = torch.mm(r_src, r_tgt).float()

        # (n_batch, n_dims * n_dims, n_keys, n_dims * n_dims)
        raw_scores = raw_scores.reshape(n_batch, n_dims * n_dims, n_keys, n_dims * n_dims)
        raw_scores = raw_scores / n_rkhs ** 0.5
        contrast_norm_loss = 5e-2 * (raw_scores ** 2.).mean()  # TODO
        raw_scores = tanh_clip(raw_scores, clip_val=self.tclip)

        '''
        pos_scores includes scores for all the positive samples
        neg_scores includes scores for all the negative samples, with
        scores for positive samples set to the min score (-self.tclip here)
        '''
        # (n_batch, n_dims * n_dims, n_keys, n_dims * n_dims)
        pos_scores = (pos_matrix * raw_scores)

        # (n_batch * n_dims * n_dims, n_keys * n_dims * n_dims)
        pos_scores = pos_scores.reshape(n_batch * n_dims * n_dims, -1)

        # (n_batch, n_dims * n_dims, n_keys, n_dims * n_dims)
        # neg_scores = (neg_matrix * raw_scores) - (pos_matrix * self.tclip)
        neg_scores = (neg_matrix * raw_scores) - ((1. - neg_matrix) * self.tclip)

        # (n_batch * n_dims * n_dims, n_keys * n_dims * n_dims)
        neg_scores = neg_scores.reshape(n_batch * n_dims * n_dims, -1)

        # (n_batch * n_dims * n_dims, n_keys * n_dims * n_dims)
        pos_matrix = pos_matrix.reshape(n_batch * n_dims * n_dims, -1)

        # (n_batch * n_dims * n_dims, n_keys * n_dims * n_dims)
        neg_matrix = neg_matrix.reshape(n_batch * n_dims * n_dims, -1)

        '''
        for each set of positive examples P_i, compute the max over scores
        for the set of negative samples N_i that are shared across P_i
        '''
        # (n_batch * n_dims * n_dims, 1)
        neg_maxes = torch.max(neg_scores, dim=1, keepdim=True)[0]

        '''
        compute a "partial, safe sum exp" over each negative sample set N_i,
        to broadcast across the positive samples in P_i which share N_i
        -- size will be (n_batch * n_dims * n_dims, 1)
        '''
        neg_sumexp = (neg_matrix * torch.exp(neg_scores - neg_maxes)).sum(dim=1, keepdim=True)

        '''
        use broadcasting of neg_sumexp across the scores in P_i, to compute
        the log-sum-exps for the denominators in the NCE log-softmaxes
        -- size will be (n_batch * n_dims * n_dims, n_keys * n_dims * n_dims)
        '''
        all_logsumexp = torch.log(torch.exp(pos_scores - neg_maxes) + neg_sumexp)

        # compute numerators for the NCE log-softmaxes
        # (n_batch * n_dims * n_dims, n_keys * n_dims * n_dims)
        pos_shiftexp = pos_scores - neg_maxes

        # compute the final log-softmax scores for NCE...
        # (n_batch * n_dims * n_dims, n_keys * n_dims * n_dims)
        nce_scores = pos_matrix * (pos_shiftexp - all_logsumexp)

        query_to_key_loss = -nce_scores.sum() / pos_matrix.sum()

        return query_to_key_loss, contrast_norm_loss
