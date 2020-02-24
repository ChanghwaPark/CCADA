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

    def forward(self, r_src, r_trg, pos_matrix, neg_matrix):
        """
                Compute the NCE scores for predicting r_src->r_trg.
                Input:
                  r_src    : (n_batch, n_rkhs)
                  r_trg    : (n_keys, n_rkhs)
                  pos_matrix : (n_batch, n_keys)
                  neg_matrix : (n_batch, n_keys)
                Output:
                  query_to_key_loss  : scalar
                  contrast_norm_loss : scalar
                """
        r_trg = r_trg.transpose(1, 0)
        n_rkhs = r_src.size(1)
        # neg_matrix = 1. - pos_matrix

        # compute src->trg raw scores for batch
        # (n_batch, n_keys)
        raw_scores = torch.mm(r_src, r_trg).float()

        # (n_batch, n_keys)
        raw_scores = raw_scores / n_rkhs ** 0.5
        contrast_norm_loss = 5e-2 * (raw_scores ** 2.).mean()  # TODO
        raw_scores = tanh_clip(raw_scores, clip_val=self.tclip)

        '''
        pos_scores includes scores for all the positive samples
        neg_scores includes scores for all the negative samples, with
        scores for positive samples set to the min score (-self.tclip here)
        '''
        # (n_batch, n_keys)
        pos_scores = (pos_matrix * raw_scores)

        # (n_batch, n_keys)
        # neg_scores = (neg_matrix * raw_scores) - (pos_matrix * self.tclip)
        neg_scores = (neg_matrix * raw_scores) - ((1. - neg_matrix) * self.tclip)

        '''
        for each set of positive examples P_i, compute the max over scores
        for the set of negative samples N_i that are shared across P_i
        '''
        # (n_batch, 1)
        neg_maxes = torch.max(neg_scores, dim=1, keepdim=True)[0]

        '''
        compute a "partial, safe sum exp" over each negative sample set N_i,
        to broadcast across the positive samples in P_i which share N_i
        -- size will be (n_batch, 1)
        '''
        neg_sumexp = (neg_matrix * torch.exp(neg_scores - neg_maxes)).sum(dim=1, keepdim=True)

        '''
        use broadcasting of neg_sumexp across the scores in P_i, to compute
        the log-sum-exps for the denominators in the NCE log-softmaxes
        -- size will be (n_batch, n_keys)
        '''
        all_logsumexp = torch.log(torch.exp(pos_scores - neg_maxes) + neg_sumexp)

        # compute numerators for the NCE log-softmaxes
        # (n_batch, n_keys)
        pos_shiftexp = pos_scores - neg_maxes

        # compute the final log-softmax scores for NCE...
        # (n_batch, n_keys)
        nce_scores = pos_matrix * (pos_shiftexp - all_logsumexp)

        query_to_key_loss = -nce_scores.sum() / pos_matrix.sum()

        return query_to_key_loss, contrast_norm_loss
