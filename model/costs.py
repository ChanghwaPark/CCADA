"""
This code is based on https://github.com/Philip-Bachman/amdim-public/blob/master/costs.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def tanh_clip(x, clip_val=10.):
    """
    soft clip values to the range [-clip_val, +clip_val]
    """
    if clip_val is not None:
        x_clip = clip_val * torch.tanh((1. / clip_val) * x)
    else:
        x_clip = x
    return x_clip


class NCEMIMulti(nn.Module):
    # def __init__(self, tclip=20.):
    def __init__(self, temperature=0.1):
        super(NCEMIMulti, self).__init__()
        # self.tclip = tclip
        self.temperature = temperature

    def _model_scores(self, r_src, r_trg, mask_mat):
        """
        Compute the NCE scores for predicting r_src->r_trg.
        Input:
          r_src    : (n_batch, n_rkhs)
          r_trg    : (n_rkhs, n_keys * n_locs)
          mask_mat : (n_batch, n_keys)
        Output:
          nce_scores : (n_batch, n_keys * n_locs)
          pos_scores : (n_batch, n_keys * n_locs)
          lgt_reg    : scalar
        """
        n_batch = mask_mat.size(0)
        n_keys = mask_mat.size(1)
        n_locs = r_trg.size(1) // n_keys
        n_rkhs = r_src.size(1)

        # reshape mask_mat for ease-of-use
        # (n_batch, n_keys, n_locs)
        mask_pos = mask_mat.unsqueeze(dim=2).expand(-1, -1, n_locs).float()
        mask_neg = 1. - mask_pos

        r_src = F.normalize(r_src, p=2, dim=1)
        r_trg = F.normalize(r_trg, p=2, dim=0)

        # compute src->trg raw scores for batch
        # (n_batch, n_keys * n_locs)
        raw_scores = torch.mm(r_src, r_trg).float()

        # (n_batch, n_keys, n_locs)
        raw_scores = raw_scores.reshape(n_batch, n_keys, n_locs)
        # raw_scores = raw_scores / n_rkhs ** 0.5
        # lgt_reg = 5e-2 * (raw_scores ** 2.).mean()  # TODO
        lgt_reg = 0.
        # raw_scores = tanh_clip(raw_scores, clip_val=self.tclip)
        raw_scores = torch.div(raw_scores, self.temperature)
        '''
        pos_scores includes scores for all the positive samples
        neg_scores includes scores for all the negative samples, with
        scores for positive samples set to the min score (-self.tclip here)
        '''
        # # (n_batch, n_locs)
        # pos_scores = (mask_pos * raw_scores).sum(dim=1)
        # (n_batch, n_keys, n_locs)
        pos_scores = (mask_pos * raw_scores)

        # (n_batch, n_keys * n_locs)
        pos_scores = pos_scores.reshape(n_batch, -1)

        # (n_batch, n_keys, n_locs)
        # neg_scores = (mask_neg * raw_scores) - (mask_pos * self.tclip)
        neg_scores = (mask_neg * raw_scores) - (mask_pos * (1 / self.temperature))

        # (n_batch, n_keys * n_locs)
        neg_scores = neg_scores.reshape(n_batch, -1)

        # (n_batch, n_keys * n_locs)
        mask_pos = mask_pos.reshape(n_batch, -1)

        # (n_batch, n_keys * n_locs)
        mask_neg = mask_neg.reshape(n_batch, -1)
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
        neg_sumexp = (mask_neg * torch.exp(neg_scores - neg_maxes)).sum(dim=1, keepdim=True)

        '''
        use broadcasting of neg_sumexp across the scores in P_i, to compute
        the log-sum-exps for the denominators in the NCE log-softmaxes
        -- size will be (n_batch, n_keys * n_locs)
        '''
        all_logsumexp = torch.log(torch.exp(pos_scores - neg_maxes) + neg_sumexp)

        # compute numerators for the NCE log-softmaxes
        # (n_batch, n_keys * n_locs)
        pos_shiftexp = pos_scores - neg_maxes

        # compute the final log-softmax scores for NCE...
        # (n_batch, n_keys * n_locs)
        # nce_scores = pos_shiftexp - all_logsumexp
        nce_scores = mask_pos * (pos_shiftexp - all_logsumexp)

        return nce_scores, mask_pos, lgt_reg

    def _loss_g2l(self, r_src, r_trg, mask_mat):
        # compute the nce scores for these features
        nce_scores, mask_pos, lgt_reg = self._model_scores(r_src, r_trg, mask_mat)
        loss_g2l = -nce_scores.sum() / mask_pos.sum()
        return loss_g2l, lgt_reg

    # def forward(self, r1_src_1, r1_src_2, r7_trg_1, r7_trg_2, mask_mat, mode):
    def forward(self, r1_src, rn_trg, mask_mat, mode):
        assert (mode in ['train', 'viz'])
        if mode == 'train':
            # compute values required for visualization
            # compute costs for 1->7 prediction
            loss_1tn, lgt_reg = self._loss_g2l(r1_src, rn_trg, mask_mat)
            return loss_1tn, lgt_reg
        else:
            # compute values to use for visualizations
            # nce_scores, raw_scores, lgt_reg = \
            #     self._model_scores(r1_src_1, r7_trg_2[0], mask_mat)
            # nce_scores, raw_scores, lgt_reg = self._model_scores(r1_src_1, r7_trg_2, mask_mat)
            # nce_scores, raw_scores, lgt_reg = self._model_scores(r1_src, rn_trg, mask_mat)
            # return nce_scores, raw_scores
            return None, None


class LossMultiNCE(nn.Module):
    # def __init__(self, tclip=10.):
    def __init__(self, temperature=0.1):
        super(LossMultiNCE, self).__init__()
        # initialize the dataparallel nce computer (magic!)
        # self.nce_func = NCEMIMulti(tclip=tclip)
        self.nce_func = NCEMIMulti(temperature=temperature)
        # self.nce_func = nn.DataParallel(self.nce_func)

    @staticmethod
    def _sample_src_ftr(r_cnv, masks):
        # get feature dimensions
        n_batch = r_cnv.size(0)
        n_rkhs = r_cnv.size(1)
        if masks is not None:
            # subsample from conv-ish r_cnv to get a single vector
            mask_idx = torch.randint(0, masks.size(0), (n_batch,))
            r_cnv = torch.masked_select(r_cnv, masks[mask_idx])
        # flatten features for use as globals in glb->lcl nce cost
        r_vec = r_cnv.reshape(n_batch, n_rkhs)
        return r_vec

    def forward(self, r1_x, rn_x, mask_mat):
        # compute feature dimensions
        n_batch = int(r1_x.size(0))
        n_rkhs = int(r1_x.size(1))
        n_keys = int(rn_x.size(0))
        # make masking matrix to help compute nce costs
        # mask_mat = torch.eye(n_batch).cuda()

        # sample "source" features for glb->lcl predictions
        # r1_src = self._sample_src_ftr(r1_x, None)
        r1_src = r1_x

        # before shape: (n_batch, n_rkhs, n_dim, n_dim), (n_queue, n_rkhs, n_dim, n_dim)
        # rn_tgt = torch.cat([rn_x, negative_keys], dim=0)
        # after shape: (n_batch + n_queue, n_rkhs, n_dim, n_dim)

        # before shape: (n_batch + n_queue, n_rkhs, n_dim, n_dim)
        # rn_tgt = rn_tgt.permute(1, 0, 2, 3).reshape(n_rkhs, -1)
        rn_tgt = rn_x.permute(1, 0, 2, 3).reshape(n_rkhs, -1)
        # after shape: (n_rkhs, n_keys * n_dim * n_dim)

        # negative_mask_mat = torch.zeros(n_batch, n_queue).cuda()
        # all_mask_mat = torch.cat([mask_mat, negative_mask_mat], dim=1)

        # compute nce for multiple infomax costs on a single GPU
        # loss_1tn, lgt_reg = self.nce_func(r1_src, rn_tgt, all_mask_mat, mode='train')
        loss_1tn, lgt_reg = self.nce_func(r1_src, rn_tgt, mask_mat, mode='train')

        loss_1tn = loss_1tn.mean()
        # lgt_reg = lgt_reg.mean()
        return loss_1tn, lgt_reg
