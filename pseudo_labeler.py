# from model.costs import tanh_clip
import torch
import torch.nn.functional as F
import tqdm


class TargetPseudoLabeler(object):
    def __init__(self, num_classes, tclip=10., lp_size=4096, lp_iterations=10):
        self.num_classes = num_classes
        self.tclip = tclip
        self.lp_size = lp_size
        self.lp_iterations = lp_iterations

    def pseudo_label_tgt(self, src_test_collection, tgt_test_collection):
        """
        pseudo label target samples.
        Args:
            src_test_collection: (n_src, n_rkhs)
            tgt_test_collection: (n_tgt, n_rkhs)

        Returns:
            tgt_pseudo_label : (n_tgt) contains pseudo label of entire target samples
        """
        src_features = src_test_collection['features']
        tgt_features = tgt_test_collection['features']
        src_true_labels = src_test_collection['true_labels']

        # print(src_true_labels)

        n_src = src_features.size(0)
        n_tgt = tgt_features.size(0)

        assert src_features.size(1) == tgt_features.size(1)

        num_batches = (n_src + n_tgt) // self.lp_size + 1
        src_batch_sizes = [n_src // num_batches + 1] * (n_src % num_batches) \
                          + [n_src // num_batches] * (num_batches - n_src % num_batches)
        tgt_batch_sizes = [n_tgt // num_batches + 1] * (n_tgt % num_batches) \
                          + [n_tgt // num_batches] * (num_batches - n_tgt % num_batches)

        src_perm_indices = torch.randperm(n_src).cuda()
        tgt_perm_indices = torch.randperm(n_tgt).cuda()

        tgt_pseudo_label = torch.zeros(n_tgt, self.num_classes).cuda()

        src_index = 0
        tgt_index = 0

        for i in tqdm.tqdm(range(num_batches), desc='Target pseudo labeling', leave=False, ascii=True):
            src_batch_indices = src_perm_indices[src_index:src_index + src_batch_sizes[i]]
            tgt_batch_indices = tgt_perm_indices[tgt_index:tgt_index + tgt_batch_sizes[i]]
            src_index += src_batch_sizes[i]
            tgt_index += tgt_batch_sizes[i]
            src_batch_features = src_features[src_batch_indices]
            tgt_batch_features = tgt_features[tgt_batch_indices]
            src_batch_true_labels = src_true_labels[src_batch_indices]

            n_src_batch = src_batch_features.size(0)
            n_tgt_batch = tgt_batch_features.size(0)

            batch_features = torch.cat([src_batch_features, tgt_batch_features], dim=0)
            batch_features = F.normalize(batch_features, dim=1)
            raw_scores = torch.mm(batch_features, batch_features.transpose(0, 1)).float()
            raw_scores *= self.tclip

            # max_scores = torch.max(raw_scores, dim=1, keepdim=True)[0]
            # exp_scores = torch.exp(raw_scores - max_scores)
            # prop_scores = F.normalize(exp_scores, p=1, dim=1)
            prop_scores = F.softmax(raw_scores, dim=1)

            prop_ts = prop_scores[n_src_batch:, :n_src_batch]
            prop_tt = prop_scores[n_src_batch:, n_src_batch:]

            # print('propts, proptt')
            # print(prop_ts.size())
            # print(prop_tt.size())

            # print(torch.sum(prop_ts, dim=1))
            # print(torch.sum(prop_tt, dim=1))

            # print(src_batch_true_labels)
            src_one_hot_label = torch.zeros(n_src_batch, self.num_classes).cuda()
            src_one_hot_label.scatter_(1, src_batch_true_labels.unsqueeze(1), 1)

            # print('src_one_hot_label')
            # print(src_one_hot_label)
            # print(torch.sum(src_one_hot_label, dim=1))
            # initialize tgt pseudo label
            # tgt_batch_pseudo_label = torch.ones(n_tgt_batch, self.num_classes).cuda() / self.num_classes

            # label propagation
            # for _ in range(self.lp_iterations):
            #     tgt_batch_pseudo_label = torch.mm(prop_ts, src_one_hot_label) \
            #                              + torch.mm(prop_tt, tgt_batch_pseudo_label)

            tgt_batch_pseudo_label = torch.mm(
                torch.inverse(torch.eye(n_tgt_batch).cuda() - prop_tt),
                torch.mm(prop_ts, src_one_hot_label))

            # print('tgt_batch_pseudo_label')
            # print(torch.sum(tgt_batch_pseudo_label, dim=1))
            # print(tgt_batch_pseudo_label)

            tgt_pseudo_label[tgt_batch_indices] = tgt_batch_pseudo_label

        return tgt_pseudo_label
