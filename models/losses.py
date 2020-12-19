import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import math

def robust_binary_crossentropy(pred, tgt):
    inv_tgt = -tgt + 1.0
    inv_pred = -pred + 1.0 + 1e-6
    return -(tgt * torch.log(pred + 1.0e-6) + inv_tgt * torch.log(inv_pred))

def robust_crossentropy(pred, tgt):
    return -(tgt * torch.log(pred + 1.0e-6))

def msl(pred):
    return -(pred * pred) / 2

def iic(z, zt):

    EPS = 1.0e-6
    n, c, h, w = z.size()
    z = z.permute(0, 2, 3, 1).contiguous().view(-1, c) # (n*h*w,c)
    zt = zt.permute(0, 2, 3, 1).contiguous().view(-1, c) # (n*h*w,c)
    P = (z.unsqueeze(2) * zt.unsqueeze(1)) # (n*h*w,c,1) * (n*h*w,1,c) -> (n*h*w,c,c)
    P = ((P + P.transpose(1,2)) / 2) / P.sum(dim=(1,2), keepdim=True) # (n*h*w,c,c)
    P[(P < EPS).data] = EPS
    Pi = P.sum(dim=2).view(n*h*w, c, 1).expand(n*h*w, c, c) # (n*h*w,c,c)
    Pj = P.sum(dim=1).view(n*h*w, 1, c).expand(n*h*w, c, c) # (n*h*w,c,c)
    result = -(P.matmul(torch.log(Pi) + torch.log(Pj) - torch.log(P))).sum(dim=(1,2)) # (n*h*w, c,c)
    result = result.view(n, h, w)
    return result

class Supervised_loss(nn.Module):
    def __init__(self, size_average, pad=50):

        super(Supervised_loss, self).__init__()
        self.size_average = size_average
        self.pad = pad

    def forward(self, input, target, weight=None):
        # input: (n, c, h, w), target: (n, h, w)
        n, c, h, w = input.size()
        input = input[:, :, self.pad:h - self.pad, self.pad:w - self.pad]
        target = target[:, self.pad:h - self.pad, self.pad:w - self.pad]
        n, c, h, w = input.size()
        # log_p: (n, c, h, w)
        log_p = F.log_softmax(input, dim=1)
        # log_p: (n*h*w, c)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
        log_p = log_p[(target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0) * (target.view(n, h, w, 1).repeat(1, 1, 1, c) != 255)]
        log_p = log_p.view(-1, c)
        # target: (n*h*w,)
        mask = (target >= 0) * (target != 255)
        target = target[mask]
        loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
        if self.size_average:
            loss /= mask.data.sum()
            # loss /= (n * h * w)
        return loss.unsqueeze(dim=0)

class Consistency_loss(nn.Module):

    def __init__(self, n_class, class_dist, loss_type='bce', balance_function=None, pad=50, pseudo_labeling=False):

        super(Consistency_loss, self).__init__()
        self.n_class = n_class 
        self.class_dist = class_dist
        self.loss_type = loss_type 
        self.balance_function = balance_function
        self.pad = pad
        self.pseudo_labeling = pseudo_labeling

    def forward(self, stu_out, tea_out, confidence_thresh=0.96837722, rampup_weight=None):

        # N, D, H, W = stu_out.size()

        # stu_out = stu_out[:, :, self.pad:H - self.pad, self.pad:W - self.pad]
        # tea_out = tea_out[:, :, self.pad:H - self.pad, self.pad:W - self.pad]
        N, D, H, W = stu_out.size()

        unsup_mask_count, conf_cate = torch.tensor(0.), None

        ### new balance method ### 
        pred_tea = torch.argmax(tea_out, 1)#[0]
        dist_pred_tea = pred_tea.clone().float()
        # self.class_dist = self.class_dist + 1.0e-4
        for c in range(self.n_class):
                # dist_pred_tea[pred_tea==c] = pow((1 / self.class_dist[c]),1/3.) / 10.
            dist_pred_tea[pred_tea==c] = self.balance_function(self.class_dist[c])

        if self.pseudo_labeling:
            real_tea_out = torch.zeros((N, D, H, W)).cuda()
            real_tea_out = real_tea_out.scatter_(1, pred_tea.view(N, 1, H, W), 1)
        else:
            real_tea_out = tea_out

        if self.loss_type == 'bce':
            aug_loss = robust_binary_crossentropy(stu_out, real_tea_out) # (1, 23, 320, 640)
        if self.loss_type == 'ce':
            aug_loss = robust_crossentropy(stu_out, real_tea_out) # (1, 23, 320, 640)
        if self.loss_type == 'ce_':
            aug_loss = robust_crossentropy(stu_out, stu_out) # (1, 23, 320, 640)
        if self.loss_type == 'iic':
            raise ValueError
            aug_loss = iic(stu_out, real_tea_out) # (1, 23, 320, 640)
        if self.loss_type == 'msl':
            aug_loss = msl(stu_out) # (1, 23, 320, 640)
        else:
            d = stu_out - real_tea_out
            aug_loss = d * d

        aug_loss[ :, :, :self.pad, :] = 0
        aug_loss[ :, :, H - self.pad:, :] = 0
        aug_loss[ :, :, :, :self.pad] = 0
        aug_loss[ :, :, :, W - self.pad:] = 0

        aug_loss = aug_loss.sum(dim=1)

        aug_loss_dist = aug_loss * dist_pred_tea
        ### new balance method ###

        n, h, w = aug_loss_dist.size()
        conf_tea = torch.max(tea_out, 1)[0]
        conf_tea_cate = torch.argmax(tea_out, 1)
        if confidence_thresh is not None:
            if isinstance(confidence_thresh, float) or isinstance(confidence_thresh, int):
                unsup_mask = (conf_tea > confidence_thresh)
                conf_cate = conf_tea_cate[unsup_mask]
                unsup_mask = unsup_mask.float()
                unsup_mask_count = unsup_mask.sum()
                masked_aug_loss_dist = aug_loss_dist * unsup_mask
            else:
                assert len(confidence_thresh) == D
                confidence_thresh_mat = confidence_thresh.view(1, D, 1, 1).expand(*list(tea_out.size()))
                tea_out_reweighted = torch.max(tea_out / confidence_thresh_mat, 1)[0] 
                unsup_mask = (tea_out_reweighted > 1.)
                conf_cate = conf_tea_cate[unsup_mask]
                unsup_mask = unsup_mask.float()
                unsup_mask_count = unsup_mask.sum()
                masked_aug_loss_dist = aug_loss_dist * unsup_mask
        else:
            unsup_mask = (conf_tea > 0)
            masked_aug_loss_dist = aug_loss_dist

        unsup_loss = masked_aug_loss_dist.mean() # 

        if rampup_weight is not None:
            unsup_loss = unsup_loss * rampup_weight

        ret = [unsup_loss.unsqueeze(dim=0), 
                unsup_mask_count.unsqueeze(dim=0), 
                conf_cate, 
                aug_loss, 
                aug_loss_dist, 
                masked_aug_loss_dist, 
                unsup_mask]
        return ret

class Clustering_loss(nn.Module):

    def __init__(self, clustering_weight, sample_number=30000, loss_type="ce", pad=50):
        
        super(Clustering_loss, self).__init__()
        self.clustering_weight = clustering_weight
        self.sample_number = sample_number 
        self.loss_type = loss_type
        self.pad = pad

    def forward(self, output):

        n, c, h, w = output.size()
        output = output[:, :, self.pad:h - self.pad, self.pad:w - self.pad].contiguous()
        n, c, h, w = output.size()

        if self.loss_type == "ent":
            loss = robust_crossentropy(output, output).mean()

            return loss
            
        sample = torch.randperm(h * w)[:self.sample_number]
        prob = output.view(n, c, -1)[:,:,sample]
        prob_t = prob.permute(0, 2, 1)
        sim = torch.matmul(prob_t, prob)
        ind = torch.argmax(sim, axis=2)

        selector = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), prob_t.size(2))
        nearest = torch.gather(prob_t, 1, selector) # (n, sample_number, class_number)

        if self.loss_type == "ce":
            loss = robust_crossentropy(prob_t, nearest)
        elif self.loss_type == "bce":
            loss = robust_binary_crossentropy(prob_t, nearest)
        else:
            raise ValueError

        loss = loss.mean() * self.clustering_weight
        return loss.unsqueeze(dim=0)

class Internal_consistency_loss(nn.Module):

    def __init__(self, loss_type='bce'):

        super(Internal_consistency_loss, self).__init__()
        self.loss_type = loss_type

    def layer_loss(self, stu_out, tea_out, confidence_thresh=0.96837722):

        if self.loss_type == 'bce':
            internal_loss = robust_binary_crossentropy(stu_out, tea_out) # (1, 23, 320, 640)
        if self.loss_type == 'ce':
            internal_loss = robust_crossentropy(stu_out, tea_out) # (1, 23, 320, 640)
        if self.loss_type == 'ce_':
            internal_loss = robust_crossentropy(stu_out, stu_out) # (1, 23, 320, 640)
        if self.loss_type == 'iic':
            internal_loss = iic(stu_out, tea_out) # (1, 23, 320, 640)
        else:
            d = stu_out - tea_out
            internal_loss = d * d

        n, c, h, w = internal_loss.size()
        conf_tea = torch.max(tea_out, 1)[0]
        unsup_mask  = (conf_tea > confidence_thresh).float()
        unsup_mask_count = unsup_mask.sum()
        unsup_mask = unsup_mask.unsqueeze(dim=1).repeat(1, c, 1, 1)
        internal_loss = (internal_loss * unsup_mask).mean() # 

        return internal_loss, unsup_mask_count


    def forward(self, internal_weight, target_1_score, target_2_score, confidence_thresh=0.96837722, start_from=1):

        if internal_weight is None:
            internal_weight = [0., 0., 0.]
        assert len(internal_weight) == (len(target_1_score)-start_from)

        internal_unsup_mask_count = []
        interal_unsup_loss = []

        for i in range(start_from, len(target_1_score)):
            target_1_score_prob = F.softmax(target_1_score[i], dim=1)
            target_2_score_prob = F.softmax(target_2_score[i], dim=1) # confidence thresholding should be used in the inter-layer feature outputs, what about class-balancing?

            [single_inter_unsup_loss, inter_unsup_mask_count] = self.layer_loss(target_1_score_prob, target_2_score_prob, confidence_thresh)
            interal_unsup_loss.append(internal_weight[i-1] * single_inter_unsup_loss)
            internal_unsup_mask_count.append(inter_unsup_mask_count)

        ret = [sum(interal_unsup_loss).unsqueeze(dim=0), sum(internal_unsup_mask_count).unsqueeze(dim=0)]
        return ret