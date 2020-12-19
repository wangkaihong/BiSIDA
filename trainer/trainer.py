import datetime
from distutils.version import LooseVersion
import math
import os
import shutil
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pytz
import cv2
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import tqdm
from models import losses
import utils
from inspect import isfunction

class Trainer(object):

    def __init__(self, cuda, student_model, teacher_model, style_net, student_optimizer, teacher_optimizer, train_loader, val_loader, out, max_iter, unsup_weight, 
                 clustering_weight, interval_loss_map=1000, class_reweight=False, train_generator=False,style_net_optim=None, style_weight=None, content_weight=None,
                 loss_type='bce', balance_function=None, confidence_thresh=None, confidence_portion=None, tgt_style_method="minmax", src_ave_method="ave", tgt_ave_method="ave",
                 max_portion=None, internal_weight=None, size_average=False, class_dist_reg=True, src_style_alpha=None, source_record=False, src_temperture=1, tgt_temperture=1,
                 tgt_style_alpha=None, interval_validate=None,rampup_function=None, pad=0, src_transfer_rate=0, tgt_transfer_rate=0, only_transfer=False,
                 pseudo_labeling=False, save=False):
        self.cuda = cuda

        self.student_model = student_model
        self.teacher_model = teacher_model
        self.style_net = style_net
        # self.mixup = mixup
        self.student_optimizer = student_optimizer
        self.teacher_optimizer = teacher_optimizer
        self.unsup_weight = unsup_weight
        assert confidence_thresh is None or (confidence_portion is None and max_portion is None)
        self.confidence_thresh=confidence_thresh
        self.confidence_portion = confidence_portion
        self.max_portion = max_portion
        self.src_ave_method = src_ave_method
        self.tgt_ave_method = tgt_ave_method
        self.src_temperture = src_temperture
        self.tgt_temperture = tgt_temperture
        self.class_reweight = class_reweight
        self.rampup_function = rampup_function
        self.clustering_weight = clustering_weight
        self.train_generator = train_generator
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.style_net_optim = style_net_optim
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_type = loss_type
        self.tgt_style_method = tgt_style_method
        self.pad = pad
        self.pseudo_labeling = pseudo_labeling
        self.source_record = source_record
        self.src_style_alpha = src_style_alpha
        self.tgt_style_alpha = tgt_style_alpha
        self.timestamp_start = \
            datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        self.size_average = size_average
        self.internal_weight = internal_weight
        if interval_validate is None:
            self.interval_validate = len(self.train_loader)
        else:
            self.interval_validate = interval_validate
        self.writer = SummaryWriter(out)
        self.interval_loss_map = interval_loss_map
        self.src_transfer_rate = src_transfer_rate
        self.tgt_transfer_rate = tgt_transfer_rate
        self.out = out
        self.only_transfer = only_transfer
        if not os.path.exists(self.out):
            os.makedirs(self.out)

        self.save = save

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'train/acc',
            'train/acc_cls',
            'train/mean_iu',
            'train/fwavacc',
            'valid/loss',
            'valid/acc',
            'valid/acc_cls',
            'valid/mean_iu',
            'valid/fwavacc',
            'elapsed_time',
        ]
        if not os.path.exists(os.path.join(self.out, 'log.csv')):
            with open(os.path.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.best_mean_iu = 0
        self.balance_function = balance_function
        self.class_dist_reg = class_dist_reg
        self.n_class = self.train_loader.dataset.n_class
        self.class_dist = self.train_loader.dataset.src_class_dist

        self.supervised_loss = losses.Supervised_loss(size_average=self.size_average, pad=self.pad)
        self.consistency_loss = losses.Consistency_loss(self.n_class, self.class_dist, self.loss_type, self.balance_function, self.pad, self.pseudo_labeling)
        self.clustering_loss = losses.Clustering_loss(self.clustering_weight, sample_number=10000, loss_type="ce", pad=50)
        self.internal_consistency_loss = losses.Internal_consistency_loss(loss_type='bce')

        self.gpus = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        if len(self.gpus) > 1:
            print("Loss parallelized!")
            self.supervised_loss = nn.DataParallel(self.supervised_loss)
            self.consistency_loss = nn.DataParallel(self.consistency_loss)
            self.clustering_loss = nn.DataParallel(self.clustering_loss)
            self.internal_consistency_loss = nn.DataParallel(self.internal_consistency_loss)

        if cuda:
            self.supervised_loss = self.supervised_loss.cuda()
            self.consistency_loss = self.consistency_loss.cuda()
            self.clustering_loss = self.clustering_loss.cuda()
            self.internal_consistency_loss = self.internal_consistency_loss.cuda()

    def validate(self):
        training = self.teacher_model.training
        self.teacher_model.eval()
        self.student_model.eval()
        
        val_loss = 0
        visualizations = []
        label_trues, label_preds_teacher, label_preds_student = [], [], []
        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration=%d' % self.iteration, ncols=80,
                leave=False):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            with torch.no_grad():
                score_teacher = self.teacher_model(data)[0]
                score_student = self.student_model(data)[0]
                if score_teacher.size()[-2:] != target.size()[-2:]:
                    data = F.interpolate(data, target.size()[-2:])
                    score_teacher = F.interpolate(score_teacher, target.size()[-2:])
                    score_student = F.interpolate(score_student, target.size()[-2:])
            loss = self.supervised_loss(score_teacher, target)

            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while validating')
            val_loss += loss_data / len(data)

            imgs = data.data.cpu()
            lbl_pred_teacher = score_teacher.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_pred_student = score_student.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu()
            for img, lt, lpt, lps in zip(imgs, lbl_true, lbl_pred_teacher, lbl_pred_student):
                img, lt = self.val_loader.dataset.untransform(img, lt)
                lt[lt==255] =self.n_class
                label_trues.append(lt)
                label_preds_teacher.append(lpt)
                label_preds_student.append(lps)
                if len(visualizations) < 9:
                    viz = utils.visualize_segmentation(lbl_pred=lpt, lbl_true=lt, img=img, n_class=self.n_class + 1)
                    visualizations.append(viz)
        metrics_teacher = list(utils.label_accuracy_score(label_trues, label_preds_teacher, self.n_class))
        metrics_student = list(utils.label_accuracy_score(label_trues, label_preds_student, self.n_class))
        few_mean_iu_teacher = np.nanmean(np.delete(metrics_teacher[4], self.train_loader.dataset.few_class_index))
        few_mean_iu_student = np.nanmean(np.delete(metrics_student[4], self.train_loader.dataset.few_class_index))
        metrics_teacher.append(few_mean_iu_teacher)
        metrics_student.append(few_mean_iu_student)

        pred_class_dist_teacher = utils.class_dist_stat(label_trues, label_preds_teacher, self.n_class)
        pred_class_dist_student = utils.class_dist_stat(label_trues, label_preds_student, self.n_class)

        out = os.path.join(self.out, 'visualization_viz')
        if not os.path.exists(out):
            os.makedirs(out)
        out_file = os.path.join(out, 'iter%012d.jpg' % self.iteration)
        cv2.imwrite(out_file, cv2.cvtColor(utils.get_tile_image(visualizations), cv2.COLOR_RGB2BGR))

        val_loss /= len(self.val_loader)

        with open(os.path.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = (
                datetime.datetime.now(pytz.timezone('Asia/Tokyo')) -
                self.timestamp_start).total_seconds()
            # log = [self.epoch, self.iteration] + [''] * 5 + \
            #       [val_loss] + list(metrics) + [elapsed_time]
            # log = map(str, log)

            ius = metrics_teacher[4]
            classes = self.train_loader.dataset.rep_class_names

            report_ius = []
            for i in range(len(ius)):
                report_ius.append(" " + classes[i] + ": " + str(ius[i]))
                self.writer.add_scalar("val_individual/teacher_mIoU_"+classes[i], ius[i], self.iteration)    
            report_ius = ",".join(report_ius)

            self.writer.add_scalar("val/teacher_mIoU", metrics_teacher[2], self.iteration)
            self.writer.add_scalar("val/student_mIoU", metrics_student[2], self.iteration)
            self.writer.add_scalar("val/teacher_fewmIoU", metrics_teacher[5], self.iteration)
            self.writer.add_scalar("val/student_fewmIoU", metrics_student[5], self.iteration)
            self.writer.add_scalar("val/teacher_fwavacc", metrics_teacher[3], self.iteration)
            self.writer.add_scalar("val/student_fwavacc", metrics_student[3], self.iteration)

            self.writer.add_histogram("val/teacher_class_distribution", pred_class_dist_teacher, self.iteration)

            log = "validation: epoch: "+str(self.epoch) + ", iteration: " + str(self.iteration) + \
                ", loss: " + str(loss_data) + ", teahcer: acc: " + str(metrics_teacher[0]) + \
                ", acc_cls: " + str(metrics_teacher[1]) + ", mean_iu: " + str(metrics_teacher[2]) + ", few_mean_iu: " + str(metrics_teacher[5]) + \
                ", fwavacc: " + str(metrics_teacher[3]) + ", time: " + str(elapsed_time) + \
                ", IoU: " + report_ius + '\n' + "student: acc: " + str(metrics_student[0]) + \
                ", acc_cls: " + str(metrics_student[1]) + ", mean_iu: " + str(metrics_student[2]) + ", few_mean_iu: " + str(metrics_student[5]) + \
                ", fwavacc: " + str(metrics_student[3]) + ", time: " + str(elapsed_time) + \
                ", IoU: " + report_ius

            f.write(log + '\n')

            dist = []
            for i in range(len(pred_class_dist_teacher)):
                dist.append(" " + classes[i] + ": " + str(pred_class_dist_teacher[i]))
            log_class_dist_teacher = ",".join(dist)

            f.write("Distribution of predicted classes of the teacher network: " + log_class_dist_teacher + '\n')

            dist = []
            for i in range(len(pred_class_dist_student)):
                dist.append(" " + classes[i] + ": " + str(pred_class_dist_student[i]))
            log_class_dist_student = ",".join(dist)

            f.write("Distribution of predicted classes of the student network: " + log_class_dist_student + '\n')

        mean_iu = metrics_teacher[2]
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
        if training:
            self.teacher_model.train()
            self.student_model.train()

    def train_epoch(self):
        self.teacher_model.train()
        self.student_model.train()
        
        # src_metrics = []
        # src_ius = []

        all_src_lbls = []
        all_src_preds = []

        for batch_idx, (source_image, target_image_1, target_image_2, source_label, target_lbl_1, target_lbl_2, transform_param1, transform_param2, source_full_imgs, target_full_imgs) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):

            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            if batch_idx % 1 == 0 and not batch_idx == 0:
                if self.iteration % self.interval_validate == 0 :#and self.iteration != 0:
                    self.validate()

                    if self.source_record:
                        all_src_lbls = np.concatenate(all_src_lbls, axis=0)
                        all_src_preds = np.concatenate(all_src_preds, axis=0)
                        acc, acc_cls, mean_iu, fwavacc, ius = utils.label_accuracy_score(all_src_lbls, all_src_preds, n_class=self.n_class)
                        few_mean_iu = np.nanmean(np.delete(ius, self.train_loader.dataset.few_class_index))
                        
                        src_metrics = []
                        src_metrics.append((acc, acc_cls, mean_iu, fwavacc, few_mean_iu))
                        src_metrics = np.mean(src_metrics, axis=0)

                        classes = self.train_loader.dataset.rep_class_names
                        report_ius = []
                        for i in range(len(ius)):
                            report_ius.append(" " + classes[i] + ": " + str(ius[i]))
                        report_ius = ",".join(report_ius)
                
                        self.writer.add_scalar("val/source acc", src_metrics[0], self.iteration)
                        self.writer.add_scalar("val/source acc_cls", src_metrics[1], self.iteration)
                        self.writer.add_scalar("val/source mean_iu", src_metrics[2], self.iteration)
                        self.writer.add_scalar("val/source few_mean_iu", src_metrics[4], self.iteration)
                        self.writer.add_scalar("val/source fwavacc", src_metrics[3], self.iteration)

                        with open(os.path.join(self.out, 'log.csv'), 'a') as f:
                            log = "Source Stat: training: epoch: " + str(self.epoch) + ", iteration: " + str(self.iteration) + ", acc: " + str(src_metrics[0]) + \
                                ", acc_cls: " + str(src_metrics[1]) + ", mean_iu: " + str(src_metrics[2]) + ", few_mean_iu: " + str(src_metrics[4]) + \
                                ", fwavacc: " + str(src_metrics[3]) + ", time: " + str(elapsed_time) + ", IoU: " + report_ius

                            f.write(log + '\n')

                        all_src_lbls = []
                        all_src_preds = []

            assert self.student_model.training
            assert self.teacher_model.training

            if self.cuda:
                source_image, target_image_1, target_image_2, source_label = source_image.cuda(), target_image_1.cuda(), target_image_2.cuda(), source_label.cuda()
                source_full_imgs = [s.cuda() for s in source_full_imgs]
                target_full_imgs = [t.cuda() for t in target_full_imgs]

            source_image, target_image_1, target_image_2, source_label = Variable(source_image), Variable(target_image_1), Variable(target_image_2), Variable(source_label)

            if self.src_style_alpha is None:
                src_alpha = np.random.rand()
            elif type(self.src_style_alpha) == list:
                assert len(self.src_style_alpha) == 2
                src_alpha = np.random.uniform(self.src_style_alpha[0], self.src_style_alpha[1])
            elif isfunction(self.src_style_alpha):
                # src_alpha = np.random.uniform(0, self.src_style_alpha(float(self.epoch + 1) / float(self.max_epoch)))
                min_alpha = min(self.src_style_alpha(float(self.epoch + 1) / float(self.max_epoch)), self.src_style_alpha(float(self.epoch) / float(self.max_epoch)))
                max_alpha = max(self.src_style_alpha(float(self.epoch + 1) / float(self.max_epoch)), self.src_style_alpha(float(self.epoch) / float(self.max_epoch)))
                src_alpha = np.random.uniform(min_alpha, max_alpha)
            else:
                src_alpha = self.src_style_alpha

            if self.tgt_style_alpha is None:
                target_alpha = np.random.rand()
            elif type(self.tgt_style_alpha) == list:
                assert len(self.tgt_style_alpha) == 2
                target_alpha = np.random.uniform(self.tgt_style_alpha[0], self.tgt_style_alpha[1])
            elif isfunction(self.tgt_style_alpha):
                # target_alpha = np.random.uniform(0, self.tgt_style_alpha(float(self.epoch + 1) / float(self.max_epoch)))
                min_alpha = min(self.tgt_style_alpha(float(self.epoch + 1) / float(self.max_epoch)), self.tgt_style_alpha(float(self.epoch) / float(self.max_epoch)))
                max_alpha = max(self.tgt_style_alpha(float(self.epoch + 1) / float(self.max_epoch)), self.tgt_style_alpha(float(self.epoch) / float(self.max_epoch)))
                target_alpha = np.random.uniform(min_alpha, max_alpha)
            else:
                target_alpha = self.tgt_style_alpha

            source_transfer = np.random.rand() > (1 - self.src_transfer_rate)
            target_transfer = np.random.rand() > (1 - self.tgt_transfer_rate)

            if not self.train_generator:
                with torch.no_grad():                
                    if source_transfer:
                        source_transferred = [self.style_net(source_image, t, alpha=src_alpha) for t in target_full_imgs]
                        source_image = [self.train_loader.dataset.fit_image(s[2], cuda=self.cuda) for s in source_transferred]
                    else:
                        source_image = [source_image]

                    if target_transfer:
                        if self.tgt_style_method == "maxmin":
                            target_image_1 = [self.train_loader.dataset.fit_image(self.style_net(target_image_1, s, alpha=max(target_alpha, 1 - target_alpha))[2], cuda=self.cuda) for s in source_full_imgs]
                            target_image_2 = [self.train_loader.dataset.fit_image(self.style_net(target_image_2, s, alpha=min(target_alpha, 1 - target_alpha))[2], cuda=self.cuda) for s in source_full_imgs]
                        elif self.tgt_style_method == "minmax":
                            target_image_1 = [self.train_loader.dataset.fit_image(self.style_net(target_image_1, s, alpha=min(target_alpha, 1 - target_alpha))[2], cuda=self.cuda) for s in source_full_imgs]
                            target_image_2 = [self.train_loader.dataset.fit_image(self.style_net(target_image_2, s, alpha=max(target_alpha, 1 - target_alpha))[2], cuda=self.cuda) for s in source_full_imgs]
                        elif self.tgt_style_method == "only_1":
                            target_image_1 = [self.train_loader.dataset.fit_image(self.style_net(target_image_1, s, alpha=target_alpha)[2], cuda=self.cuda) for s in source_full_imgs]
                            target_image_2 = [target_image_2]
                        elif self.tgt_style_method == "only_2":
                            target_image_1 = [target_image_1]
                            target_2_transferred = [self.style_net(target_image_2, s, alpha=target_alpha) for s in source_full_imgs]
                            target_image_2 = [self.train_loader.dataset.fit_image(t[2], cuda=self.cuda) for t in target_2_transferred]
                        elif self.tgt_style_method == "half_1":
                            target_image_1 = [self.train_loader.dataset.fit_image(self.style_net(target_image_1, s, alpha=target_alpha / 2.)[2], cuda=self.cuda) for s in source_full_imgs]
                            target_image_2 = [self.train_loader.dataset.fit_image(self.style_net(target_image_2, s, alpha=target_alpha)[2], cuda=self.cuda) for s in source_full_imgs]
                        else:
                            raise ValueError
                    else:
                        target_image_1 = [target_image_1]
                        target_image_2 = [target_image_2]

            else:
                transfer_loss = torch.tensor(0.)
                if self.cuda:
                    transfer_loss = transfer_loss.cuda()
                if source_transfer:
                    source_transferred = [self.style_net(source_image, t, alpha=src_alpha) for t in target_full_imgs]
                    source_image = [self.train_loader.dataset.fit_image(s[2].detach(), cuda=self.cuda) for s in source_transferred]
                    source_content_loss = sum([i[0] for i in source_transferred]) * self.content_weight
                    source_style_loss = sum([i[1] for i in source_transferred]) * self.style_weight * src_alpha
                    transfer_loss = transfer_loss + source_content_loss + source_style_loss
                else:
                    source_image = [source_image]

                if target_transfer:
                    if self.tgt_style_method == "maxmin":
                        raise ValueError
                        target_image_1 = [self.train_loader.dataset.fit_image(self.style_net(target_image_1, s, alpha=max(target_alpha, 1 - target_alpha))[2], cuda=self.cuda) for s in source_full_imgs]
                        target_image_2 = [self.train_loader.dataset.fit_image(self.style_net(target_image_2, s, alpha=min(target_alpha, 1 - target_alpha))[2], cuda=self.cuda) for s in source_full_imgs]
                    elif self.tgt_style_method == "minmax":
                        raise ValueError
                        target_image_1 = [self.train_loader.dataset.fit_image(self.style_net(target_image_1, s, alpha=min(target_alpha, 1 - target_alpha))[2], cuda=self.cuda) for s in source_full_imgs]
                        target_image_2 = [self.train_loader.dataset.fit_image(self.style_net(target_image_2, s, alpha=max(target_alpha, 1 - target_alpha))[2], cuda=self.cuda) for s in source_full_imgs]
                    elif self.tgt_style_method == "only_1":
                        raise ValueError
                        target_image_1 = [self.train_loader.dataset.fit_image(self.style_net(target_image_1, s, alpha=target_alpha)[2], cuda=self.cuda) for s in source_full_imgs]
                        target_image_2 = [target_image_2]
                    elif self.tgt_style_method == "only_2":
                        target_image_1 = [target_image_1]
                        target_2_transferred = [self.style_net(target_image_2, s, alpha=target_alpha) for s in source_full_imgs]
                        target_image_2 = [self.train_loader.dataset.fit_image(t[2].detach(), cuda=self.cuda) for t in target_2_transferred]
                        target_content_loss = sum([i[0] for i in target_2_transferred]) * self.content_weight
                        target_style_loss = sum([i[1] for i in target_2_transferred]) * self.style_weight * target_alpha
                        transfer_loss = transfer_loss + target_content_loss + target_style_loss
                    elif self.tgt_style_method == "half_1":
                        raise ValueError
                        target_image_1 = [self.train_loader.dataset.fit_image(self.style_net(target_image_1, s, alpha=target_alpha / 2.)[2], cuda=self.cuda) for s in source_full_imgs]
                        target_image_2 = [self.train_loader.dataset.fit_image(self.style_net(target_image_2, s, alpha=target_alpha)[2], cuda=self.cuda) for s in source_full_imgs]
                    else:
                        raise ValueError
                else:
                    target_image_1 = [target_image_1]
                    target_image_2 = [target_image_2]

                if source_transfer or target_transfer:
                    self.style_net_optim.zero_grad()
                    transfer_loss.backward()
                    self.style_net_optim.step()


            self.student_optimizer.zero_grad()
            source_score = [self.student_model(s) for s in source_image]#[0][0]

            if self.src_ave_method == "max":
                N, C, H, W = source_score[0][0].size()
                score_prob = [F.softmax(s[0], dim=1) for s in source_score]
                max_map1 = [torch.max(s, 1)[0] for s in score_prob] 
                max_sum1 = sum(max_map1).view(N, 1, H, W).repeat(1, C, 1, 1)
                max_map1 = [m / max_sum1 for m in max_map1]
                source_score = sum([source_score[j][0] * max_map1[j] for j in range(len(source_score))])
            elif self.src_ave_method == "negative_max":
                N, C, H, W = source_score[0][0].size()
                score_prob = [F.softmax(s[0], dim=1) for s in source_score]
                max_map1 = [1 - (torch.max(s, 1)[0]) for s in score_prob] 
                max_sum1 = sum(max_map1).view(N, 1, H, W).repeat(1, C, 1, 1)
                max_map1 = [m / max_sum1 for m in max_map1]
                source_score = sum([source_score[j][0] * max_map1[j] for j in range(len(source_score))])
            elif self.src_ave_method == "ce":
                N, C, H, W = source_score[0][0].size()
                score_prob = [F.softmax(s[0], dim=1) for s in source_score]
                ce_map1 = [-torch.sum(losses.robust_crossentropy(s, s), 1) for s in score_prob] 
                ce_sum1 = sum(ce_map1).view(N, 1, H, W).repeat(1, C, 1, 1)
                ce_map1 = [m / ce_sum1 for m in ce_map1]
                source_score = sum([source_score[j][0] * ce_map1[j] for j in range(len(source_score))])
            elif self.src_ave_method == "negative_ce":
                N, C, H, W = source_score[0][0].size()
                score_prob = [F.softmax(s[0], dim=1) for s in source_score]
                ce_map1 = [(1 + torch.sum(losses.robust_crossentropy(s, s), 1)) for s in score_prob] 
                ce_sum1 = sum(ce_map1).view(N, 1, H, W).repeat(1, C, 1, 1)
                ce_map1 = [m / ce_sum1 for m in ce_map1]
                source_score = sum([source_score[j][0] * ce_map1[j] for j in range(len(source_score))])
            elif self.src_ave_method == "ave":
                source_score = sum([source_score[j][0] for j in range(len(source_score))])
            elif self.src_ave_method == "ave_":
                source_score = sum([source_score[j][0] for j in range(len(source_score))]) / len(source_score) 
            else:
                raise ValueError

            target_1_score = [self.student_model(t) for t in target_image_1]
            with torch.no_grad():
                target_2_score = [self.teacher_model(t) for t in target_image_2]

            # internal_target_1_score = [sum([target_1_score[j][i] for j in range(len(target_1_score))]) / (len(target_1_score)) for i in range(1, len(target_1_score[0]))] # temperture
            # internal_target_2_score = [sum([target_2_score[j][i] for j in range(len(target_2_score))]) / (len(target_2_score)) for i in range(1, len(target_2_score[0]))]

            # if self.tgt_ave_method == "max":
            #     N, C, H, W = target_1_score[0][0].size()
            #     target_1_score_prob = [F.softmax(t[0], dim=1) for t in target_1_score]
            #     target_2_score_prob = [F.softmax(t[0], dim=1) for t in target_2_score]
            #     max_map1 = [torch.max(t, 1)[0] for t in target_1_score_prob] 
            #     max_sum1 = sum(max_map1).view(N, 1, H, W).repeat(1, C, 1, 1)
            #     max_map1 = [m / max_sum1 for m in max_map1]
            #     max_map2 = [torch.max(t, 1)[0] for t in target_2_score_prob]
            #     max_sum2 = sum(max_map2).view(N, 1, H, W).repeat(1, C, 1, 1)
            #     max_map2 = [m / max_sum2 for m in max_map2]
            #     target_1_score = sum([target_1_score[j][0] * max_map1[j] for j in range(len(target_1_score))])
            #     target_2_score = sum([target_2_score[j][0] * max_map2[j] for j in range(len(target_2_score))]) * self.tgt_temperture
            # elif self.tgt_ave_method == "negative_max":
            #     N, C, H, W = target_1_score[0][0].size()
            #     target_1_score_prob = [F.softmax(t[0], dim=1) for t in target_1_score]
            #     target_2_score_prob = [F.softmax(t[0], dim=1) for t in target_2_score]
            #     max_map1 = [1 - torch.max(t, 1)[0] for t in target_1_score_prob] 
            #     max_sum1 = sum(max_map1).view(N, 1, H, W).repeat(1, C, 1, 1)
            #     max_map1 = [m / max_sum1 for m in max_map1]
            #     max_map2 = [1 - torch.max(t, 1)[0] for t in target_2_score_prob]
            #     max_sum2 = sum(max_map2).view(N, 1, H, W).repeat(1, C, 1, 1)
            #     max_map2 = [m / max_sum2 for m in max_map2]
            #     target_1_score = sum([target_1_score[j][0] * max_map1[j] for j in range(len(target_1_score))])
            #     target_2_score = sum([target_2_score[j][0] * max_map2[j] for j in range(len(target_2_score))]) * self.tgt_temperture
            # elif self.tgt_ave_method == "ce":
            #     N, C, H, W = target_1_score[0][0].size()
            #     target_1_score_prob = [F.softmax(t[0], dim=1) for t in target_1_score]
            #     target_2_score_prob = [F.softmax(t[0], dim=1) for t in target_2_score]
            #     ce_map1 = [-torch.sum(losses.robust_crossentropy(t, t), 1) for t in target_1_score_prob] 
            #     ce_sum1 = sum(ce_map1).view(N, 1, H, W).repeat(1, C, 1, 1)
            #     ce_map1 = [m / ce_sum1 for m in ce_map1]
            #     ce_map2 = [-torch.sum(losses.robust_crossentropy(t, t), 1) for t in target_2_score_prob] 
            #     ce_sum2 = sum(ce_map2).view(N, 1, H, W).repeat(1, C, 1, 1)
            #     ce_map2 = [m / ce_sum2 for m in ce_map2]
            #     target_1_score = sum([target_1_score[j][0] * ce_map1[j] for j in range(len(target_1_score))]) 
            #     target_2_score = sum([target_2_score[j][0] * ce_map2[j] for j in range(len(target_2_score))]) * self.tgt_temperture
            # elif self.tgt_ave_method == "negative_ce":
            #     N, C, H, W = target_1_score[0][0].size()
            #     target_1_score_prob = [F.softmax(t[0], dim=1) for t in target_1_score]
            #     target_2_score_prob = [F.softmax(t[0], dim=1) for t in target_2_score]
            #     ce_map1 = [1 + torch.sum(losses.robust_crossentropy(t, t), 1) for t in target_1_score_prob] 
            #     ce_sum1 = sum(ce_map1).view(N, 1, H, W).repeat(1, C, 1, 1)
            #     ce_map1 = [m / ce_sum1 for m in ce_map1]
            #     ce_map2 = [1 + torch.sum(losses.robust_crossentropy(t, t), 1) for t in target_2_score_prob] 
            #     ce_sum2 = sum(ce_map2).view(N, 1, H, W).repeat(1, C, 1, 1)
            #     ce_map2 = [m / ce_sum2 for m in ce_map2]
            #     target_1_score = sum([target_1_score[j][0] * ce_map1[j] for j in range(len(target_1_score))]) 
            #     target_2_score = sum([target_2_score[j][0] * ce_map2[j] for j in range(len(target_2_score))]) * self.tgt_temperture
            # elif self.tgt_ave_method == "ave":
            #     target_1_score = sum([target_1_score[j][0] for j in range(len(target_1_score))]) # no tempeture handling for training targets
            #     target_2_score = sum([target_2_score[j][0] for j in range(len(target_2_score))]) * self.tgt_temperture
            # elif self.tgt_ave_method == "ave_":
            #     target_1_score = sum([target_1_score[j][0] for j in range(len(target_1_score))]) / len(target_1_score) 
            #     target_2_score = sum([target_2_score[j][0] for j in range(len(target_2_score))]) / len(target_2_score) * self.tgt_temperture
            # else:
            #     raise ValueError
            
            # supervised loss
            sup_loss = self.supervised_loss(source_score, source_label)

            # consistency loss
            target_1_score_prob = sum([F.softmax(target_1_score[j][0], dim=1) for j in range(len(target_1_score))]) / len(target_1_score) 
            target_2_score_prob = sum([F.softmax(target_2_score[j][0], dim=1) for j in range(len(target_2_score))]) / len(target_2_score)

            # temperture

            temped_score = target_2_score_prob.pow(1. / self.tgt_temperture)
            target_2_score_prob = temped_score / temped_score.sum(1)
            
            if self.rampup_function:
                current_unsup_weight = self.rampup_function(self.unsup_weight, (float(self.epoch) + 1) / float(self.max_epoch))
            else:
                current_unsup_weight = self.unsup_weight

            if self.confidence_portion and self.max_portion:
                if self.class_reweight:
                    current_portion = self.confidence_portion + (float(self.epoch) / float(self.max_epoch)) * (self.max_portion - self.confidence_portion)
                    N, D, H, W = target_2_score_prob.size()
                    conf_tea = target_2_score_prob.transpose(0,1).contiguous().view(D, -1)
                    size = N * H * W
                    self.confidence_thresh = torch.sort(conf_tea, dim=1, descending=True)[0][:, int(current_portion * size)].numpy()
                else:
                    current_portion = self.confidence_portion + (float(self.epoch) / float(self.max_epoch)) * (self.max_portion - self.confidence_portion)
                    conf_tea = torch.max(target_2_score_prob, 1)[0]
                    size = np.prod(list(conf_tea.size()))
                    self.confidence_thresh = torch.sort(conf_tea.view((-1)),descending=True)[0][int(current_portion * size)].item()
            # else:
            #     self.confidence_thresh = torch.as_tensor(self.confidence_thresh).repeat(target_1_score_prob.size()[0])

            unsup_loss, unsup_mask_count, conf_cate, aug_loss, aug_loss_dist ,masked_aug_loss_dist, unsup_mask = \
                    self.consistency_loss(target_1_score_prob, target_2_score_prob, self.confidence_thresh, rampup_weight=current_unsup_weight) 
            
            # clustering loss
            clust_loss = self.clustering_loss(target_1_score_prob)

            # internal consistency loss
            # interal_unsup_loss, internal_unsup_mask_count = self.internal_consistency_loss(self.internal_weight, internal_target_1_score, internal_target_2_score, self.confidence_thresh, start_from=0)

            sup_loss = torch.sum(sup_loss)
            unsup_loss = torch.sum(unsup_loss)
            unsup_mask_count = torch.sum(unsup_mask_count)
            clust_loss = torch.sum(clust_loss)
            # interal_unsup_loss = torch.sum(interal_unsup_loss)
            # internal_unsup_mask_count = torch.sum(internal_unsup_mask_count)
            if self.only_transfer and not target_transfer:
                loss = sup_loss + clust_loss
            else:
                loss = sup_loss + unsup_loss + clust_loss #+ interal_unsup_loss

            loss /= len(source_image[0])

            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')
            loss.backward()
            self.student_optimizer.step()
            self.teacher_optimizer.step()

            # stat
            metrics = []
            lbl_pred = source_score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = source_label.data.cpu().numpy()
            
            if self.source_record:
                all_src_lbls.append(lbl_true)
                all_src_preds.append(lbl_pred)
            
            acc, acc_cls, mean_iu, fwavacc, ius = \
                utils.label_accuracy_score(
                    lbl_true, lbl_pred, n_class=self.n_class)
            few_mean_iu = np.nanmean(np.delete(ius, self.train_loader.dataset.few_class_index))

            metrics.append((acc, acc_cls, mean_iu, fwavacc, few_mean_iu))
            metrics = np.mean(metrics, axis=0)
            # src_metrics.append(metrics)
            # src_ius.append(ius)
            classes = self.train_loader.dataset.rep_class_names

            if not conf_cate is None:
                conf_cate = conf_cate.data.cpu().numpy()
                conf_cate_list = utils._fast_cate_hist(conf_cate, self.n_class)
                report_conf = []
                for i in range(len(conf_cate_list)):
                    report_conf.append(" " + classes[i] + ": " + str(conf_cate_list[i]))
                report_conf = ",".join(report_conf)
            else:
                report_conf = "N/A"


            report_ius = []
            for i in range(len(ius)):
                report_ius.append(" " + classes[i] + ": " + str(ius[i]))
            report_ius = ",".join(report_ius)

            if  self.save and self.iteration % 1000 == 0:
                p = os.path.join(self.out, 'model'+str(self.iteration)+'.pt')
                torch.save(self.teacher_model.state_dict(), p)

            if  self.iteration % self.interval_loss_map == 0 and self.iteration != 0:
                src_img = self.train_loader.dataset.untransform(source_image[0][0].data.cpu())
                vis_target_img_1 = self.train_loader.dataset.untransform(target_image_1[0][0].data.cpu())
                vis_target_img_2 = self.train_loader.dataset.untransform(target_image_2[0][0].data.cpu())
                vis_source_lbl = source_label[0].cpu().numpy()
                vis_source_lbl[vis_source_lbl==255] = self.n_class
                vis_target_lbl_1 = target_lbl_1[0].numpy()
                vis_target_lbl_1[vis_target_lbl_1==255] = self.n_class
                vis_target_lbl_2 = target_lbl_2[0].numpy()
                vis_target_lbl_2[vis_target_lbl_2==255] = self.n_class
                src_pred = source_score.data.max(1)[1][0].cpu().numpy()
                vis_target_pred_stu = target_1_score_prob.data.max(1)[1][0].cpu().numpy()
                vis_target_pred_tea = target_2_score_prob.data.max(1)[1][0].cpu().numpy()

                viz = utils.visualize_segmentation_aug(
                        src_img=src_img, tgt_img_1=vis_target_img_1, tgt_img_2=vis_target_img_2,
                        src_pred=src_pred, lbl_pred_stu=vis_target_pred_stu, lbl_pred_tea=vis_target_pred_tea, 
                        src_lbl=vis_source_lbl, lbl_true_1=vis_target_lbl_1, lbl_true_2=vis_target_lbl_2,
                        n_class=self.n_class + 1, aug_loss=aug_loss[0].detach().cpu().numpy(),
                        aug_loss_dist=aug_loss_dist[0].detach().cpu().numpy(), 
                        masked_aug_loss_dist=masked_aug_loss_dist[0].detach().cpu().numpy(),
                        unsup_mask=unsup_mask[0].detach().cpu().numpy())

                out = os.path.join(self.out, 'loss_map')
                if not os.path.exists(out):
                    os.makedirs(out)
                out_file = os.path.join(out, 'iter%012d.jpg' % self.iteration)
                cv2.imwrite(out_file, cv2.cvtColor(utils.get_tile_image([viz]), cv2.COLOR_RGB2BGR))

            with open(os.path.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.datetime.now(pytz.timezone('Asia/Tokyo')) -
                    self.timestamp_start).total_seconds()
                # log = [self.epoch, self.iteration] + [loss_data] + \
                #     metrics.tolist() + [''] * 5 + [elapsed_time]
                # log = map(str, log)
                self.writer.add_scalar("train/loss", loss_data, self.iteration)
                if self.train_generator:
                    self.writer.add_scalar("train/transfer loss", (transfer_loss).data.item(), self.iteration)
                self.writer.add_scalar("train/sup loss", (sup_loss/len(source_image[0])).data.item(), self.iteration)
                self.writer.add_scalar("train/unsup loss", (unsup_loss/len(source_image[0])).data.item(), self.iteration)
                self.writer.add_scalar("train/clustering loss", (clust_loss/len(source_image)).data.item(), self.iteration)
                # self.writer.add_scalar("train/internal loss", (interal_unsup_loss/len(source_image)).data.item(), self.iteration)
                self.writer.add_scalar("train/train_mIoU", metrics[2], self.iteration)
                self.writer.add_scalar("train/train_few_mIoU", metrics[4], self.iteration)
                self.writer.add_scalar("train/current_unsup_weight", current_unsup_weight, self.iteration)

                if self.train_generator:
                    log = "training: epoch: " + str(self.epoch) + ", iteration: " + str(self.iteration) + \
                        ", loss: " + str(loss_data) + ", transfer loss: " + str(transfer_loss.data.item()) + \
                        ", sup loss: " + str((sup_loss/len(source_image[0])).data.item()) + \
                        ", unsup loss: " + str((unsup_loss/len(source_image[0])).data.item()) + \
                        ", acc: " + str(metrics[0]) + \
                        ", acc_cls: " + str(metrics[1]) + ", mean_iu: " + str(metrics[2]) + ", few_mean_iu: " + str(metrics[4]) + \
                        ", fwavacc: " + str(metrics[3]) + ", time: " + str(elapsed_time) + \
                        ", unsup_mask_count: " + str(unsup_mask_count.data.item()) + \
                        ", confidence over threshold: " + report_conf + ", IoU: " + report_ius

                else:
                    log = "training: epoch: " + str(self.epoch) + ", iteration: " + str(self.iteration) + \
                        ", loss: " + str(loss_data) + ", sup loss: " + str((sup_loss/len(source_image[0])).data.item()) + \
                        ", unsup loss: " + str((unsup_loss/len(source_image[0])).data.item()) + \
                        ", acc: " + str(metrics[0]) + \
                        ", acc_cls: " + str(metrics[1]) + ", mean_iu: " + str(metrics[2]) + ", few_mean_iu: " + str(metrics[4]) + \
                        ", fwavacc: " + str(metrics[3]) + ", time: " + str(elapsed_time) + \
                        ", unsup_mask_count: " + str(unsup_mask_count.data.item()) + \
                        ", confidence over threshold: " + report_conf + ", IoU: " + report_ius

                f.write(log + '\n') 
                #                        ", clustering loss: " + str((clust_loss/len(source_image[0])).data.item()) + \
                #                        ", internal loss: " + str((interal_unsup_loss/len(source_image[0])).data.item()) + \


            if self.iteration >= self.max_iter:
                break

    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch,
                                 desc='Train', ncols=80):
            self.max_epoch = max_epoch
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break

