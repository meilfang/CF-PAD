import sys
sys.path.append('..')

import numpy as np
import os
import csv
import copy
import logging
from tqdm import tqdm
import math
import random
from collections import defaultdict
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from torch.nn.utils import clip_grad_norm_

from dataset import TrainDataset, TestDataset, ApplyWeightedRandomSampler
from utils import  AvgrageMeter, performances_cross_db, compute_video_score
from model import MixStyleResCausalModel
#import metric_utils

torch.autograd.set_detect_anomaly(True)

def save_checkpoint(save_path, epoch, model, loss, lr_scheduler, optimizer):
    save_state = {'model': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'lr_scheduler': lr_scheduler.state_dict(),
              'loss': loss,
              'epoch': epoch}
    torch.save(save_state, save_path)

def run_training(train_csv, test_csv, log_file, output_path, args):
    train_dataset = TrainDataset(csv_file=train_csv, input_shape=args.input_shape)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,  sampler=ApplyWeightedRandomSampler(train_csv),
                                num_workers=4, pin_memory=True, drop_last=True)

    test_dataset = TestDataset(csv_file=test_csv, input_shape=args.input_shape)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    checkpoint_save_dir = os.path.join('checkpoints/', args.prefix)
    print('Checkpoints folders', checkpoint_save_dir)
    if not os.path.isdir(checkpoint_save_dir):
        os.makedirs(checkpoint_save_dir)

    model = torch.nn.DataParallel(MixStyleResCausalModel(model_name=args.model_name, pretrained=args.pretrain, num_classes=args.num_classes, prob=args.prob))
    model = model.cuda()

    optimizer = torch.optim.SGD([
            {'params': model.module.feature_extractor.parameters()},
            {'params': model.module.classifier.parameters(), 'lr': float(args.lr[1])},
            ], lr=float(args.lr[0]), momentum=0.9, weight_decay=0.0005)

    #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,45], gamma=0.5)
    #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=15)
    cen_criterion = torch.nn.CrossEntropyLoss().cuda()
    scaler = GradScaler()

    flooding_impactor = 0.001
    for epoch in range(1, args.max_epoch+1):
        if os.path.isfile(os.path.join(checkpoint_save_dir, '{}.pth'.format(epoch))):
            model.load_state_dict(torch.load(os.path.join(checkpoint_save_dir, '{}.pth'.format(epoch))))
            continue
        else:

            print('-------------- train ------------------------')
            model.train()
            loss_total = AvgrageMeter()
            loss_1_total = AvgrageMeter()
            loss_2_total = AvgrageMeter()

            progress_bar = tqdm(train_loader)
            for i, data in enumerate(progress_bar):
                progress_bar.set_description('Epoch ' + str(epoch))
                raw = data["images"].cuda()
                labels = data["labels"].cuda()

                with autocast():
                    output, cf_output = model(raw, labels=labels, cf=args.ops, norm=args.norm)
                    loss_1 = cen_criterion(output, labels.to(torch.int64)) * 2
                    loss_2 = cen_criterion((output - cf_output), labels.to(torch.int64))
                    loss_1 = (loss_1 - 0.001).abs() + 0.001
                    loss_2 = (loss_2 - 0.001).abs() + 0.001
                    loss = loss_1 + loss_2
                    #loss = (loss - flooding_impactor).abs() + flooding_impactor

                loss_total.update(loss.data, raw.shape[0])
                loss_1_total.update(loss_1.data, raw.shape[0])
                loss_2_total.update(loss_2.data, raw.shape[0])

                clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                progress_bar.set_postfix(
                    loss ='%.5f' % (loss_total.avg),
                    loss_1 ='%.5f' % (loss_1_total.avg),
                    loss_2 ='%.5f' % (loss_2_total.avg),
                )

            torch.save( model.state_dict(), os.path.join(checkpoint_save_dir, '{}.pth'.format(epoch)))

            tqdm.write('Epoch: %d, Train: loss_total= %.4f,  loss_1_total= %.4f, loss_2_total= %.4f, lr_1=%.6f, lr_2=%.6f \n' % (epoch, loss_total.avg, loss_1_total.avg, loss_2_total.avg, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))# , curr_lr[0]
            log_file.write('Epoch: %d, Train: loss_total= %.4f, loss_1_total= %.4f, loss_2_total= %.4f,  lr_2=%.6f, lr_2=%.6f \n' % (epoch, loss_total.avg, loss_1_total.avg, loss_2_total.avg, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'])) #,  curr_lr[0]
            log_file.flush()

        print ('------------ test 1 -------------------')
        AUC_value, HTER_value = test_model(model, test_loader)

        lr_scheduler.step()
        #lr_scheduler.step(hter)
        write_txt = 'Test: AUC=%.4f, HTER= %.4f \n' % (AUC_value, HTER_value)
        tqdm.write(write_txt)
        log_file.write(write_txt)
        log_file.flush()

def test_model(model, data_loader, video_format=True):
    model.eval()

    raw_test_scores, gt_labels = [], []
    raw_scores_dict = []
    raw_test_video_ids = []
    with torch.no_grad():
        # for train
        for i, data in enumerate(tqdm(data_loader)):
            raw, labels, img_pathes = data["images"].cuda(), data["labels"], data["img_path"]
            output = model(raw, cf=None)

            raw_scores = output.softmax(dim=1)[:, 1].cpu().data.numpy()
            #raw_scores = 1 - raw_scores
            raw_test_scores.extend(raw_scores)
            gt_labels.extend(labels.data.numpy())

            for j in range(raw.shape[0]):
                image_name = os.path.splitext(os.path.basename(img_pathes[j]))[0]
                video_id = os.path.join(os.path.dirname(img_pathes[j]), image_name.rsplit('_', 1)[0])
                raw_test_video_ids.append(video_id)

        if video_format:
            raw_test_scores, gt_labels, _ = compute_video_score(raw_test_video_ids, raw_test_scores, gt_labels)
        raw_test_stats = [np.mean(raw_test_scores), np.std(raw_test_scores)]
        raw_test_scores = ( raw_test_scores - raw_test_stats[0]) / raw_test_stats[1]

        AUC_values, _, _, HTER_values = performances_cross_db(raw_test_scores, gt_labels)

    return AUC_values, HTER_values


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    #torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == "__main__":

    torch.cuda.empty_cache()
    #cudnn.benchmark = True
    set_seed(seed=777)

    parser = argparse.ArgumentParser(description='CF-PAD')
    parser.add_argument("--prefix", default='CFPAD', type=str, help="description")
    parser.add_argument("--model_name", default='resnet50d', type=str, help="model backbone")
    parser.add_argument("--training_csv", type=str, help="csv contains training data")
    parser.add_argument("--test_csv", type=str, help="csv contains test data")

    parser.add_argument('--lr', type=list, help='Learning rate', default=[0.001, 0.01])
    parser.add_argument("--input_shape", default=(224, 224), type=tuple, help="Neural Network input shape")
    parser.add_argument("--max_epoch", default=50, type=int, help="maximum epochs")
    parser.add_argument("--batch_size", default=128, type=int, help="train batch size")
    parser.add_argument("--pretrain", default=True, type=lambda x: (str(x).lower() in ['true','1', 'yes']))
    parser.add_argument("--num_classes", default=2, type=int, help="number of classes (bona fide and attack)")

    parser.add_argument("--ops", default=['cs','dropout','replace'], type=str, nargs='*', help="operations for causality")
    parser.add_argument("--norm", default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']))
    parser.add_argument("--prob", default=0.2, type=float, help="probabilities of CF")

    args = parser.parse_args()

    #training_csv = '/data/mfang/FacePAD_DB/WholeFrames/CropFaceFrames/Protocols/casia.csv'
    #test_csv = '/data/mfang/FacePAD_DB/WholeFrames/CropFaceFrames/Protocols/replayattack.csv'

    logging_filename = os.path.join('logs/',  '{}.txt'.format(args.prefix))
    if not os.path.isdir('logs/'):
        os.makedirs('logs/')

    log_file = open(logging_filename, 'a')
    log_file.write(f'train data: {args.training_csv} \n test data: {args.test_csv} \n  causality ops: {args.ops}, prob: {args.prob}, norm feature: {args.norm} \n  model_name: {args.model_name}, {args.pretrain}, lr: {args.lr}, prefix: {args.prefix}, bs: {args.batch_size} \n')
    log_file.write(f"-------------------------------------------- \n")
    log_file.flush()

    run_training(train_csv=args.training_csv,
                 test_csv=args.test_csv,
                 log_file=log_file,
                 output_path='prediction_scores',
                 args=args)
