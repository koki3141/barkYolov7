
import argparse
import json
import os
from pathlib import Path
from threading import Thread
from os import path as osp
import pandas as pd
import numpy as np
import torch
import yaml
from glob import glob
from tqdm import tqdm
from mymake.mymake import resave,mkdir,restore,reinit

# relative
import sys
# sys.path.append("yolov7/")

# absolute
yolov7_path = os.path.join(os.path.dirname(__file__), "yolov7/")
sys.path.append(yolov7_path)

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized, TracedModel
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd


import pdb

def test(opt):
    wandb_logger=opt.wandb_logger
    compute_loss=opt.compute_loss
    print(opt.train_data)
    print(opt.valid_data)
    
    # Initialize/load model and set device
    training = opt.model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=opt.batch_size)

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
#        mkdir(save_dir)

        # Load model
        model = attempt_load(opt.weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(opt.imgsz, s=gs)  # check img_size
        
        if opt.trace:
            model = TracedModel(model, device, imgsz)

    # Half
    half = device.type != 'cpu' and opt.half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    if isinstance(opt.valid_data, str):

        with open(opt.valid_data) as f:
            valid_data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(valid_data)  # check
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(valid_data[task], imgsz, opt.batch_size, gs, opt, pad=0.5, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]
        

    
    seen = 0
    valid_nc = 1 if opt.single_cls else int(valid_data['nc'])  # number of classes
    valid_names=valid_data['names']
    if opt.train_data!=None:
        with open(opt.train_data) as f:
            train_data = yaml.load(f, Loader=yaml.SafeLoader)
    else:
        train_data=valid_data    
    
    train_nc=train_data['nc']
    train_names=train_data['names']
        
  
    loss = torch.zeros(3, device=device)


    #my add
    result_matrix,back_img,nmatcls_img,subt_img,iou_conf_scatter=reinit(int(train_nc),int(valid_nc))

    for batch_i,(img, targets, paths, shapes) in enumerate(tqdm(dataloader)): #,desc=s)):

        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            # Run model
            t = time_synchronized()
            out, train_out = model(img, augment=opt.augment)  # inference and training outputs
       

            # Compute loss
            if opt.compute_loss:
                loss += opt.compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if opt.save_hybrid else []  # for autolabelling
            t = time_synchronized()
            out = non_max_suppression(out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, labels=lb, multi_label=True)


        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            imgpath = Path(paths[si])
              
            seen += 1

            # if len(pred) == 0:
            #     if nl:
            #         stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            #     continue

            # Predictions
            predn = pred.clone()

            predn[:,:4]=scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            #gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh   
            txtpath=str(imgpath).replace('/images/','/labels/').replace('.jpg','.txt')
            
            result_matrix,back_img,nmatcls_img,subt_img,iou_conf_scatter=restore(txtpath,-1,imgpath,predn,opt.conf_result,opt.iou_result,result_matrix,back_img,nmatcls_img,subt_img,iou_conf_scatter,train_names,valid_names)
 
    return resave(train_names,valid_names, result_matrix,save_dir,back_img,nmatcls_img,subt_img,opt.save_inc_img,iou_conf_scatter,opt.save_result)


def valid(train_opt,save_path,train_data=None,save_result=None):
    class MyClass:
        def __init__(self,train_opt,save_path,train_data,save_result):
            self.weights=save_path#helpp='model.pt path(s)')
            self.valid_data=train_opt.data #helpp='*.data path')
            self.train_data=train_data
            self.batch_size=16#helpp='size of each image batch')
            self.imgsz=train_opt.img_size[0]#helpp='inference size (pixels)')
            self.conf_thres=0.5#helpp='object confidence threshold')
            self.iou_thres=0.5#helpp='IOU threshold for NMS')
            self.conf_result=0.5
            self.iou_result=0.5
            save_inc_img=False
            self.task='val'#helpp='train, val, test, speed or study')
            self.device=train_opt.device#helpp='cuda device, i.e. 0 or 0,1,2,3 or cpu')
            self.single_cls=None#helpp='treat as single-class dataset')
            self.augment=None#helpp='augmented inference')
            self.verbose=None#helpp='report mAP by class')
            self.save_txt=False #helpp='save results to *.txt')
            self.save_hybrid=False #helpp='save label+prediction hybrid results to *.txt')
            self.save_conf=None#helpp='save confidences in --save-txt labels')
            self.save_json=None#helpp='save a cocoapi-compatible JSON results file')
            self.project='runs/valid/'#helpp='save to project/name')
            self.name=train_opt.name #helpp='save to project/name')
            self.exist_ok=True#helpp='existing project/name ok, do not increment')
            self.no_trace=None#helpp='don`t trace model')
            self.v5_metric=None#helpp='assume maximum recall as 1.0 in AP calculation')
            self.trace=None
            self.plots=True
            self.model=None
            self.half_precision=True
            self.wandb_logger=None
            self.compute_loss=None
            self.save_inc_img=False
            self.save_result=save_result

    opt=MyClass(train_opt,save_path,train_data,save_result)        
    print(opt.weights)
            
    #check_requirements()

    # if opt.task in ('train', 'val', 'test'):  # run normally
    result=test(opt)

    return result
    

                # elif opt.task == 'speed':  # speed benchmarks
                #     for w in opt.weights:
                #         test(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, save_json=False, plots=False, v5_metric=opt.v5_metric)

                # elif opt.task == 'study':  # run over a range of settings and save/plot
                #     # python test.py --task study --data coco.yaml --iou 0.65 --weights yolov7.pt
                #     x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
                #     for w in opt.weights:
                #         f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
                #         y = []  # y axis
                #         for i in x:  # img-size
                #             print(f'\nRunning {f} point {i}...')
                #             r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                #                         plots=False, v5_metric=opt.v5_metric)
                #             y.append(r + t)  # results and times
                #         np.savetxt(f, y, fmt='%10.4g')  # save
                #     os.system('zip -r study.zip study_*.txt')
                #     plot_study_txt(x=x)  # plot
                    
            
                    
        
        
                
                
            
            
            
            
