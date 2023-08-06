
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import numpy as np
import os
import cv2
import pathlib
import torch
from glob import glob
import statistics
from pathlib import Path

import sys
sys.path.append("yolov7/")

import sys
yolov7_path = os.path.join(os.path.dirname(__file__), "yolov7/")
sys.path.append(yolov7_path)

from utils.general import xyxy2xywh


def accuracy_plot_save(accuracy_list,epochs,epoch_accuracy_best,accuracy_best,save_path):

    fig,ax1 = plt.subplots()
    ax1.plot(range(1,epochs+1),accuracy_list,color="blue",label="accuracy")
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy')
    ax1.scatter(epoch_accuracy_best, accuracy_best,color="darkblue", marker="o") # この行は前回の回答と同じ
    ax1.annotate('best({},{:.2f})'.format(str(epoch_accuracy_best),accuracy_best), xy=(epoch_accuracy_best,accuracy_best*0.999))# この行からテキストと矢印を追加

    plt.legend(loc='upper left')
    plt.savefig(str(Path(save_path)/'accuracy.png'), dpi=150, bbox_inches='tight')


def mkdir(path):
    try:
        os.makedirs(path,exist_ok=False)
    except FileExistsError:
        shutil.rmtree(path)
        os.makedirs(path)
def reinit(train_nc,valid_nc):
    result_matrix=[np.zeros((valid_nc,train_nc+2)),np.zeros((valid_nc,train_nc)),np.zeros((train_nc)),0]
    iou_conf_scatter=[np.array([[],[]],dtype=float),np.array([[],[]],dtype=float),np.array([[],[]],dtype=float)]
    back_img=[];nmatcls_img=[];subt_img=[]
    
    return result_matrix,back_img,nmatcls_img,subt_img,iou_conf_scatter

def restore(txtpath,txtdata,imgpath,predn,conf_result,iou_result,result_matrix,back_img,nmatcls_img,subt_img,iou_conf_scatter,train_names,valid_names):
    
    def calc_ious(box1, box2):
        """
        Calculate IoU of two rectangles.
        Args:
            box1 (numpy array): [x,y,w,h]
            box2 (numpy array): [x,y,w,h]
        Returns:
            float: IoU score
        """
        # convert to float
        box1 = box1.astype(float)
        box2 = box2.astype(float)

        # get coordinates of intersection rectangle
        x1 = max(box1[0] - box1[2] / 2, box2[0] - box2[2] / 2)
        y1 = max(box1[1] - box1[3] / 2, box2[1] - box2[3] / 2)
        x2 = min(box1[0] + box1[2] / 2, box2[0] + box2[2] / 2)
        y2 = min(box1[1] + box1[3] / 2, box2[1] + box2[3] / 2)

        # get area of intersection rectangle
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        # get area of union rectangle
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        union_area = box1_area + box2_area - inter_area

        # calculate IoU score
        iou_score = inter_area / union_area

        return iou_score

    def resize(label,size):
        label=np.array(label,dtype='float')
        x=size[1]
        y=size[0]

        return  np.array([label[0]*x,label[1]*y,label[2]*x,label[3]*y])
 
    imgsize=cv2.imread(str(imgpath))
    
    txtdata=np.loadtxt(txtpath)
    corr_cls=txtdata[0].astype(int);corr_xywh=txtdata[1:].astype(float)
        
    # if txtdata.ndim==1:
    #     corr_cls=txtdata[0].astype(int);corr_xywh=txtdata[1:].astype(float)
    # else:
    #     corr_cls=txtdata[:,0].astype(int);corr_xywh=txtdata[:,1:].astype(float)
        
    result_matrix[2][corr_cls]+=1
    #store max conf   
    if predn.nelement()==0:
        # not predn
        back_img+=[str(imgpath)]
        result_matrix[0][corr_cls][-1]=+1
    
    else:
        
        index=torch.argmax(predn[:,4])
        *predn_xyxy,predn_conf,predn_cls=predn[index,:].tolist()
        
        predn_xywh=(xyxy2xywh(torch.tensor(predn_xyxy).view(1, 4))).view(-1).tolist() 
        iou=calc_ious(resize(corr_xywh,imgsize.shape),np.array(predn_xywh))

        predn_cls=int(predn_cls)
        if predn_conf>conf_result and iou_result<iou:
            if train_names[predn_cls]==valid_names[corr_cls]:
                result_matrix[0][corr_cls][predn_cls]+=1
                result_matrix[3]+=1
                iou_conf_scatter[0]=np.concatenate([iou_conf_scatter[0], np.array([[iou],[predn_conf]],dtype=float)], 1)
            else:
                result_matrix[0][corr_cls][predn_cls]+=1
                nmatcls_img+=[str(imgpath)]
                iou_conf_scatter[1]=np.concatenate([iou_conf_scatter[1], np.array([[iou],[predn_conf]],dtype=float)], 1)
    
        else:
            # subthreshold
            result_matrix[0][corr_cls][-2]+=1
            result_matrix[1][corr_cls][predn_cls]+=1
            subt_img+=[str(imgpath)]
            iou_conf_scatter[2]=np.concatenate([iou_conf_scatter[2], np.array([[iou],[predn_conf]],dtype=float)], 1)

    return result_matrix,back_img,nmatcls_img,subt_img,iou_conf_scatter
    
        
        

def resave(train_names,valid_names, result_matrix,save_dir,back_img,nmatcls_img,subt_img,save_inc_img,iou_conf_scatter,save_result):
    
        
    
    
  
    # mkdir new folder
    def to_pathlib(path):
        if isinstance(path, str):
            return pathlib.Path(path)
        elif isinstance(path, pathlib.Path):
            return path
        else:
            raise TypeError(f"Unsupported type: {type(path)}")
    def mkdir(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            shutil.rmtree(path)
            os.makedirs(path)    
    
    def confusion_matrix(result_matrix=None,deresult_matrix=None,
                         xcls_list=None,ycls_list=None,xlabel='predict',ylabel='Ture',save_dir=None,file_name='result_matrix'):
        
        normalize_result_matrix=result_matrix/deresult_matrix[:,np.newaxis]

       
        mask = result_matrix == 0
        sn.set(font_scale=1.0)
        annot_kws={"size": 12}
        
        
        fig = plt.figure(figsize=(12, 9), tight_layout=True)
        
        sn.heatmap(normalize_result_matrix, mask=mask,annot=True, annot_kws=annot_kws, cmap='Blues', fmt='.2f', square=True,vmax=1, vmin=0,
                xticklabels=xcls_list,
                yticklabels=ycls_list).set_facecolor((1, 1, 1))
        fig.axes[0].set_xlabel(xlabel)
        fig.axes[0].set_ylabel(ylabel)
        plt.savefig(save_dir/('normalize_'+file_name+'.jpg'))
        np.savetxt(save_dir/("normalize_"+file_name+".csv"), result_matrix, delimiter=",")
    if save_result==True:
        int_nc=len(train_names)   
        save_dir=to_pathlib(save_dir)
        # folder reset
        mkdir(save_dir)
                    
        confusion_matrix(result_matrix=result_matrix[0],deresult_matrix=result_matrix[2],
                            xcls_list=train_names+['background','nothing'],ycls_list=valid_names,xlabel='predict',ylabel='Ture',save_dir=save_dir,file_name='devresult_matrix')
        
        confusion_matrix(result_matrix=np.delete(result_matrix[0],result_matrix[0].shape[1]-1,1),deresult_matrix=result_matrix[2],
                            xcls_list=train_names+['background'],ycls_list=valid_names,xlabel='predict',ylabel='Ture',save_dir=save_dir,file_name='result_matrix')
        
        confusion_matrix(result_matrix=result_matrix[1],deresult_matrix=result_matrix[2],
                            xcls_list=train_names,ycls_list=valid_names,xlabel='background',ylabel='Ture',save_dir=save_dir,file_name='subthreshold_result_matrix')
        plt.rcParams['font.family'] ='sans-serif'#使用するフォント
        fig = plt.figure(figsize=(8,6), tight_layout=True)
        plt.xlim(0, 1) # x軸の表示範囲
        plt.ylim(0, 1) # y軸の表示範囲

        plt.title('IoU_Confidence_Scatter',
                            fontsize=10) # タイトル
        plt.xlabel("IoU", fontsize=10) # x軸ラベル
        plt.ylabel("Confidence", fontsize=10) # y軸ラベル
        plt.grid(True) # 目盛線の表示
        plt.tick_params(labelsize = 12) # 目盛線のラベルサイズ

        # グラフの描画
        plt.scatter(iou_conf_scatter[0][0], iou_conf_scatter[0][1], s=10, c="blue",
                            marker="o", alpha=0.3, label="correct") #(5)散布図の描画
        plt.scatter(iou_conf_scatter[1][0], iou_conf_scatter[1][1], s=10, c="red",
                            marker="o", alpha=0.3, label="incorrect") #(6)散布図の描画
        plt.scatter(iou_conf_scatter[2][0], iou_conf_scatter[2][1], s=10, c="gray",
                            marker="o", alpha=0.3, label="background") #(6)散布図の描画
        plt.legend(bbox_to_anchor=(1.05, 1),loc='upper left', fontsize=10) # (7)凡例表示
        plt.savefig(save_dir/('IoU_Confidence_Scatter.jpg'))
    
    
    if save_inc_img==True:
        def inc_save(inc_path,atpath,save_dir):
            for i in inc_path:
                im = cv2.imread(i)
                cv2.imwrite(str(save_dir/'Not_predict'/atpath/i.split('/')[-1]),im)
                # txtpath=str(i).replace('/images','/labels').replace('.jpg','.txt')
                # with open(txtpath,'r') as f:
                #     folder_name=train_names[int(f.read().split()[0])]
                # cv2.imwrite(str(save_dir/'Not_predict'/folder_name/i.split('/')[-1]),im)
                # Not able pre images save folder
                
        for i in ['all','subt','back','nmatcls']:
            mkdir(save_dir/'Not_predict'/i)
            
        inc_save(back_img,"back",save_dir)
        inc_save(subt_img,"subt",save_dir)
        inc_save(nmatcls_img,"nmatcls",save_dir)
        inc_save(back_img+subt_img+nmatcls_img,'all',save_dir)
        
    return result_matrix[3]/np.sum(result_matrix[0])
    