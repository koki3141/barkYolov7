import cv2
import os
import shutil
import random
import cv2
from pathlib import Path
import numpy as np
import yaml
from os import path as osp

def txt_re(txtdata):  # sourcery skip: avoid-builtin-shadow
    opentxt=txtdata
    import numpy as np
    def str_float(x):
        return float(x)

    def float_str(x):
        return str(x)


    floatlisttxt=np.vectorize(str_float)(opentxt.split())

    id=int(floatlisttxt[0])
    xmin=floatlisttxt[1]-floatlisttxt[3]/2;ymin=floatlisttxt[2]-floatlisttxt[4]/2
    xmax=floatlisttxt[1]+floatlisttxt[3]/2;ymax=floatlisttxt[2]+floatlisttxt[4]/2

    strlisttxt=np.vectorize(float_str)([xmin,ymin,xmax,ymax,id])

    savetxt=' '.join(strlisttxt)
    return savetxt
def txt_re(txtdata,id):
    
    opentxt=txtdata

    listtxt=opentxt.split()
    listtxt[0]=str(id)

    return ' '.join(listtxt)

def txt_save(txtfile,prepath,id):  # sourcery skip: avoid-builtin-shadow
    for j, i in enumerate(txtfile):
        list=i.split('/')
        afpath = f"{list[1]}_{list[-1]}"
        path = f'{prepath}/{str(j)}{afpath}'
        with open(i,'r') as f:
            txtdata=f.read()

        txtdata=txt_re(txtdata,id)
        with open(path,mode='w',encoding='UTF-8') as f:
            f.write(txtdata)  
            
def image_save(imgfile,prepath):     # sourcery skip: avoid-builtin-shadow
    for j, i in enumerate(imgfile):
        list=i.split('/')
        afpath = f"{list[1]}_{list[-1]}"
        path = f'{prepath}/{str(j)}{afpath}'
        image=cv2.imread(i)

        cv2.imwrite(path,image)

def mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        shutil.rmtree(path)
        os.makedirs(path)
        
        

def yolov7Dataset(tjf,ttf,vjf,vtf,savePath,classId):
    
        


    image_save(tjf,savePath[0])
    txt_save(ttf,savePath [1],classId)
    image_save(vjf,savePath [2])
    txt_save(vtf,savePath [3],classId)
    
    

def yolov7Yaml(outputPath,classNames):
    
    with open(str(outputPath/('dataset.yaml')), "w") as yf:
        yaml.dump(
            {
                "train": f"{str(outputPath)}/images/train",
                "val": f"{str(outputPath)}/images/valid",
                "nc": len(classNames),
                "names": classNames,
            },
            yf,
            default_flow_style=False,
        )
    
            
        
        