from glob import glob
import cv2
import os
import shutil
import random
import cv2
import csv
from pathlib import Path
import pandas as pd
import itertools
import numpy as np
import yaml
from mymake.datasetformat import mkdir,yolov7Dataset,yolov7Yaml

from os import path as osp

class Data:
    def __init__(self,folderName,depths,numOfDatasets):
        
        self.originalPath=Path("data")/"original"/folderName
        self.outputPath=Path("data")/"processed"/folderName
        
        folderDepths=len(self.originalPath.parts)+depths
        self.classPaths=[i for i in list(self.originalPath.glob('**')) if len(i.parts)==folderDepths]
        self.classNames=[i.name for i in self.classPaths]
        self.classCounts=Data.check(self)
        self.minCount=min(self.classCounts)
        
        
        self.minClassName=self.classNames[self.classCounts.index(min(self.classCounts))]
        self.numOfDatasets=numOfDatasets
        
        
        Data.checkExist(self.originalPath)
        Data.dataAnalysis(self)
        
        # self.minCount=131
        
    def checkExist(self):  
        assert self.exists(), f'"{str(self)}" does not exist'
    

        
    def check(self):
        imagefileCounts=[len(list(path.glob('**/*.jpg'))) for path in self.classPaths]
        textfileCounts=[len(list(path.glob('**/*.txt'))) for path in self.classPaths]
        if imagefileCounts == textfileCounts:
            return textfileCounts
        
        print(imagefileCounts)
        print(textfileCounts)
        assert False,'imageFile and textFile count do not match'    
    
    def dataAnalysis(self):
        display(pd.DataFrame(self.classCounts, index=self.classNames,columns=['number of sheets']))
    
    def divTrainValid(self):
        div0=int(self.minCount*2/3)
        div1=int(self.minCount)
        
        mkdir(self.outputPath)
        for num in range(self.numOfDatasets):
            def mkdirFolder(outputPathNum):
                definePath=[Path('images')/'train',Path('labels')/'train',Path('images')/'valid',Path('labels')/'valid']
                savePath = [str(outputPathNum/path) for path in definePath]
                np.vectorize(mkdir)(savePath)
                return savePath
            
            outputPathNum=self.outputPath/str(num)
            
            mkdir(outputPathNum)
            yolov7Yaml(outputPathNum,self.classNames)
            savePath=mkdirFolder(outputPathNum)
            for classId,classPath in enumerate(self.classPaths):

                jpg_files=np.array(sorted(classPath.glob("**/*.jpg")),dtype=str)
                txt_files=np.array(sorted(classPath.glob("**/*.txt")),dtype=str)
                
                random_index=random.sample(range(len(jpg_files)),k=self.minCount)
                train_index=random_index[:div0]
                valid_index=random_index[div0:div1]
                
                train_jpg_files=jpg_files[train_index]
                valid_jpg_files=jpg_files[valid_index]
                train_txt_files=txt_files[train_index]
                valid_txt_files=txt_files[valid_index]
                
                yolov7Dataset(train_jpg_files,train_txt_files,valid_jpg_files,valid_txt_files,savePath,classId)
    def mkCommand(self):
        epochs=500
        batch_size=16
        img_size=320
        save_period=100
        weight="yolov7/weights/yolov7_training.pt"
        
        command="python train.py --device 0 --epochs {0} --batch-size {1} --img {2} --data {3}/dataset.yaml --weights {4} --name {3} --save_period {5};"

        
        for num in range(self.numOfDatasets):
            print(command.format(epochs,batch_size,img_size,str(Path(self.outputPath.name)/str(num)),weight,str(save_period)))


original_path=Path("data")/"original"
datasets=list(original_path.glob("*"))

for dataset in datasets:
    print(dataset.name)
    

folderName="8ahkmnuyy"


folderDepth=1
numOfDatasets=5

subData=Data(folderName,folderDepth,numOfDatasets)


subData.mkCommand()
