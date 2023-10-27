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
from mymake.datasetformat import mkdir, yolov7Dataset, yolov7Yaml
from tqdm import tqdm
from os import path as osp


class Data:
    def __init__(self, infolderName,outfolderName,depths, numOfDatasets):

        self.originalPath = Path("data")/"original"/infolderName
        self.outputPath = Path("data")/"processed"/(outfolderName)

        self.folderDepths = len(self.originalPath.parts)+depths
        self.classPaths = sorted([i for i in list(
            self.originalPath.glob('**')) if len(i.parts) == self.folderDepths])
        
        self.foloderMostDepths=len(self.originalPath.parts)+3
        self.foloderMostDepths=sorted([i for i in list(
            self.originalPath.glob('**')) if len(i.parts) == self.foloderMostDepths])
        
        if depths==1:
            self.classNames =[i.name for i in self.classPaths]
            
        elif depths==2:
            self.classNames = ['_'.join([i.parts[self.folderDepths-2],i.name]) for i in self.classPaths]
        
        else:
            self.classNames = ['_'.join([i.parts[self.folderDepths-3],i.parts[self.folderDepths-2],i.name]) for i in self.classPaths]
        
        self.classCounts = Data.check(self)
        self.minCount = min(self.classCounts)

        self.minClassName = self.classNames[self.classCounts.index(
            min(self.classCounts))]
        self.numOfDatasets = numOfDatasets

        Data.checkExist(self.originalPath)
        Data.connectFolderName(self)
        Data.dataAnalysis(self)     

        # self.minCount=131

    def checkExist(self):
        assert self.exists(), f'"{str(self)}" does not exist'
        
    def connectFolderName(self):
        
        self.parentClassNames = ['_'.join([i.parts[self.folderDepths-3],i.parts[self.folderDepths-2],i.name]) for i in self.classPaths ]
        self.barkNames = [i.parts[self.folderDepths-3] for i in self.classPaths ]
        self.prefectreNames = [i.parts[self.folderDepths-2] for i in self.classPaths ]
        self.barkNumberNames = [i.name for i in self.classPaths ]
        
        NumberOfClass=pd.DataFrame(np.array([self.barkNames,self.prefectreNames,self.barkNumberNames,self.classCounts]).T,
                                   index=self.parentClassNames, columns=['barks','prefecture','barkNumber','number of images'])
        NumberOfClass.to_csv('NumberOfClass.csv')
    def remove(self,folder_number,if_state): 
        for i in self.foloderMostDepths:
            if if_state==True:
                if i.name==str(folder_number):
                    shutil.rmtree(i)
            else:
                if i.name!=str(folder_number):
                    shutil.rmtree(i)
    def check(self):
        imagefileCounts = [len(list(path.glob('**/*.jpg')))
                           for path in self.classPaths]
        textfileCounts = [len(list(path.glob('**/*.txt')))
                          for path in self.classPaths]
        if imagefileCounts == textfileCounts:
            return textfileCounts

        print(imagefileCounts)
        print(textfileCounts)
        assert False, 'imageFile and textFile count do not match'

    def dataAnalysis(self):
        NumberOfClass=pd.DataFrame(self.classCounts,
                index=self.classNames , columns=['number of images'])
        print(NumberOfClass)
        
        

    def divTrainValid(self):
        if 'train' in str(self.outputPath):
            div0 = int(self.minCount)
            div1 = int(self.minCount)*0
        elif 'valid' in str(self.outputPath):
            div0 = int(self.minCount)*0
            div1 = int(self.minCount)
        else:
            div0 = int(self.minCount*2/3)
            div1 = int(self.minCount)

        mkdir(self.outputPath)
        for num in tqdm(range(self.numOfDatasets)):
            def mkdirFolder(outputPathNum):
                definePath = [Path('images')/'train', Path('labels') /
                              'train', Path('images')/'valid', Path('labels')/'valid']
                savePath = [str(outputPathNum/path) for path in definePath]
                np.vectorize(mkdir)(savePath)
                return savePath

            outputPathNum = self.outputPath/str(num)

            mkdir(outputPathNum)
            yolov7Yaml(outputPathNum, self.classNames)
            savePath = mkdirFolder(outputPathNum)
            for classId, classPath in enumerate(self.classPaths):

                jpg_files = np.array(
                    sorted(classPath.glob("**/*.jpg")), dtype=str)
                txt_files = np.array(
                    sorted(classPath.glob("**/*.txt")), dtype=str)

                random_index = random.sample(
                    range(len(jpg_files)), k=self.minCount)
                train_index = random_index[:div0]
                valid_index = random_index[div0:div1]

                train_jpg_files = jpg_files[train_index]
                valid_jpg_files = jpg_files[valid_index]
                train_txt_files = txt_files[train_index]
                valid_txt_files = txt_files[valid_index]

                yolov7Dataset(train_jpg_files, train_txt_files,
                              valid_jpg_files, valid_txt_files, savePath, classId)

    def mkCommand(self):
        epochs = 500
        batch_size = 16
        img_size = 320
        save_period = 100
        weight = "yolov7/weights/yolov7_training.pt"

        command = "python train.py --device {6} --epochs {0} --batch-size {1} --img {2} --data {3}/dataset.yaml --weights {4} --name {3} --save_period {5};"

        for num in range(self.numOfDatasets):
            print(command.format(epochs, batch_size, img_size, str(
                Path(self.outputPath.name)/str(num)), weight, str(save_period),str(num)))


original_path = Path("data")/"original"
datasets = list(original_path.glob("*"))

for dataset in datasets:
    print(dataset.name)




infolderName ="8_barks"
"5_barks_3tree_fukuoka_valid"
"5_barks_1tree_fukuoka_train"
"8_bark_clean_valid"
"original_clean_20211011"
on_remove=False
div=False
mkcommand=True
outfolderName=infolderName.replace("8",'8')
print(infolderName,outfolderName)
folderDepth = 1
numOfDatasets = 5
pd.set_option("display.max_rows", None, "display.max_columns", None)

subData = Data(infolderName,outfolderName, folderDepth, numOfDatasets)

if on_remove==True:
    folder_number=1
    if_state=False
    subData.remove(folder_number,if_state)
subData = Data(infolderName,outfolderName, folderDepth, numOfDatasets)

if div==True:
    subData.divTrainValid()
print(subData.minCount)
if mkcommand==True:
    subData.mkCommand()
