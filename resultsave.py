import numpy as np
from pathlib import Path
from dataclasses import dataclass
import seaborn as sn
import matplotlib.pyplot as plt
import yaml
import pandas as pd
import sys

sys.path.append("mymake/")
from valid import valid
from mymake import mkdir



class Setting:
    def __init__(self):
        "22dip8barks"
        self.train_project="8ahkmnuyy" # weight 20221011 8ahkmnuyy 22dip8barks
        self.train_numbers=1
        self.valid_project="3ahy20221011"# 20230705
        self.valid_numbers=5
        self.epoch_min=500
        self.epoch_step=100
        self.epoch_max=500
        self.accuracy_file="accuracy.csv"
        self.confusion_matrix_file="normalize_result_matrix.csv"
        self.run_mean=True
        self.run_valid=False
        self.accuracy_graph=True
        
    
    def data_valid(self,valid_number_count,epoch_count):
        self.valid_number_count=str(valid_number_count)
        self.epoch_count=str(epoch_count)
        self.mean_save_path=f"runs/valid/{self.valid_project}/{self.valid_number_count}/{self.train_project}/mean"
        self.epoch_mean_save_path=f"{self.mean_save_path}/{self.epoch_count}"
        
    
    def data_path(self,number_count):
        self.train_number_count=str(number_count)
        self.train_data= f"data/processed/{self.train_project}/{self.train_number_count}/dataset.yaml"
        self.data = f"data/processed/{self.valid_project}/{self.valid_number_count}/dataset.yaml"
        self.valid_data = f"runs/valid/{self.valid_project}/{self.valid_number_count}/{self.train_project}/{self.train_number_count}/{self.epoch_count}"
        
        
    def evaluate(self):
        train_path=Path('runs')/'train'/self.train_project/self.train_number_count/'weights'
        # self.weights=str(list(weight_path.glob(str(savetype)+"*"))[0])
        self.weights = str(train_path / f"epoch_{self.epoch_count}.pt")
        # self.train_data= str(f"data/processed/{setting.train_project}/{setting.train_number_count}/dataset.yaml")
        # self.data = str(f"data/processed/{setting.valid_project}/{setting.valid_number_count}/dataset.yaml")
        self.project='runs/valid/'#helpp='save to project/name')
        self.name = str(f"{self.valid_project}/{self.valid_number_count}/{self.train_project}/{self.train_number_count}/{self.epoch_count}")
        self.device="0"#helpp='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        self.batch_size=16#helpp='size of each image batch')
        self.img_size=[320]#helpp='inference size (pixels)')
        self.conf_thres=0.5#helpp='object confidence threshold')
        self.iou_thres=0.45#helpp='IOU threshold for NMS')
        self.conf_result=0.5
        self.iou_result=0.5
        self.save_inc_img=False
        self.task='val'#helpp='train, val, test, speed or study')
        self.single_cls=None#helpp='treat as single-class dataset')
        self.augment=None#helpp='augmented inference')
        self.verbose=None#helpp='report mAP by class')
        self.save_txt=False #helpp='save results to *.txt')
        self.save_hybrid=False #helpp='save label+prediction hybrid results to *.txt')
        self.save_conf=None#helpp='save confidences in --save-txt labels')
        self.save_json=None#helpp='save a cocoapi-compatible JSON results file')
        self.exist_ok=True#helpp='existing project/name ok, do not increment')
        self.no_trace=None#helpp='don`t trace model')
        self.v5_metric=None#helpp='assume maximum recall as 1.0 in AP calculation')
        self.trace=None
        self.plots=True
        self.model=None
        self.half_precision=True
        self.wandb_logger=None
        self.compute_loss=None
        self.save_result=True


        
        

class MeanValid:
    def __init__(self,Setting):
        self.setting=Setting
        mkdir(self.setting.epoch_mean_save_path)
        print(self.setting.epoch_mean_save_path)
        df = pd.DataFrame([list(["epoch"]+list(range(0,self.setting.train_numbers))+['mean',"std"])])
        df.to_csv(f"{self.setting.epoch_mean_save_path}/{self.setting.accuracy_file}",index=False, header=False)
            
    def mean_accuracy(self):
        data_group=MeanValid.process_data(self,self.setting.accuracy_file)

        df = pd.DataFrame([[self.setting.epoch_count]+data_group.data_group_array.tolist()])
        print(df)
        df.to_csv(f"{self.setting.epoch_mean_save_path}/{self.setting.accuracy_file}", mode='a', index=False,header=False)

            
    def mean_confusion_matrix(self):
        data_group=MeanValid.process_data(self,self.setting.confusion_matrix_file)
        
        mean_data=data_group.data_mean
        print(mean_data)
        
        with open(self.setting.train_data) as f:
            xcls_list = yaml.load(f, Loader=yaml.SafeLoader)['names']+['background']
        
        with open(self.setting.data) as f:
            ycls_list = yaml.load(f, Loader=yaml.SafeLoader)['names']
        
        
        xlabel='predict'
        ylabel='Ture'
        mask = mean_data == 0
        sn.set(font_scale=1.0)
        annot_kws={"size": 12}
        
        
        fig = plt.figure(figsize=(12, 9), tight_layout=True)
        
        # sn.heatmap(mean_data, mask=mask,annot=True, annot_kws=annot_kws, cmap='Blues', fmt='.2f', square=True,vmax=1, vmin=0,
        #         xticklabels=xcls_list,
        #         yticklabels=ycls_list).set_facecolor((1, 1, 1))
        sn.heatmap(mean_data, mask=mask,annot=True, annot_kws=annot_kws, cmap='Blues', fmt='.2f', linewidths=1,vmax=1, vmin=0,
        xticklabels=xcls_list,
        yticklabels=ycls_list).set_facecolor((1, 1, 1))
        fig.axes[0].set_xlabel(xlabel)
        fig.axes[0].set_ylabel(ylabel)
        plt.savefig(f'{self.setting.epoch_mean_save_path}/normalize_result_matrix.jpg',dpi=150)
        np.savetxt(f"{self.setting.epoch_mean_save_path}/{self.setting.confusion_matrix_file}", mean_data, delimiter=",")
        
        return mean_data
    
    def process_data(self,file_name):
        
        def file_load(self,file_name):
            load_path=f"{self.setting.valid_data}/{file_name}"
            data=np.loadtxt(load_path,delimiter=",")
            return data
        
        data_list=[]    
        
        for number_count in range(setting.train_numbers):
            setting.data_path(number_count)
            data=file_load(self,file_name)
            data_list+=[data]
        data_array=np.array(data_list)
        class DataGroup:
            def __init__(self,data_array):
                self.data_array=data_array
                self.data_mean=np.mean(data_array,axis=0)
                self.data_std=np.std(data_array)
                if data_array.ndim==1:
                    self.data_mean=np.mean(data_array)
                    self.data_group_array=np.hstack((data_array,np.array([self.data_mean]),np.array([self.data_std])))
        data_group=DataGroup(data_array)
        return data_group
    
    def run_evaluate(self):
        for number_count in range(setting.train_numbers):
            setting.data_path(number_count)
            self.setting.evaluate()
            opt=self.setting
            result=valid(opt,opt.weights,opt.train_data,opt.save_result)
            np.savetxt(str(Path(opt.project)/opt.name/self.setting.accuracy_file),np.array([result]),delimiter=",")
        

        
        
setting=Setting()       
        
if setting.run_mean==True:
    for valid_number_count in range(setting.valid_numbers):
        for epoch_count in range(setting.epoch_min,setting.epoch_max+setting.epoch_step,setting.epoch_step):
            setting.data_valid(valid_number_count,epoch_count)
            mean_valid=MeanValid(setting)
            if setting.run_valid==True:
                mean_valid.run_evaluate()
            mean_valid.mean_accuracy()
            mean_valid.mean_confusion_matrix()



if setting.accuracy_graph==True:
    colors = ['aqua', 'orange','red']
    epoch_list=list(range(setting.epoch_min,setting.epoch_max+setting.epoch_step,setting.epoch_step))
    def imgSet(xticks,min,max,figname):
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.xticks(xticks)
        plt.ylim(min-0.005, max+0.005)
        #plt.figtext(0.515, -0.015,figname, ha='center')
        plt.legend(loc='lower right')
        print(figname)
        plt.savefig(figname, dpi=150, bbox_inches='tight')

    def minmax(min_value,max_value,array):
        min_value=min([min_value]+array)
        max_value=max([max_value]+array)
        return min_value,max_value
    min_value=1;max_value=0

    for valid_number_count in range(setting.valid_numbers):
        epoch_mean_list=[]
        for epoch_count in range(setting.epoch_min,setting.epoch_max+setting.epoch_step,setting.epoch_step):
            setting.data_valid(valid_number_count,epoch_count)
            df=pd.read_csv(str(Path(setting.epoch_mean_save_path)/setting.accuracy_file))
            epoch_mean_list+=[df.iloc[0]['mean']]
        min_value,max_value=minmax(min_value,max_value,epoch_mean_list)
        print(max_value)
        plt.plot(epoch_list, epoch_mean_list ,color=colors[valid_number_count],label=valid_number_count)

    imgSet(epoch_list,min_value,max_value,str(Path(setting.mean_save_path)/"accuracy.jpg"))