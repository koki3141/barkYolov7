from mymake.valid import valid
import numpy as np
from pathlib import Path
from mymake.mymake import mkdir
import seaborn as sn
import matplotlib.pyplot as plt
import yaml
import pandas as pd

class Setting:
    def __init__(self):
        self.train_project="22dip8barks" # weight 20221011 8ahkmnuyy 22dip8barks
        self.valid_project="20230705"
        self.numbers=5
        self.epoch_min=500
        self.epoch_step=100
        self.epoch_max=500
        self.train_number_count="0"#str(number_count)
        self.mean_save_path=f"runs/valid/{self.valid_project}/mean/{self.train_project}/{self.train_number_count}"
        
        class SaveFIle:
            def __init__(self):
                self.accuracy_file="accuracy.txt"
                self.confusion_matrix_file="normalize_result_matrix.csv"
        self.save_file=SaveFIle()

    
    def valid_saved(self,number_count,epoch_count):
        self.train_epoch_count=str(epoch_count)
        self.valid_number_count=str(number_count)
        self.train_data= str(f"data/processed/{self.train_project}/{self.train_number_count}/dataset.yaml")
        self.data = str(f"data/processed/{self.valid_project}/{self.valid_number_count}/dataset.yaml")
        self.valid_data = str(f"runs/valid/{self.valid_project}/{self.valid_number_count}/{self.train_project}/{self.train_number_count}")
        self.save_file=self.save_file
        

class MeanValid:
    def __init__(self,Setting):
        self.setting=Setting
        mkdir(self.setting.mean_save_path)
            
    def mean_accuracy(self):
        data_group=MeanValid.process_data(self,self.setting.save_file.accuracy_file)
        index_names=list(range(Setting.epoch_min,(Setting.epoch_max+Setting.epoch_step),Setting.epoch_step))
        columns_names=list(range(0,Setting.numbers))
        df = pd.DataFrame(data_group.data_group_array, index = index_names,columns = columns_names)
        df.to_csv(f"{self.setting.mean_save_path}/accuracy.csv")
        return mean_data
            
    def mean_confusion_matrix(self):
        data_group=MeanValid.process_data(self,self.setting.save_file.confusion_matrix_file)
        mean_data=data_group.data_mean
        
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
        
        sn.heatmap(mean_data, mask=mask,annot=True, annot_kws=annot_kws, cmap='Blues', fmt='.2f', square=True,vmax=1, vmin=0,
                xticklabels=xcls_list,
                yticklabels=ycls_list).set_facecolor((1, 1, 1))
        fig.axes[0].set_xlabel(xlabel)
        fig.axes[0].set_ylabel(ylabel)
        plt.savefig(f'{self.setting.mean_save_path}/normalize_result_matrix.jpg',dpi=150)
        np.savetxt(f"{self.setting.mean_save_path}/normalize_result_matrix.csv", mean_data, delimiter=",")
        
        return mean_data
    
    def process_data(self,file_name):
        
        def file_load(self,file_name):
            load_path=f"{self.setting.valid_data}/{file_name}"
            data=np.loadtxt(load_path,delimiter=",")
            return data
        
        data_list=[]    
        
        for number_count in range(setting.numbers):
            setting.valid_saved(number_count,0)
            data=file_load(self,file_name)
            data_list+=[data]
        data_array=np.array([data_list])
        class DataGroup:
            def __init__(self,data_array):
                self.data_array=data_array
                self.data_mean=np.mean(data_array)
                self.data_std=np.std(data_array)
                self.data_group_array=np.hstack((DataGroup.data_array,np.array([DataGroup.data_mean]),np.array([DataGroup.data_std])))
        data_group=DataGroup(data_array)
        return data_group
        

        
        
        
        

setting=Setting()
mean_valid=MeanValid(setting)


mean_valid.mean_accuracy()
mean_valid.mean_confusion_matrix()


    