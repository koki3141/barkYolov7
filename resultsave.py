from mymake.valid import valid
import numpy as np
from pathlib import Path


class Setting:
    def __init__(self):
        self.train_project="20221011" # weight 20221011 8ahkmnuyy 22dip8barks
        self.valid_project="20230705"
        self.numbers=5
        self.epoch_min=500
        self.epoch_step=100
        self.epoch_max=500
        class SaveFIle:
            def __init__(self):
                self.accuracy_file="accuracy.txt"
                self.confusion_matrix_file="normalize_result_matrix.csv"
        self.save_file=SaveFIle()

    
    def valid_saved(self,number_count,epoch_count):
        self.train_number_count="0"#str(number_count)
        self.train_epoch_count=str(epoch_count)
        self.valid_number_count=str(number_count)
        train_path=Path('runs')/'train'/self.train_project/self.train_number_count/'weights'
        self.weights = str(train_path / f"epoch_{setting.train_epoch_count}.pt")
        self.train_data= str(f"data/processed/{self.train_project}/{self.train_number_count}/dataset.yaml")
        self.data = str(f"data/processed/{self.valid_project}/{self.valid_number_count}/dataset.yaml")
        self.task='val'#helpp='train, val, test, speed or study')
        self.device="0"#helpp='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        self.project='runs/valid'#helpp='save to project/name')
        self.name = str(f"{self.valid_project}/{self.valid_number_count}/{self.train_project}/{self.train_number_count}")
        self.save_file=self.save_file
        

class MeanValid:
    def __init__(self,Setting):
        self.setting=Setting
            
    def mean_accuracy(self):
        mean_data=MeanValid.mean_data(self.save_file.accuracy_file)
        
        return mean_data
            
    def mean_confusion_matrix(self):
        mean_data=MeanValid.mean_data(self.save_file.confusion_matrix_file)
        
        return mean_data
    
    def mean_data(self,file_name):
        
        def file_load(self,file_name):
            load_path=f"{self.project}/{self.name}/{file_name}"
            data=np.load(load_path)
            return data
        
        data_list=[]    
        
        for number_count in range(setting.numbers):
            setting.valid_saved(number_count,0)
            data=file_load(self,file_name)
            data_list+=[data]
        mean_data=sum(data_list)/len(data_list)
        
        return mean_data
        

        
        
        
        

setting=Setting()
mean_valid=MeanValid(setting)
mean_valid.mean_accuracy()

    