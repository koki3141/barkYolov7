from mymake.valid import valid
import numpy as np
from pathlib import Path


class Setting:
    def __init__(self):
        self.train_project="20221011" # weight
        self.evaluate_project="20230705"
        self.numbers=3
        self.epoch_min=100
        self.epoch_step=100
        self.epoch_max=500

    def add(self,number_count,epoch_count):
        self.number_count=str(number_count)
        self.epoch_count=str(epoch_count)

setting=Setting()

for number_count in range(setting.numbers):
    for epoch_count in range(setting.epoch_min,setting.epoch_max+setting.epoch_step,setting.epoch_step):
        # train_data=None
        class ValidSetting:
            def __init__(self,setting):
                # savetype="best"
                train_path=Path('runs')/'train'/setting.train_project/setting.number_count/'weights'
                # self.weights=str(list(weight_path.glob(str(savetype)+"*"))[0])
                self.weights = str(train_path / f"epoch_{setting.epoch_count}.pt")
                
                self.train_data= str(f"data/processed/{setting.train_project}/{setting.number_count}/dataset.yaml")
                self.data = str(f"data/processed/{setting.evaluate_project}/{setting.number_count}/dataset.yaml")
                
                self.batch_size=16#helpp='size of each image batch')
                self.img_size=[320]#helpp='inference size (pixels)')
                self.conf_thres=0.5#helpp='object confidence threshold')
                self.iou_thres=0.45#helpp='IOU threshold for NMS')
                self.conf_result=0.5
                self.iou_result=0.5
                save_inc_img=False
                self.task='val'#helpp='train, val, test, speed or study')
                self.device="0"#helpp='cuda device, i.e. 0 or 0,1,2,3 or cpu')
                self.single_cls=None#helpp='treat as single-class dataset')
                self.augment=None#helpp='augmented inference')
                self.verbose=None#helpp='report mAP by class')
                self.save_txt=False #helpp='save results to *.txt')
                self.save_hybrid=False #helpp='save label+prediction hybrid results to *.txt')
                self.save_conf=None#helpp='save confidences in --save-txt labels')
                self.save_json=None#helpp='save a cocoapi-compatible JSON results file')
                self.project='runs/valid/'#helpp='save to project/name')
                if setting.train_project == setting.evaluate_project:
                    self.name = str(f"{setting.train_project}/{setting.number_count}")
                else:
                    self.name = str(f"{setting.train_project}/{setting.number_count}/{setting.evaluate_project}")
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
                self.save_result=True
                
        setting.add(number_count,epoch_count)
        opt=ValidSetting(setting)  
    
    accuracy=valid(opt,opt.weights,opt.train_data,opt.save_result)

    with open(str(Path(opt.project)/opt.name/'accuracy.txt'), 'w') as f:
        f.write(str(accuracy))

