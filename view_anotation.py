import os
import sys
import numpy as np
import cv2
from pathlib import Path
import random

def plot_one_box(x, img, output_path, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
 
    if output_path.exists():
        os.remove(str((output_path)))       
    cv2.imwrite(str((output_path)), img)
    
def xywh_to_xyxy(bbox,img):
    cls, x, y, w, h = bbox
    return [x-w/2, y-h/2, x + w/2, y + h/2]

data_path=Path("data/original/5_barks_1tree_fukuoka/honnsugi/fukuoka/0/")


img_path_list=list(sorted(data_path.glob("*.jpg")))
txt_path_list=list(sorted(data_path.glob("*.txt")))

for img, txt in zip(img_path_list,txt_path_list):
    output_path=Path("img_and_txt")/"data"
    output_path=output_path/str(img.name)
    
    im0=cv2.imread(str(img))
    xywh=np.loadtxt(str(txt))
    
    xyxy=xywh_to_xyxy(xywh,im0)
    
    for i,normalize_value in enumerate(xyxy):
        value=normalize_value*im0.shape[(i+1)%2]
        xyxy[i]=value
        
    plot_one_box(xyxy, im0, output_path, line_thickness=1)
        
    


