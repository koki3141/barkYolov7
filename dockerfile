FROM nvcr.io/nvidia/pytorch:21.08-py3

RUN apt update && apt-get install -y wget
RUN apt install -y zip htop screen libgl1-mesa-glx
RUN pip install seaborn thop

WORKDIR /workspace/yolov7
RUN git clone https://github.com/WongKinYiu/yolov7.git ./yolov7
RUN mkdir ./yolov7/weights && \
    wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt -P ./yolov7/weights

CMD ["bash"]
