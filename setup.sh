
git clone https://github.com/WongKinYiu/yolov7.git ./workspace/yolov7

mkdir workspace/yolov7/weights && \
    wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt -P ./workspace/yolov7/weights


mkdir data/original
mkdir data/processed

docker compose build

docker compose run yolov7