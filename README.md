# メモ
## wsl2のインストール
https://learn.microsoft.com/ja-jp/windows/wsl/install

## dockerのインストール
https://docs.docker.com/engine/install/
## NVIDIA Driverのダウンロード
https://www.nvidia.com/Download/index.aspx
## nvidia-container-runtimeのインストール
```
curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
sudo apt-get update
sudo apt-get install nvidia-container-runtime
```
https://nvidia.github.io/nvidia-container-runtime/

## dockerのコンテナ作成
```
docker compose build
```
## dockerの起動
```
docker compose run yolov7
```
## YOLOv7の推論
```
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --device 0 --source inference/images/horses.jpg
```

## 参考文献
>https://docs.docker.jp/compose/gpu-support.html
>https://docs.docker.jp/config/containers/resource_constraints.html#resource-constraints-gpu
>https://nvidia.github.io/nvidia-container-runtime/
>https://www.nvidia.com/Download/index.aspx