FROM nvcr.io/nvidia/pytorch:21.08-py3

RUN apt update && apt-get install -y wget
RUN apt install -y zip htop screen libgl1-mesa-glx
RUN pip install seaborn thop

RUN mkdir workspace && cd workspace

CMD ["bash"]
