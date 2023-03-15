FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

RUN apt update && \
	apt install -y screen vim

RUN pip install --upgrade pip

COPY requirements.txt ./
RUN pip install -r requirements.txt && rm requirements.txt

WORKDIR /sentinel-segmentation
