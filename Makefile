IMAGE:=sentinel-segmentation
VERSION:=1.0
LOCAL_DATASET:=/data/jk/sentinel
DOCKER_DATASET:=/data/datasets/sentinel

all: run

build:
	docker build -t $(IMAGE):$(VERSION) .

run:
	docker run -it --rm \
		--gpus all --shm-size="7g" \
		--net=host \
		-v $(LOCAL_DATASET):$(DOCKER_DATASET) -v $(PWD):/sentinel-segmentation \
		$(IMAGE):$(VERSION)

