## About The Project

An OCR server that runs in a Docker container and provides a RESTful API. Support for Chinese and English.
Model has integrated line detection and text recognition , out of the box .
Model files and inference code from [ModelScope](https://modelscope.cn).

The line detection model is tested on the MTWI test set with the following results:

| Backbone | Recall | Precision | F-score |
| -------- | ------ | --------- | ------- |
| ResNet18 | 68.1   | 84.9      | 75.6    |

BenchMark for text recognition model is not yet available.

## Usage

### 1. Environmental requirements

1. Requires Docker engine or Docker Desktop.

2. Since GPUs are used in Docker, Nvidia Docker also needs to be installed to provide GPU invocation capabilities in the container.The installation process can be referenced:[Nvidia container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

All other dependencies and models are included in the pre-built Docker image, but of course you can build the exact same image from scratch based on the source code.

### 2. Installation

1. Clone the repo

```shell
git clone https://github.com/kenwaytis/OCR_modelscope.git
```

2. (opsition) Start the server with the default Docker Image

```shell
docker compose up
```

3. (opsition) Build the image from scratch

3.1 Modify the docker-compose.yml file from

```
image: paidax/ocr_modelscope:0.6.3
```

​		to

```
image: namespace/ocr_modelscope:0.6.3
```

3.2 Start the server

```
docker compose up
```

### 3. API interface description

- Because fastAPI was used to build the server, you can view the automatically generated documentation instructions at **localhost:9533/docs**.

- Description:

**URL:**

localhost:9533/ocr_system

**Request method:**

POST

**json description:**

| field name | required or not | type      | note                     |
| ---------- | --------------- | --------- | ------------------------ |
| images     | yes             | list[str] | URL address of the image |

**Request json example:**

```json
{
  "images":["img1", "img2", "img3"]
}
```

## Acknowledgments

[ModelScope](https://modelscope.cn)

```
@article{tang2019seglink++,
  title={Seglink++: Detecting dense and arbitrary-shaped scene text by instance-aware component grouping},
  author={Tang, Jun and Yang, Zhibo and Wang, Yongpan and Zheng, Qi and Xu, Yongchao and Bai, Xiang},
  journal={Pattern recognition},
  volume={96},
  pages={106954},
  year={2019},
  publisher={Elsevier}
}
```

