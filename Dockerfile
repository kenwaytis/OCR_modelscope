FROM registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.3.0-py38-torch1.11.0-tf1.15.5-1.6.1

RUN pip install \
    fastapi \
    uvicorn \
    pydantic==1.10.8 \
    loguru && \
    rm -rf /root/.cache/pip/*

WORKDIR /home/ocr

COPY . .

RUN python download_model.py