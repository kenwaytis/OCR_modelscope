FROM paidax/dev-containers:modelscope-v0.7

RUN pip install \
    fastapi \
    uvicorn \
    pydantic==1.10.8 \
    loguru && \
    rm -rf /root/.cache/pip/*

WORKDIR /home/ocr

COPY . .

RUN python download_model.py
