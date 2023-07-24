FROM paidax/dev-containers:modelscope-v0.7.1

RUN pip install \
    fastapi \
    uvicorn \
    pydantic==1.10.8 \
    loguru && \
    rm -rf /root/.cache/pip/*

WORKDIR /home/ocr

COPY ./download_model.py /home/ocr/download_model.py

RUN python download_model.py

COPY . .
