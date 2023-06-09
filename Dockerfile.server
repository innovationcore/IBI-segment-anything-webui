# syntax=docker/dockerfile:1

FROM nikolaik/python-nodejs:python3.11-nodejs18
WORKDIR /app

RUN pip install git+https://github.com/facebookresearch/segment-anything.git
RUN pip install opencv-python pycocotools matplotlib onnxruntime onnx

RUN mkdir -p model
RUN mkdir -p overlays
RUN wget --no-clobber https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O model/sam_vit_b_01ec64.pth
RUN pip install torch torchvision ftfy regex tqdm
RUN pip install git+https://github.com/openai/CLIP.git

COPY scripts/requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python3" ,"scripts/server.py"]
EXPOSE 8000