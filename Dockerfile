FROM nvcr.io/nvidia/pytorch:22.12-py3 
ARG DEBIAN_FRONTEND=noninteractive

RUN pip3 install --upgrade pip
#RUN pip3 install clearml
#RUN pip3 install clearml-agent
#RUN pip3 install optuna
#ADD clearml.conf /root

#update api
RUN apt-get update
RUN apt-get install openslide-tools -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install -y curl
RUN apt-get install -y unzip
RUN apt-get install -y libtiff5-dev

#build dep openslide
RUN apt-get install build-essential -y
RUN apt-get install -y git
RUN apt-get install autoconf -y
RUN apt-get -y install libtool
RUN apt-get install libopenjp2-7-dev -y
RUN apt-get install -y libglib2.0-dev
RUN apt-get install -y libcairo2-dev
RUN apt-get install -y libgdk-pixbuf2.0-dev
RUN apt-get install -y libxml2-dev
RUN apt-get install -y libsqlite3-dev
RUN apt-get install -y libsdl2-dev

#remove existing openslide
RUN apt-get remove libopenslide0 --purge -y
#build openslide
WORKDIR /opt/
RUN git clone https://github.com/innovationcore/openslide.git
#ADD openslide /opt/openslide
WORKDIR /opt/openslide
RUN git checkout origin/isyntax-support
RUN autoreconf --install --force --verbose
RUN ./configure
RUN make install

WORKDIR /opt/

RUN git clone https://github.com/innovationcore/IBI-segment-anything-webui

WORKDIR /opt/IBI-segment-anything-webui
RUN mkdir model
# download the model to `model/`
RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O model/sam_vit_b_01ec64.pth
# https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
# https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# e.g. for Segment Anything
RUN pip3 install git+https://github.com/facebookresearch/segment-anything.git
RUN pip3 install opencv-python pycocotools matplotlib onnxruntime onnx

# e.g. for CLIP
#RUN pip3 install pytorch torchvision
#RUN pip3 install ftfy regex tqdm
RUN pip3 install git+https://github.com/openai/CLIP.git

# python server as backend
RUN pip3 install torch numpy 'uvicorn[standard]' fastapi pydantic python-multipart Pillow click

RUN curl -sL https://deb.nodesource.com/setup_16.x | bash -
RUN apt-get update
RUN apt-get install -y nodejs

RUN npm install -g npm@latest
RUN npm install
RUN npm install form-data
RUN npm i

ADD run_server.sh /opt/IBI-segment-anything-webui/
CMD /opt/IBI-segment-anything-webui/run_server.sh