FROM tensorflow/tensorflow:latest-gpu-jupyter

WORKDIR /trainer

COPY requirements.txt /trainer

RUN pip install Cython numpy

RUN apt-get update -y && apt-get install -y \
    --no-install-recommends \
    build-essential \
    gcc \
    libsndfile1 \
    fluidsynth \
    ffmpeg \
    portaudio19-dev

RUN pip install -r requirements.txt

EXPOSE 5000

CMD jupyter notebook --ip=0.0.0.0 --port 8888 --allow-root
