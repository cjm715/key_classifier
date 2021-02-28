
FROM tensorflow/tensorflow:latest-gpu-jupyter

WORKDIR /app


COPY requirements.txt /app

RUN pip install Cython numpy

RUN apt-get update -y && apt-get install -y \
    --no-install-recommends \
    build-essential \
    gcc \
    libsndfile1 \
    fluidsynth \
    ffmpeg \
    portaudio19-dev \
    libpcre3 \
    libpcre3-dev

RUN pip install -r requirements.txt

ADD . /app
    
CMD ["uwsgi", "app.ini"]



