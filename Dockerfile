FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
#redo file
ARG DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV LANG C.UTF-8
ENV MUJOCO_GL "egl"

RUN apt-get -yqq update && \
	apt-get --no-install-recommends install -y \
        wget \
        ffmpeg \
        libgl1-mesa-glx && \
        #libglew2.0 libglfw3 libglew-dev \
        #libgl1-mesa-glx libosmesa6-dev && \
    apt-get clean && rm -rf /var/lib/apt/list

RUN wget \
        https://github.com/deepmind/mujoco/releases/download/2.1.3/mujoco-2.1.3-linux-x86_64.tar.gz	-O /tmp/mujoco.tar.gz && \
    mkdir /root/.mujoco && \
    tar -zxf /tmp/mujoco.tar.gz -C /root/.mujoco && \
    rm /tmp/mujoco.tar.gz

ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco-2.1.3/lib:$LD_LIBRARY_PATH \
    #LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    #LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so \
    MJLIB_PATH=/root/.mujoco/mujoco-2.1.3/lib/libmujoco.so

COPY src /tmp/contrastive/
COPY environment.yml train.py /tmp/

RUN conda env update --name base --file /tmp/environment.yml && \
    #conda env create -f environment.yml && \
    rm /tmp/environment.yml

EXPOSE 8888 6006


VOLUME /app
WORKDIR /app

#ENTRYPOINT ['bash']
ENTRYPOINT ["/opt/conda/bin/python", "/tmp/train.py"]