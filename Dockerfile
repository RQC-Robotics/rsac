FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
#redo file
ARG DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV LANG C.UTF-8
ENV MUJOCO_GL "egl"

RUN apt-get -yqq update && \
	apt-get --no-install-recommends install -y \
        wget \
        libegl1 \
        libgl1-mesa-glx && \
        #ffmpeg \
        #libglew2.0  && \
        #libglew2.0 libglfw3 libglew-dev \
        #libgl1-mesa-glx libosmesa6-dev && \
    apt-get clean && rm -rf /var/lib/apt/list

RUN wget \
        https://github.com/deepmind/mujoco/releases/download/2.1.3/mujoco-2.1.3-linux-x86_64.tar.gz	-O /tmp/mujoco.tar.gz && \
    mkdir /root/.mujoco && \
    tar -zxf /tmp/mujoco.tar.gz -C /root/.mujoco && \
    rm /tmp/mujoco.tar.gz

ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco-2.1.3/lib:$LD_LIBRARY_PATH \
    MJLIB_PATH=/root/.mujoco/mujoco-2.1.3/lib/libmujoco.so

RUN pip install --no-cache-dir \
      pytorch3d \
      matplotlib \
      tensorboard \
      gym \
      dm_control \
      ruamel.yaml

EXPOSE 8888 6006

VOLUME /app
WORKDIR /app

COPY src ./contrastive

ENTRYPOINT ["/opt/conda/bin/python", "-m", "contrastive.train"]