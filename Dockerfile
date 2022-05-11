FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
ARG DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV LANG C.UTF-8
ENV MUJOCO_GL "egl"

RUN apt-get -yqq update && \
	apt-get --no-install-recommends install -y \
        wget \
        libegl1 \
        libgl1-mesa-glx && \
    apt-get clean && rm -rf /var/lib/apt/list

RUN wget \
        https://github.com/deepmind/mujoco/releases/download/2.1.5/mujoco-2.1.5-linux-aarch64.tar.gz -O /tmp/mujoco.tar.gz && \
    mkdir /root/.mujoco && \
    tar -zxf /tmp/mujoco.tar.gz -C /root/.mujoco && \
    rm /tmp/mujoco.tar.gz

ENV MJLIB_PATH=/root/.mujoco/mujoco-2.1.5/lib/libmujoco.so

# todo: pytorch3d is the only reason why conda should be used instead of pip which results in container's size is being increased
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && \
    rm /tmp/environment.yml

EXPOSE 8888 6006

VOLUME /app
WORKDIR /app

COPY src ./contrastive

ENTRYPOINT ["/opt/conda/envs/contrastive/bin/python", "-m", "contrastive.train"]
