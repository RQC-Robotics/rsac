FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime
#redo file
ARG DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV LANG C.UTF-8
ENV MUJOCO_GL "egl" #glfw

RUN apt-get -yqq update && \
	apt-get --no-install-recommends install -y \
        build-essential \
        software-properties-common \
        gcc git curl wget cmake unzip vim xorg-dev xvfb \
        libglew2.0 libglfw3 libglew-dev \
        libgl1-mesa-glx libosmesa6-dev && \
    apt-get clean && rm -rf /var/lib/apt/list

RUN wget \
        https://github.com/deepmind/mujoco/releases/download/2.1.3/mujoco-2.1.3-linux-x86_64.tar.gz	-O /tmp/mujoco.tar.gz && \
    mkdir /root/.mujoco && \
    tar -zxf /tmp/mujoco.tar.gz -C /root/.mujoco && \
    #mv /root/.mujoco/mujoco-2.1.1 /root/.mujoco/mujoco211 && \
    rm /tmp/mujoco.tar.gz

ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco-2.1.3/lib:$LD_LIBRARY_PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so \
    MJLIB_PATH=/root/.mujoco/mujoco-2.1.3/lib/libmujoco.so

RUN cd /tmp && \
    git clone https://github.com/glfw/glfw.git && \
    cd glfw && \
    cmake . && make && make install


COPY src /tmp/contrastive/
COPY environment.yml /tmp
#COPY setup.py pyproject.toml main.py requirements.txt environment.yml /tmp

#RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
RUN cd /tmp && \
    conda env create -f environment.yml && \
    rm /tmp/environment.yml
    #/opt/conda/envs/sac/bin/python setup.py install

EXPOSE 8888 6006


VOLUME /app
WORKDIR /app

ENTRYPOINT ['bash']
#ENTRYPOINT ["/opt/conda/envs/sac/bin/python", "/tmp/main.py"]