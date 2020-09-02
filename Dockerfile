#ARG USER
#FROM amr-registry.caas.intel.com/aipg/$USER-muj_base
#FROM amr-registry.caas.intel.com/aipg/smiret-muj_base
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libosmesa6-dev \
    libglew-dev \
    software-properties-common \
    net-tools \
    python3-pip \
    python3-numpy \
    python3-scipy \
    unzip \
    vim \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

ENV LANG C.UTF-8

 # Mujoco_py
#RUN curl libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev software-properties-common
RUN mkdir -p ~/.mujoco \
    && wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
    && unzip -n mujoco.zip -d ~/.mujoco \
    && rm mujoco.zip

ARG MUJOCO_KEY
ENV MUJOCO_KEY=$MUJOCO_KEY
ENV LD_LIBRARY_PATH /root/.mujoco/mjpro150/bin:$LD_LIBRARY_PATH
RUN echo $MUJOCO_KEY | base64 --decode > /root/.mujoco/mjkey.txt
ADD .mujoco/mjkey.txt /root/.mujoco/mjkey.txt

#ENV LD_LIBRARY_PATH /.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

RUN python3 -m pip install --upgrade pip

RUN git clone https://github.com/openai/mujoco-py.git
WORKDIR /mujoco-py
RUN cd /mujoco-py
RUN cp vendor/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy
RUN cp vendor/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

COPY ./requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install mujoco_py==1.50.1.68
RUN chmod -R 777 /usr/local/lib/python3.6

COPY ./pddm_requirements.txt .
RUN pip3 install --no-cache-dir -r pddm_requirements.txt

ADD .mujoco/ .mujoco/

#Optional
# Create default user
RUN groupadd -g 17685 aipg_labs
RUN echo '%aipg_labs ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Hacky add uid to sudoers
RUN useradd -u 11960590 -g 100 nikhila1 && \
    rm -rf /var/log/lastlog && \
    rm -rf /var/log/faillog
RUN usermod -aG sudo nikhila1
RUN mkdir -p /home/nikhila1; cp -r /root/.mujoco  /home/nikhila1/.; chown -R nikhila1:aipg_labs /home/nikhila1/.
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mbertran/.mujoco/mjpro150/bin' >> /home/nikhila1/.bash_profile
RUN chown nikhila1:aipg_labs /home/nikhila1/.bash_profile









