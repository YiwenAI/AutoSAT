FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN  sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list && \
    apt-get clean && \
    apt-get update && \
    apt-get install -y --fix-missing vim \
    wget \
    zip \
    python3.10 \
    python3-pip \
    python3-opengl \
    xvfb \
    ffmpeg \
    xorg-dev \
    curl \
    cmake \
    zlib1g \
    zlib1g-dev \
    swig \
    rsync \
    tree \
    git \
    build-essential \
    openssh-server \
    sshfs\
    tmux \
    zsh \
    bc \
    locales && locale-gen en_US.UTF-8

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# install python libs
ADD requirements.txt /root/
RUN pip3 install -r /root/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

ADD ./ /root/AutoSAT/
WORKDIR /root/AutoSAT/

# ssh config
RUN mkdir -p /root/.ssh
ADD ./src_files/.ssh/ /root/.ssh/
RUN chmod 600 /root/.ssh/id_rsa
RUN rm /etc/ssh/sshd_config
ADD ./src_files/sshd_config /etc/ssh/

# pip config
RUN mkdir -p /root/.pip
ADD ./src_files/pip.conf /root/.pip/

# mount data_server
RUN mkdir -p /root/data_server
ADD ./src_files/run.sh /root/
WORKDIR /root/
RUN chmod 777 run.sh
#ENTRYPOINT ["bash", "-c", "/root/run.sh start && tail -f /dev/null"]

