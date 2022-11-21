FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive

WORKDIR "/"
RUN apt update
RUN apt install -y gedit=3.36.2-0ubuntu1
RUN apt install -y python3=3.8.2-0ubuntu2
RUN apt install -y python-is-python3=3.8.2-4
RUN apt install -y curl=7.68.0-1ubuntu2.14
RUN curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN apt install -y python3-distutils=3.8.10-0ubuntu1~20.04
RUN python3 get-pip.py "pip == 21.1" "setuptools == 57.0" "wheel == 0.36"
RUN apt install -y git=1:2.25.1-1ubuntu3.6
RUN apt install -y python3-pip=20.0.2-5ubuntu1.6
RUN git clone https://github.com/iwbn/flow-supervisor.git
WORKDIR "/flow-supervisor"
RUN git submodule init
RUN git submodule update
RUN pip install -r requirements.txt

RUN apt-get clean && \
	    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

