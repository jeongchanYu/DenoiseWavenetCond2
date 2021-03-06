FROM tensorflow/tensorflow:2.5.0
MAINTAINER yjc133

RUN apt-get -y update
RUN apt-get install -y tmux
RUN apt-get install -y nano
RUN apt-get install -y libsndfile1
RUN apt-get install -y python3.7
RUN pip3 install soundfile
RUN pip3 install matplotlib
RUN pip3 install pypesq

CMD ["bash"]
