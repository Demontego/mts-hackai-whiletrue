FROM nvidia/cuda:11.0-base

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl

RUN apt-get -y install python3.9
RUN apt-get -y install python3-pip
RUN apt-get -y install zip
RUN python3.9 -V

ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update &&\
    apt-get install -y\
    bash\
    build-essential\
    gcc\
    libffi-dev\
    musl-dev\
    openssl\
    libpq-dev\
    libxml2\
    git
   
COPY requirements.txt ./requirements.txt
RUN pip3 install -r ./requirements.txt
COPY ./backend.py ./backend.py

ENTRYPOINT python3 ./backend.py
