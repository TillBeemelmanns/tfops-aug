FROM tensorflow/tensorflow:2.11.0-gpu

RUN apt update && apt install -y python3-opencv libopencv-dev

# install Python package dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt && \
    rm requirements.txt
