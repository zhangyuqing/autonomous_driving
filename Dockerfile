# Use latest Python runtime as base image
FROM continuumio/anaconda3

LABEL maintainer="Yuqing Zhang <yqng.zh@gmail.com>" \
    description=" This image contains a pickled version of the predictive model\
    that can be accessed using a REST API (created in Flask)."

# Set the working directory to /app and copy current dir
WORKDIR /app
COPY environment.yml ./
COPY detra ./
COPY exploration/img_cttrk/CenterTrack ./
# COPY exploration/img_ctnet/cocoapi ./
COPY user_data ./

RUN conda env create -f environment.yml
RUN echo "conda activate CenterTrk" > ~/.bashrc
ENV PATH /opt/conda/envs/CenterTrk/bin:$PATH

EXPOSE 5000

CMD ["python", "detra/app.py"]