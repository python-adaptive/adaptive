# Docker image for testing adaptive
FROM conda/miniconda3:latest

# make our environment sane
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

COPY environment.yml test-requirements.txt /

RUN conda env update --quiet -n root -f environment.yml
RUN conda clean --yes --all
RUN pip install -r test-requirements.txt
