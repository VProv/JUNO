FROM python:3.7-slim
# install the notebook package
RUN pip install --no-cache --upgrade pip && \
    pip install --no-cache notebook

# create user with a home directory
ARG NB_USER jovian
ARG NB_UID 10001
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}
WORKDIR ${HOME}

COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}
RUN pip install -r ~/tasks/requirements.txt
USER ${NB_USER}