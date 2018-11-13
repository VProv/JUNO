FROM jupyter/scipy-notebook:cf6258237ff9

# Basic
ENV NB_USER dockprov
ENV NB_UID 1001
ENV HOME /home/${NB_USER}

COPY . ${HOME}
USER root

RUN pip install -r ~/tasks/requirements.txt
RUN pip install --no-cache-dir notebook==5.*

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}
WORKDIR ${HOME}
#Start