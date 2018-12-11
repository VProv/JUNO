FROM python:3.7-slim
# install the notebook package
RUN pip install --no-cache --upgrade pip && \
    pip install --no-cache notebook

# create user with a home directory
ARG NB_USER
ARG NB_UID
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

RUN apt-get update && apt-get install -y \
        zip \
        unzip 
        
RUN apt-get update
RUN apt-get -y install hdf5-devel lzo-devel bzip2-devel
RUN pip install git+git://github.com/pytables/pytables@develop
RUN python -c "import tables; tables.test()"

RUN pip install -r ~/tasks/requirements.txt

USER ${NB_USER}