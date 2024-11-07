#
# Created on Sat May 03 2025 21:59:02
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#

# Do not add --platform=linux/blabla since this is intended for multiplatform builds
ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

ARG IMAGE_USER
ARG HOME_FOLDER
ARG CODE_FOLDER
ARG TIME_ZONE

ENV HOME=${HOME_FOLDER}
WORKDIR ${HOME_FOLDER}/

# Fix apt install stuck problem
ENV DEBIAN_FRONTEND=noninteractive
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Copy home folder config
COPY --from=home-folder-config . ${HOME_FOLDER}/

# Remove cuda source list
# E: Failed to fetch https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/Packages.gz File has unexpected size (853674 != 833365). Mirror sync in progress? [IP: 23.58.157.10 443]
RUN [ ! -f /etc/apt/sources.list.d/cuda.list ] || rm /etc/apt/sources.list.d/cuda.list

# update all obsolete packages to latest, install sudo, and cleanup
RUN apt update -o Acquire::Check-Valid-Until=false -o Acquire::AllowInsecureRepositories=true -o Acquire::AllowDowngradeToInsecureRepositories=true && \
    apt full-upgrade -y && \
    apt install -y sudo ca-certificates && \
    apt autoremove -y && \
    apt autoclean -y

# fix local time problem
RUN apt-get install -y --no-install-recommends tzdata && \
    ln -fs /usr/share/zoneinfo/${TIME_ZONE} /etc/localtime && \
    echo "${TIME_ZONE}" > /etc/timezone && \
    dpkg-reconfigure --frontend noninteractive tzdata

# Install some goodies
RUN apt-get install -y --no-install-recommends \
    build-essential \
    clang-format \
    curl \
    dirmngr \
    git \
    gnupg \
    htop \
    less \
    locate \
    lsb-release \
    nano \
    ncdu \
    net-tools \
    perl \
    screen \
    software-properties-common \
    tmux \
    tmuxp \
    tree \
    unzip \
    valgrind \
    vim \
    wget \
    zsh

# upgrade cmake to kitware official apt-get repo release version
RUN wget https://apt-get.kitware.com/kitware-archive.sh -O- | sh -s && \
    apt-get upgrade -y cmake && \
    apt-get autoremove -y

# # Add a new group and user
# RUN addgroup --gid 1000 ${IMAGE_USER} && \
#     adduser --uid 1000 --ingroup ${IMAGE_USER} --home ${HOME_FOLDER} --shell /bin/zsh --disabled-password --gecos "" ${IMAGE_USER} && \
#     echo "${IMAGE_USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# # Fix UID/GID when mounting from host using this: https://github.com/boxboat/fixuid
# RUN curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.6.0/fixuid-0.6.0-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - && \
#     chown root:root /usr/local/bin/fixuid && \
#     chmod 4755 /usr/local/bin/fixuid && \
#     mkdir -p /etc/fixuid

# # Switch to the new user
# USER ${IMAGE_USER}:${IMAGE_USER}

# install zsh, Oh-My-Zsh, and plugins
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/latest/download/zsh-in-docker.sh)" -- \
    -t https://github.com/romkatv/powerlevel10k \
    -p git \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions \
    -p https://github.com/zsh-users/zsh-syntax-highlighting \
    -a "[[ ! -f ${HOME_FOLDER}/.p10k.zsh ]] || source ${HOME_FOLDER}/.p10k.zsh" \
    -a "POWERLEVEL9K_DISABLE_GITSTATUS=true" \
    -a "bindkey -M emacs '^[[3;5~' kill-word" \
    -a "bindkey '^H' backward-kill-word" \
    -a "autoload -U compinit && compinit" \
    -a "export PATH=~/.local/bin:${PATH}"

# change default shell for the ${IMAGE_USER} in the image building process for extra environment safety
RUN chsh -s $(which zsh)

# Use bash -lc for conda commands in build layers
SHELL ["/bin/bash", "-lc"]

# Install miniconda at /opt/conda
RUN wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh" -O ~/miniconda.sh && \
    sh ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Add conda to PATH
ENV PATH=/opt/conda/bin:${PATH}

# Init conda for zsh
RUN conda init zsh

# Disable auto activation
RUN conda config --set auto_activate_base false

# Install package system dependencies
RUN apt-get install -y --no-install-recommends \
    libcairo2-dev \
    libegl1-mesa-dev \
    libglib2.0-0 \
    libmagic-dev \
    libosmesa6-dev \
    libpango1.0-dev \
    libx11-dev \
    libxkbcommon-dev \
    pkg-config

# Mount Repo & install conda + uv dependencies
RUN --mount=type=bind,source=.,target=${HOME_FOLDER}/${CODE_FOLDER},rw \
    cd ${HOME_FOLDER}/${CODE_FOLDER} && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda env create -f environment.yml && \
    conda clean -afy && \
    conda run -n usf uv pip install -e '.[full]' && \
    conda run -n usf uv cache clean

# Auto-activate 'usf' for interactive shells
RUN echo 'source $(conda info --base)/etc/profile.d/conda.sh && conda activate usf' >> ${HOME_FOLDER}/.zshrc && \
    echo 'source $(conda info --base)/etc/profile.d/conda.sh && conda activate usf' >> ${HOME_FOLDER}/.bashrc

# Set usf as default conda environment
ENV PATH=/opt/conda/envs/usf/bin:${PATH}
ENV CONDA_DEFAULT_ENV=usf

# Add git safe directory
RUN git config --global --add safe.directory "*"

# end of sudo apt installs
RUN apt-get full-upgrade -y && \
    apt-get autoremove -y && \
    apt-get autoclean -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# change owner of home folder
RUN chown -R ${IMAGE_USER}:${IMAGE_USER} ${HOME_FOLDER}

# change user
USER ${IMAGE_USER}

# Set the default shell to zsh
SHELL [ "/bin/zsh", "-c" ]

# # move fixuid config
# RUN mv ${HOME_FOLDER}/fixuid-config.yml /etc/fixuid/config.yml

# Entrypoint command
# ENTRYPOINT [ "/bin/sh" , "-c", "fixuid; /bin/zsh" ]
ENTRYPOINT [ "/bin/zsh" ]
