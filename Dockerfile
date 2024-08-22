FROM hpcaitech/pytorch-cuda:2.1.0-12.1.0

# metainformation
LABEL org.opencontainers.image.source = "https://github.com/hpcaitech/Open-Sora"
LABEL org.opencontainers.image.licenses = "Apache License 2.0"
LABEL org.opencontainers.image.base.name = "docker.io/library/hpcaitech/pytorch-cuda:2.1.0-12.1.0"

# Set the working directory
WORKDIR /workspace/Open-Sora
# Copy the current directory contents into the container at /workspace/Open-Sora
COPY . .

# inatall library dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Install some basic utilities.
RUN apt-get update &&  DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata && apt-get install -y \
  curl \
  ca-certificates \
  sudo \
  git \
  bzip2 \
  libx11-6 \
  tmux \
  wget \
  build-essential \
  zsh \
  && rm -rf /var/lib/apt/lists/*

# set the mount
RUN mkdir /app && mkdir /app/workspace && mkdir /app/nas_hdd0 && mkdir /app/nas_hdd1 && mkdir /app/nas_hdd2 

# Create a non-root user and switch to it.
RUN adduser -u 1000 --disabled-password --gecos '' --shell /bin/bash user \
  && chown -R user:user /app

# 改变用户的UID和GID
RUN usermod -u 1000 user && usermod -G 1000 user

RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# install flash attention
RUN pip install flash-attn --no-build-isolation

# install apex
RUN pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git

# install xformers
# RUN pip install xformers --index-url https://download.pytorch.org/whl/cu121

# install this project
RUN pip install -v .

# install other dependencies
RUN pip install  webdataset==0.2.5 trimesh==3.23.5 icecream jaxtyping==0.2.23 omegaconf==2.3.0 kornia==0.6.0 scikit-image==0.20.0 lpips==0.1.4



# All users can use /home/user as their home directory.
# 给user创建一个home，但是需要注意这个home的权限必须< 0755否则ssh无法连接
# 
ENV HOME=/home/user 
RUN mkdir $HOME/.cache $HOME/.config $HOME/.ssh \
  && chmod -R 0755 $HOME && \  
  echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCiOeBN5ak5K2HH4lPbwFMjpsRnHbF5XRPY2gF61HVKbDQR3ZsCnOe6f5s5GMybhvFqilHPfvW8ASJIp+16qzOGwwuRba0o8YtBkRv65LfQj+abT86G2g8HvAcFseobnxu1e0CbsINZngH1E2dr/wo0d0QTSo90CEOL9+kg+UKB0w7iNFJ8UuOxinIKrTip2J7xTOj1Xl64ZI3Txc1WNI/R/6la+qvMr5RtPBelUhA1+T8gB5iOwwIkMdwKnDBBQ5L22UjhasiLD26AOIbHdcijSgCeIrtv9unWwO9g78d7DKqliEOofAwQ0+hcs5PQosixQzevADeodGgEp6aA8Z7EXWPnVimeKkSbqJloLKpJtBLoehFBd7BrCWxMYTdmYVbFnWNEGBWBXNLN9J9dk/QMOarD25VbodjbCi78CHGu78JTvu4xg+GZhtoWNgy+t7oMODo+QHHy1dtAY5ZTyZILzalRrOj8NAmDqyPjx+Ss9bJ9bQ9hgZooaEs1fqDP0Q0= fc@macprodeMacBook-Pro-278.local\n" >  $HOME/.ssh/authorized_keys && \ 
  chmod 700 $HOME/.ssh && \
  chmod 600 $HOME/.ssh/authorized_keys && \
  chmod 777 /app && cd /app  

# # set zsh
# RUN sh -c "$(wget https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)" 
# RUN git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k 
# RUN git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting \
#   && git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

# setup ssh
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
