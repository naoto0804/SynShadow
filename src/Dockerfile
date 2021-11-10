FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel
RUN apt-get update && apt-get install -y sudo git wget tmux htop zip imagemagick
RUN echo 'export PATH="/home/${USER}/.local/bin:${PATH}"' >> ~/.bashrc
RUN pip install --upgrade pip setuptools wheel
RUN pip install cython dill dominate imageio opencv-python pillow scipy==1.2.0 tensorflow==1.13.1 tqdm torchvision==0.4.0
