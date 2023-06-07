#!/bin/bash
# Update the apt
apt update && apt upgrade
apt install -y software-properties-common
add-apt-repository ppa:deadsnakes/ppa -y
apt update

# Install Python 3.10.11
apt install python3.10

# Configure default Python
update-alternatives --install /usr/bin/python python /usr/bin/python3.10 2

# Download Python relevant modules
apt install python3.10-dbg -y
apt install python3.10-dev -y
apt install python3.10-venv -y
apt install python3.10-distutils -y
apt install curl
# Download get-pip.py
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

# Install pip for Python 3.10
python3.10 get-pip.py
# torchlight install( not by usual way )
cp -r /home/torchlight/torchlight /usr/local/lib/python3.10/dist-packages/
#cp -r /home/torchlight/torchlight /home/ske/lib/python3.10/site-packages/
#fixing the "collection" module error
pip install typing
pip install torchpack==0.0.6
#sed -i '4c\from typing import Iterable' /home/ske/lib/python3.10/site-packages/torchpack/config.py
sed -i '4c\from typing import Iterable' /usr/local/lib/python3.10/dist-packages/torchpack/config.py
# torch , tensorboardX install ( for cuda 11.7 )
pip install tensorboardX
pip install torch==2.0 torchvision==0.15 torchaudio
# torch 2.0.1 torchvision 0.15.2
#install other modules
apt-get update
apt-get upgrade
apt-get install rsync
pip install scikit-learn
pip install tqdm
pip install h5py
pip install matplotlib
pip install PyYAML