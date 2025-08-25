#!/usr/local/bin/sh

set -e # exit script on error

# Create and source virtual environment
#python3 -m venv .venv
source /home/ghesquie/projects/lsms_codes/Frag3D/.venv/bin/activate 

# Install dependencies
#pip3 install -r requirements.txt

# Source akantu built from source 
source /home/ghesquie/projects/lib/akantu/build/akantu_environement.sh

set +e # return to default shell behaviour 
