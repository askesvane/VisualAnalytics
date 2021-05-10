#!/usr/bin/env bash

VENVNAME=Covid_env 

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

pip install ipython
pip install jupyter
pip install matplotlib
pip install opencv-python

test -f requirements.txt && pip install -r requirements.txt

deactivate
echo "build $VENVNAME"