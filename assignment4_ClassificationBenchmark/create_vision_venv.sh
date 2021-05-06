#!/usr/bin/env bash

VENVNAME=classification_env 

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

pip install ipython
pip install jupyter
pip install matplotlib

test -f requirements.txt && pip install -r requirements.txt

deactivate
echo "build $VENVNAME"