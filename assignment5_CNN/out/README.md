# Language Analytics - Spring 2021

This repository contains all of the assignments and code-along sessions of the Language Analytics course.

## Running my scripts

For running my scripts I'd recommend doing the following from your terminal as a setup as well as following the README's for each assignment.

__MAC/LINUX/WORKER02__
```bash
git clone https://github.com/emiltj/cds-language.git
cd cds-language
bash ./create_lang_venv.sh
```
__WINDOWS:__
```bash
git clone https://github.com/emiltj/cds-language.git
cd cds-language
bash ./create_lang_venv_win.sh
```

## Repo structure and files

This repository has the following directory structure:

| Column | Description|
|--------|:-----------|
```data```| Contains the data used in both the scripts and the notebooks.
```out``` | Contains the outputs from running the script.
```cnn_artists.py```| The script to be executed from the terminal.

Furthermore it contains the files:
- ```./create_lang_venv.sh``` -> A bash script which automatically generates a new virtual environment, and install all the packages contained within ```requirements.txt```
- ```requirements.txt``` -> A list of packages along with the versions that are certain to work
- ```README.md``` -> This very readme file
