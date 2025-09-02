SHELL := /bin/bash

lora:
	module purge  && \
	module load JupyterLab/3.5.0-GCCcore-11.3.0  && \
	source /projects/ec517/venvs/in5310h25/bin/activate  && \
	python lora.py

vit:
	module purge && \
	module load JupyterLab/3.5.0-GCCcore-11.3.0 && \
	source /fp/projects01/ec517/venvs/in5310h25/bin/activate && \
	python vanilla_vit.py
