.PHONY: setup

setup:
	conda create -n nus-dsa5106-gp python=3.10
	conda activate nus-dsa5106-gp && python -m pip install -r upstream/requirements.txt

