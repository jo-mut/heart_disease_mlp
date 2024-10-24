VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

activate: $(VENV)/bin/activate

all: train test

train: 
	python3 src/train.py

test: 
	python3 src/test.py

clean:
	# Indent with a tab
	rm -rf __pycache__
	# Indent with a tab
	rm -rf venv