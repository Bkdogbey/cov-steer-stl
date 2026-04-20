.PHONY: run test test-all lint clean

PYTHON = python
SRC = src

run:
	PYTHONPATH=$(SRC) $(PYTHON) main.py

test:
	PYTHONPATH=$(SRC) $(PYTHON) -m pytest tests/ -v -m "not slow and not integration"

test-all:
	PYTHONPATH=$(SRC) $(PYTHON) -m pytest tests/ -v

lint:
	ruff check $(SRC) main.py tests/

clean:
	find . -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf data/results/
