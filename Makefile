.PHONY: install fix clean char word eval galle multi

VENV := .venv
ACTIVATE := . $(VENV)/bin/activate

install:
	@python3 -m venv $(VENV)
	@${ACTIVATE} && pip install --upgrade pip
	@$(ACTIVATE) && pip install -r requirements.txt
	@$(ACTIVATE) && python -m spacy download en_core_web_sm

fix:
	@$(ACTIVATE) && isort . && black .

clean:
	rm -rf $(VENV)
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete


char: fix
	@$(ACTIVATE) && python -m galle.char

word: fix
	@$(ACTIVATE) && python -m galle.word

eval: fix
	@$(ACTIVATE) && python -m galle.eval

galle: fix char word eval

multi: fix
	@$(ACTIVATE) && python -m galle.multi