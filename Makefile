.PHONY: setup vendor simulate classify fit brief build serve eval eval-fast test lint

setup:
	pip install -e .

vendor:
	python -c "import httpx, pathlib; pathlib.Path('src/loop/web/static/vendor').mkdir(parents=True, exist_ok=True); r = httpx.get('https://cdn.plot.ly/plotly-2.35.2.min.js', follow_redirects=True, timeout=60.0); open('src/loop/web/static/vendor/plotly-2.35.2.min.js', 'wb').write(r.content)"

simulate:
	python -m loop.simulate.campaigns

classify:
	python -m loop.reply_intel.router

fit:
	python -m loop.attribution.inference

brief:
	python -m loop.brief.builder

build: simulate classify fit brief

serve:
	uvicorn loop.api.main:app --host 0.0.0.0 --port 8000

eval:
	python eval.py

eval-fast:
	python eval.py --fast

test:
	pytest

lint:
	ruff check .
