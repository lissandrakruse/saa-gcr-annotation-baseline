# SAA/GCR annotation baseline

Python code for a reproducible annotation-comparison baseline, with a command-line interface, synthetic software tests and an unexecuted notebook template.

## Install and test

Python 3.12 is required. Dependencies are pinned in `requirements.txt`.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
python -m saa_baseline --help
```

The CLI provides `normalize`, `train`, `predict`, `evaluate` and `reproduce` commands. Run a command with `--help` to inspect its required local inputs. Tests use synthetic fixtures and need no project data. The notebook template uses the same package; see its setup cell for local path configuration.

## Code-only distribution

This repository contains source code, synthetic tests, an unexecuted notebook and supporting configuration. It does not distribute datasets, annotation records, trained models, result tables, images, videos or private source-access links. Obtain required inputs and permission to use them separately. Keep local outputs out of this repository.

The software computes descriptive annotation agreement. Its tests do not establish scientific validity, physical accuracy or independent evaluation. A generated local artifact's publication or license flags do not grant permission to distribute that artifact.

Only load joblib/pickle artifacts that you produced or independently trust: loading them can execute code. A matching adjacent hash does not authenticate an unknown sender.

## License

The code in this repository is available under the [MIT License](LICENSE). This license does not cover data, annotations, third-party materials or trained artifacts that are not included here. External dependencies retain their own licenses.
