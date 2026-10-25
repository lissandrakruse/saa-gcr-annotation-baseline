# Software verification

The package has 38 automated tests: 35 unit/regression tests plus 3 notebook publication checks. All fixtures are synthetic software inputs, not scientific observations or new human labels.

Coverage includes interval boundaries, observation support, gaps, empty metrics, event matching, input validation, feature selection, fixed model configuration, source-preserving normalization, serialization, notebook execution and output-free publication. Randomized checks include interval-union and exhaustive-assignment oracles.

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m saa_baseline --help
python -m compileall -q src tests
```

Local checks passed on CPython 3.12.14 with the pinned dependencies. GitHub Actions runs synthetic tests in Python 3.12 on Ubuntu with read-only repository permissions. Check Actions for the status of a specific commit.

Full reproduction requires separately obtained authorized inputs and is not run by public CI. These checks do not establish physical accuracy, data rights or an independent evaluation. Cross-platform bitwise identity is not guaranteed.
