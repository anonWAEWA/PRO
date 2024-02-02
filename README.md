# PRO

## Setup Environment

```
conda env update --file environment.yml
```

## Run Test-time training Experiment

Run the zero-shot code first to build features

```
python scripts/zeroshot.py -d {dataset}
```

The following command runs TTT experiment.

```
python scripts/run_tta.py -d {dataset}
```

To see all the options

```
python scripts/run_tta.py -h
```
