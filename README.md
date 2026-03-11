# calypso

calypso (acronym: Circumbinary Accretion Lightcurves Yielded via Parameter-Driven Sequence Outputs), is a probabilistic emulator for circumbinary accretion variability built on global Principal Component Analysis, parameter-space interpolation, and uncertainty-aware time-series reconstruction. It provides a packaged runtime for loading trained emulator artifacts and generating synthetic accretion timeseries via stochastic sampling conditioned on binary eccentricity and mass ratio.

## Requirements

calypso requires Python 3.12 or newer.

## Installation

From the repository root, install the package in editable mode:

```bash
pip install -e .
```

This installs the `calypso` package and its declared runtime dependencies.

## How To Use calypso

### 1. Import The Package

```python
import calypso
```

### 2. Load The Emulator

```python
emu = calypso.load_emulator()
```

`load_emulator()` loads the default runtime artifact defined by the packaged manifest. If the artifact is not already available locally, calypso downloads it automatically and stores it in the local cache.

### 3. Generate Predictions

To draw stochastic samples:

```python
samples = emu.predict(eb=0.35, qb=0.75, n_samples=16)
```

To compute the mean prediction:

```python
mean_curve = emu.predict_mean(eb=0.35, qb=0.75)
```

### 4. Inspect The Output

Prediction results are returned as dictionaries keyed by component name, typically `Mb`, `M1`, and `M2`, together with the input parameter record.

To inspect the available components:

```python
print(emu.component_names)
```

## Artifact Storage

calypso stores downloaded runtime artifacts in its default cache location unless an override is provided.

To set a custom artifact directory:

```bash
export CALYPSO_ARTIFACTS_DIR=/path/to/cache
```

or

```bash
export CALYPSO_WEIGHTS_DIR=/path/to/cache
```

Set the environment variable before calling `calypso.load_emulator()`.

## Demos

Example notebooks and demonstration scripts are available in `demo/`.

These files are intended for inspection, experimentation, and downstream application development after the package has been installed.
