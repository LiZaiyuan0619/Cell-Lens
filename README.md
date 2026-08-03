# Cell-Lens

This repository contains the code and processed data accompanying the paper
*Cell-Lens: Advancing LLM Reasoning at Single-Cell Resolution*.

Cell-Lens is a training-free structured representation that exposes measured
single-cell modalities to large language models as typed text blocks.

## Repository structure

- `code/preprocessing/`: data preprocessing and benchmark construction.
- `code/evaluation/`: Cell-Lens evaluation runners.
- `code/analysis/`: statistical analysis scripts.
- `code/plotting/`: plotting scripts and source data for the main figures.
- `data/`: compact processed datasets and benchmark inputs.
- `AtomWorldBench/src/config/models.yaml`: model configuration templates.

## Installation

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## API configuration

The evaluation runners use an OpenAI-compatible chat-completions interface.
Set the API credentials in the shell before running an evaluation:

```bash
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="https://your-authorized-endpoint.example/v1"
```

## Example

Run commands from the repository root. For example:

```bash
python code/evaluation/run_cite_ab.py \
  --model deepseek_v4_flash \
  --npz data/cellspectrum_examples/cite_labeled.npz \
  --arms l0spec,f3,f3shuf \
  --seed 0
```

The repository includes compact processed data for representative evaluations.
Large public source matrices are not redistributed. Third-party datasets and
benchmark inputs remain governed by their original terms.

## License

The code is released under the MIT License; see `LICENSE-CODE`.

