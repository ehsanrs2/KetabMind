# Documentation

Set `EMBED_MODEL` in `.env` to choose the embedding backend:

- `small` (default) uses BGE small (384 dims)
- `base` uses BGE base (768 dims)
- `mock` uses a lightweight mock embedder for tests

The first run with a real model downloads roughly 400 MB of weights.

Set the LLM environment variables in `.env` to control answer generation:

- `LLM_BACKEND`: `mock`, `ollama`, or `transformers`.
- `LLM_MODEL`: backend-specific model identifier.
- `LLM_MAX_INPUT_TOKENS`: prompt token budget (default `4096`).
- `LLM_MAX_NEW_TOKENS`: maximum tokens to sample (default `256`).
- `LLM_TEMPERATURE`: sampling temperature (default `0.2`).
- `LLM_TOP_P`: nucleus sampling cutoff (default `0.95`).

Example configuration for a local Ollama server:

```
OLLAMA_HOST=http://localhost:11434
LLM_BACKEND=ollama
LLM_MODEL=mistral:7b-instruct-q4_K_M
```

Note: for GPU backend see next steps.

## GPU setup (tested on RTX 3060)

1. Install the optional GPU dependencies:
   - Poetry: `poetry install --with gpu`
   - Pip: `pip install "transformers>=4.42" accelerate bitsandbytes sentencepiece safetensors tokenizers`
2. Confirm CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"` (should print `True`).
3. Suggested transformer models:
   - `microsoft/Phi-3-mini-4k-instruct` (fast 4-bit model for prototyping).
   - `mistralai/Mistral-7B-Instruct` (load with 4-bit quantization on 12 GB GPUs).

Set `LLM_BACKEND=transformers` in `.env` together with an `LLM_MODEL` from the list above. Leave `LLM_DEVICE` unset to auto-detect CUDA and keep `LLM_LOAD_IN_4BIT=true` for reduced VRAM usage.

Set `QDRANT_LOCATION` in `.env` for local vector storage:

```
QDRANT_LOCATION=./qdrant_local
# For unit tests set QDRANT_LOCATION=:memory:
```
