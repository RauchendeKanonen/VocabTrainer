# VocabTrainer

## Overview

**VocabTrainer** is a vocabulary training application with integrated text-to-speech support. It appears designed to help learners practice vocabulary with both text and spoken output, using a local TTS system called **Kugel**.

The repository contains the trainer itself, a TTS server/client setup, batch generation tools, sample vocabulary CSV datasets, stats files, and a screenshot.

## Top-level contents

- `trainer.py`
- `kugel_server.py`
- `tts_client.py`
- `tts_client.sh`
- `batch_tts.py`
- `batch_tts.sh`
- `crop_wave.py`
- `start_kugel_server.sh`
- `requirements.txt`
- multiple sample CSV vocabulary files
- paired `.stats.json` files
- `vocabtrainer.png`

## What this project appears to do

### Interactive training
`trainer.py` is likely the main application for presenting vocabulary items and tracking answers.

### Local speech output
`kugel_server.py` strongly suggests a local TTS service.

### TTS generation pipeline
`batch_tts.py`, `batch_tts.sh`, and `crop_wave.py` suggest generation and post-processing of audio snippets.

### Vocabulary datasets
The CSV files indicate that the trainer can work with themed vocabulary lists, including Croatian examples.

## Installation

```bash
git clone https://github.com/RauchendeKanonen/VocabTrainer.git
cd VocabTrainer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

Trainer:

```bash
python3 trainer.py
```

TTS backend:

```bash
python3 kugel_server.py
```

or

```bash
./start_kugel_server.sh
```

## Suggested workflow

```text
Create or edit vocabulary CSV
    ↓
Generate TTS clips
    ↓
Start local TTS server if needed
    ↓
Launch trainer
    ↓
Practice with spoken prompts and tracked progress
```

## What the README should document better

- CSV schema,
- how stats are generated,
- how new lessons are added,
- whether training works fully offline,
- how Kugel is installed or started.

## License

No visible license from the public top-level snapshot.
