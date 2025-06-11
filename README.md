# Voice Embedding Meta-Learning

This project implements meta-learning approaches (MAML, Reptile) for voice embedding tasks.

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your dataset in the following structure:
```
dataset/
    speaker1/
        audio1.wav
        audio2.wav
        ...
    speaker2/
        audio1.wav
        audio2.wav
        ...
    ...
```

2. Run the training:
```bash
python train.py --dataset_path path/to/dataset --model_type maml  # or reptile
```

## Features

- Multiple voice encoders support (VoiceEncoder, Wav2Vec2, HuBERT)
- Meta-learning implementations (MAML, Reptile)
- Baseline model for comparison
- EER (Equal Error Rate) evaluation
- Embedding normalization and extraction utilities

## Requirements

- Python 3.7+
- PyTorch 1.9.0+
- CUDA (recommended for faster training) 