import numpy as np
import torch
from data.preprocessing import EmbeddingExtractor
from models import MAML, Reptile, BaselineTrainer
from models.voice_models import FullyConnectedNeuralNetwork
from utils.audio_utils import synthesize_speech

def generate_voice(model, model_type, target_audio, reference_audios, extractor):
    if model_type == "baseline":
        embedding = model.generate_embedding_after_finetune(
            target_audio, reference_audios, extractor
        )
    elif model_type == "maml":
        embedding = model.generate_embedding_after_finetune(
            target_audio, reference_audios, extractor
        )
    elif model_type == "reptile":
        embedding = model.generate_embedding_after_finetune(
            target_audio, reference_audios, extractor
        )
    
    np.save(f"{model_type}_embedding.npy", embedding)
    return embedding

def main():
    # Load model based on type
    model_type = "maml"  # or "baseline" or "reptile"
    model = load_model(model_type)
    
    extractor = EmbeddingExtractor()
    target_audio = "path/to/target.wav"
    reference_audios = ["ref1.wav", "ref2.wav", ...]
    
    # Generate embedding
    embedding = generate_voice(model, model_type, target_audio, reference_audios, extractor)
    
    # Synthesize speech
    audio, stoi_score = synthesize_speech(embedding, target_audio)
    print(f"STOI Score: {stoi_score:.4f}")

def load_model(model_type):
    # Load pretrained model based on type
    ...

if __name__ == "__main__":
    main()