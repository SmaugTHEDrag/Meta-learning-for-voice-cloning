import numpy as np
import torch
import librosa
from resemblyzer import VoiceEncoder, preprocess_wav
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor, HubertModel, Wav2Vec2Processor
import torchaudio
from speechbrain.inference import SpeakerRecognition
from config import Config
import torch.nn as nn

class EmbeddingNormalizer(nn.Module):
    def __init__(self, target_dim=256):
        super().__init__()
        self.target_dim = target_dim
        
    def forward(self, embedding, encoder_type):
        if encoder_type == 'Wav2Vec2':
            # Wav2Vec2 embeddings are already normalized
            return embedding
        else:
            # Normalize other embeddings
            norm = torch.norm(embedding, p=2, dim=-1, keepdim=True)
            return embedding / (norm + 1e-8)

class EmbeddingExtractor:
    def __init__(self, encoder_type='VoiceEncoder', target_dim=256):
        self.encoder_type = encoder_type
        self.target_dim = target_dim
        self.normalizer = EmbeddingNormalizer(target_dim)
        
        if encoder_type == 'Wav2Vec2':
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
            self.model.eval()
            
    def extract_embedding(self, wav_path):
        # Load audio
        waveform, sample_rate = torchaudio.load(wav_path)
        
        if self.encoder_type == 'Wav2Vec2':
            # Process audio for Wav2Vec2
            inputs = self.processor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Get the last hidden state
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
        else:
            # Default voice encoder processing
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Extract features (simplified version)
            mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=1024,
                hop_length=512,
                n_mels=80
            )(waveform)
            
            # Convert to log scale
            mel_spec = torch.log(mel_spec + 1e-8)
            
            # Use mean pooling as a simple embedding
            embeddings = mel_spec.mean(dim=-1).squeeze()
            
        return embeddings
    
    def get_normalized_embedding(self, wav_path):
        embedding = self.extract_embedding(wav_path)
        return self.normalizer(embedding, self.encoder_type)