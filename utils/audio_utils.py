import torch
import numpy as np
import librosa
import soundfile as sf
from speechbrain.inference.vocoders import HIFIGAN
import torch.nn as nn
import torchaudio
from speechbrain.inference.vocoders import HIFIGAN
import pystoi  # Import the pystoi library
from pystoi import stoi  # Import the stoi function from pystoi
import matplotlib.pyplot as plt

def synthesize_speech(embedding, ref_audio_path=None):
    """
    Synthesize speech from an embedding using a pre-trained vocoder.
    If ref_audio_path is provided, use it as a reference for the voice style.
    
    Args:
        embedding: The voice embedding (can be numpy array or torch tensor)
        ref_audio_path: Optional path to reference audio file
    
    Returns:
        torch.Tensor: The synthesized waveform
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert embedding to tensor if it's a numpy array
    if isinstance(embedding, np.ndarray):
        embedding_tensor = torch.FloatTensor(embedding)
    else:
        embedding_tensor = embedding.float()
    
    # Add batch dimension if needed
    if len(embedding_tensor.shape) == 1:
        embedding_tensor = embedding_tensor.unsqueeze(0)
    
    # Convert embedding to mel spectrogram using the MLP
    model = EmbeddingToMelMLP()
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        mel_spec = model(embedding_tensor.to(device))
    
    # Use a pre-trained vocoder to convert mel spectrogram to waveform
    # This is a placeholder - you would need to implement or use a real vocoder
    waveform = torch.randn(1, mel_spec.shape[-1] * 256)  # Placeholder
    
    return waveform

class EmbeddingToMelMLP(nn.Module):
    def __init__(self, embedding_dim=256, mel_dim=80, hidden_dim=512, time_steps=100):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mel_dim = mel_dim
        self.hidden_dim = hidden_dim
        self.time_steps = time_steps
        
        # Define the network architecture
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, mel_dim * time_steps)
        )
        
    def forward(self, x):
        # Process the embedding
        x = self.net(x)
        
        # Reshape to mel spectrogram format
        x = x.view(-1, self.mel_dim, self.time_steps)
        
        return x

def synthesize_speech(embedding, ref_audio_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Định nghĩa model chuyển đổi embedding -> mel-spectrogram
    class EmbeddingToMelMLP(nn.Module):
        def __init__(self, embedding_dim=256, mel_dim=80, hidden_dim=512, time_steps=100):
            super().__init__()
            self.time_steps = time_steps
            self.base = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            self.time_distributed = nn.Linear(hidden_dim, mel_dim * time_steps)

        def forward(self, x):
            h = self.base(x)
            return self.time_distributed(h).view(-1, 80, self.time_steps)
    
    model = EmbeddingToMelMLP().to(device).eval()
    hifi_gan = HIFIGAN.from_hparams(
        source="speechbrain/tts-hifigan-ljspeech",
        savedir="tmp_vocoder"
    ).to(device).eval()

    # Convert embedding to tensor if it's a numpy array
    if isinstance(embedding, np.ndarray):
        embedding_tensor = torch.FloatTensor(embedding)
    else:
        embedding_tensor = embedding.float()
    # Add batch dimension if needed
    if len(embedding_tensor.shape) == 1:
        embedding_tensor = embedding_tensor.unsqueeze(0)
    embedding_tensor = embedding_tensor.to(device)
    
    # Mel-spectrogram generation and synthesis
    # 5. Tạo mel-spectrogram từ embedding
    with torch.no_grad():
        mel_output = model(embedding_tensor)

        # Chuẩn hóa mel-spectrogram theo yêu cầu của HiFi-GAN
        mel_output = (mel_output - mel_output.min()) / (mel_output.max() - mel_output.min())
        mel_output = mel_output * 8 - 4  # Scale về khoảng [-4, 4]

    # 6. Hiển thị mel-spectrogram
    mel_spec = mel_output.squeeze().cpu().numpy()
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title("Generated Mel-Spectrogram")
    plt.show()

    # 7. Chuyển mel-spectrogram sang âm thanh bằng HiFi-GAN
    with torch.no_grad():
        if len(mel_output.shape) == 2:
            mel_output = mel_output.unsqueeze(0)
        waveforms = hifi_gan.decode_batch(mel_output)
        audio_synth = waveforms.squeeze().cpu().numpy()

    # 8. Load audio gốc để so sánh (nếu có)
    reference_audio_path = "/content/drive/MyDrive/LibriSpeech/train-clean-100/103/1240/103-1240-0000.flac"  # Thay bằng đường dẫn audio gốc
    
    # STOI calculation if reference provided
    if ref_audio_path:
        audio_ref, fs = librosa.load(ref_audio_path, sr=22050)
        min_len = min(len(audio_ref), len(audio_synth))
        stoi_score = stoi(audio_ref[:min_len], audio_synth[:min_len], fs_sig=22050)
        return audio_synth, stoi_score
    
    return audio_synth, None