import torch
from torch.utils.data import Dataset
import os
import numpy as np
from typing import List, Optional, Tuple

class LibriSpeechDataset(Dataset):
    def __init__(self, root_path: str, extractor, n_speakers: Optional[int] = None, 
                 samples_per_speaker: int = 20, specific_speakers: Optional[List[str]] = None):
        """
        Initialize the LibriSpeech dataset.
        
        Args:
            root_path: Path to the LibriSpeech dataset
            extractor: Voice embedding extractor
            n_speakers: Number of speakers to use (if None, use all)
            samples_per_speaker: Number of samples per speaker
            specific_speakers: List of specific speaker IDs to use
        """
        self.root_path = root_path
        self.extractor = extractor
        self.samples_per_speaker = samples_per_speaker
        
        # Get all speaker directories
        speaker_dirs = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
        
        if specific_speakers is not None:
            speaker_dirs = [d for d in speaker_dirs if d in specific_speakers]
        elif n_speakers is not None:
            speaker_dirs = np.random.choice(speaker_dirs, n_speakers, replace=False)
        
        # Collect audio files and speaker IDs
        self.audio_files = []
        self.speaker_ids = []
        
        for speaker_dir in speaker_dirs:
            speaker_path = os.path.join(root_path, speaker_dir)
            audio_files = []
            
            # Walk through the speaker directory
            for root, _, files in os.walk(speaker_path):
                for file in files:
                    if file.endswith('.flac'):
                        audio_files.append(os.path.join(root, file))
            
            # Randomly select samples_per_speaker files
            if len(audio_files) > samples_per_speaker:
                audio_files = np.random.choice(audio_files, samples_per_speaker, replace=False)
            
            self.audio_files.extend(audio_files)
            self.speaker_ids.extend([speaker_dir] * len(audio_files))
        
        # Extract embeddings for all audio files
        self.embeddings = []
        for audio_file in self.audio_files:
            embedding = self.extractor.get_normalized_embedding(audio_file)
            if embedding is not None:
                self.embeddings.append(embedding)
            else:
                # Remove corresponding speaker ID if embedding extraction failed
                idx = self.audio_files.index(audio_file)
                self.speaker_ids.pop(idx)
                self.audio_files.pop(idx)
    
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.embeddings)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the embedding and speaker ID at the given index."""
        return self.embeddings[idx], torch.tensor(int(self.speaker_ids[idx]))
    
    def sample_task(self, n_support: int, n_query: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a task for meta-learning.
        
        Args:
            n_support: Number of support samples
            n_query: Number of query samples
            
        Returns:
            Tuple of (support_x, support_y, query_x, query_y)
        """
        # Randomly select a speaker
        speaker_id = np.random.choice(list(set(self.speaker_ids)))
        
        # Get indices for this speaker
        speaker_indices = [i for i, sid in enumerate(self.speaker_ids) if sid == speaker_id]
        
        if len(speaker_indices) < n_support + n_query:
            raise ValueError(f"Not enough samples for speaker {speaker_id}")
        
        # Randomly select support and query samples
        selected_indices = np.random.choice(speaker_indices, n_support + n_query, replace=False)
        support_indices = selected_indices[:n_support]
        query_indices = selected_indices[n_support:]
        
        # Get support set
        support_x = torch.stack([self.embeddings[i] for i in support_indices])
        support_y = support_x.clone()  # For voice conversion, target is same as input
        
        # Get query set
        query_x = torch.stack([self.embeddings[i] for i in query_indices])
        query_y = query_x.clone()  # For voice conversion, target is same as input
        
        return support_x, support_y, query_x, query_y