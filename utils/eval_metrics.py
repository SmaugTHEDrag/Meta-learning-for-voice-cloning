import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from speechbrain.utils.metric_stats import EER
from typing import Union, Tuple, List, Optional

def calculate_eer(model, test_dataset, model_type: str, num_speakers: int = 50, 
                 num_samples_per_speaker: int = 20, **kwargs) -> float:
    """
    Calculate Equal Error Rate (EER) for speaker verification.
    
    Args:
        model: The trained model (MAML, Reptile, or Baseline)
        test_dataset: Dataset containing test samples
        model_type: Type of model ('maml', 'reptile', or 'baseline')
        num_speakers: Number of speakers to test
        num_samples_per_speaker: Number of samples per speaker
        **kwargs: Additional arguments for model-specific evaluation
    
    Returns:
        float: The calculated EER
    """
    model.eval()
    scores = []
    labels = []
    
    # Generate scores for all possible pairs
    for i in range(num_speakers):
        for j in range(num_samples_per_speaker):
            # Get anchor sample
            anchor_x, anchor_y = test_dataset[i * num_samples_per_speaker + j]
            
            for k in range(num_speakers):
                for l in range(num_samples_per_speaker):
                    # Get comparison sample
                    comp_x, comp_y = test_dataset[k * num_samples_per_speaker + l]
                    
                    # Skip same sample
                    if i == k and j == l:
                        continue
                    
                    # Generate embeddings
                    with torch.no_grad():
                        if model_type == 'maml':
                            anchor_emb = model.generate_embedding_after_finetune(
                                anchor_x, [anchor_x], test_dataset.extractor, **kwargs
                            )
                            comp_emb = model.generate_embedding_after_finetune(
                                comp_x, [comp_x], test_dataset.extractor, **kwargs
                            )
                        elif model_type == 'reptile':
                            anchor_emb = model.generate_embedding_after_finetune(
                                anchor_x, [anchor_x], test_dataset.extractor, **kwargs
                            )
                            comp_emb = model.generate_embedding_after_finetune(
                                comp_x, [comp_x], test_dataset.extractor, **kwargs
                            )
                        else:  # baseline
                            anchor_emb = model.generate_embedding_after_finetune(
                                anchor_x, [anchor_x], test_dataset.extractor, **kwargs
                            )
                            comp_emb = model.generate_embedding_after_finetune(
                                comp_x, [comp_x], test_dataset.extractor, **kwargs
                            )
                    
                    # Calculate cosine similarity
                    similarity = nn.CosineSimilarity()(anchor_emb, comp_emb)
                    scores.append(similarity.item())
                    
                    # Label: 1 if same speaker, 0 if different
                    label = 1 if i == k else 0
                    labels.append(label)
    
    # Convert to numpy arrays
    scores = np.array(scores)
    labels = np.array(labels)
    
    # Calculate EER
    thresholds = np.linspace(min(scores), max(scores), 1000)
    min_eer = 1.0
    
    for threshold in thresholds:
        # Calculate false acceptance and false rejection rates
        fa = np.mean((scores >= threshold) & (labels == 0))
        fr = np.mean((scores < threshold) & (labels == 1))
        
        # Calculate EER
        eer = (fa + fr) / 2
        
        if eer < min_eer:
            min_eer = eer
    
    # Visualize score distribution
    plt.figure(figsize=(10, 6))
    plt.hist(scores[labels == 1], bins=30, alpha=0.5, label="Positive (Same Speaker)")
    plt.hist(scores[labels == 0], bins=30, alpha=0.5, label="Negative (Different Speaker)")
    plt.axvline(x=min_eer, color='r', linestyle='--', label=f'Threshold: {min_eer:.3f}')
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title(f"Speaker Verification Score Distribution ({model_type.upper()}) - EER: {min_eer*100:.2f}%")
    plt.savefig(f'{model_type.lower()}_eer_distribution.png')
    plt.show()
    
    return min_eer