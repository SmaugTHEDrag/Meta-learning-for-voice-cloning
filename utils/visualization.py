import matplotlib.pyplot as plt

def plot_training_metrics(metrics, title_prefix):
    plt.figure(figsize=(15, 5))
    
    # Plot training loss
    plt.subplot(1, 3, 1)
    plt.plot(metrics['epoch'], metrics['meta_loss'])
    plt.title(f'{title_prefix} Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Plot validation metrics if available
    if 'val_mse' in metrics and len(metrics['val_mse']) > 0:
        epochs = metrics['epoch'][::5]
        
        plt.subplot(1, 3, 2)
        plt.plot(epochs, metrics['val_cosine_sim'])
        plt.title('Validation Cosine Similarity')
        plt.xlabel('Epoch')
        plt.ylabel('Cosine Similarity')
        plt.ylim(-1, 1)
        
        plt.subplot(1, 3, 3)
        plt.plot(epochs, metrics['val_speaker_sim'])
        plt.title('Validation Speaker Similarity')
        plt.xlabel('Epoch')
        plt.ylabel('Similarity')
        plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'{title_prefix.lower()}_training_metrics.png')
    plt.show()