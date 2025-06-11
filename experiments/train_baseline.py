import torch
import argparse
from data.preprocessing import EmbeddingExtractor
from data.dataset import LibriSpeechDataset
from models.voice_models import FullyConnectedNeuralNetwork
from models.baseline import BaselineTrainer
from utils.eval_metrics import calculate_eer
from utils.visualization import plot_training_metrics
from config import Config

def create_datasets(extractor):
    """Create train and test datasets."""
    # Create training dataset
    train_dataset = LibriSpeechDataset(
        root_path='path/to/librispeech/train',
        extractor=extractor,
        n_speakers=100,
        samples_per_speaker=20
    )
    
    # Create test dataset
    test_dataset = LibriSpeechDataset(
        root_path='path/to/librispeech/test',
        extractor=extractor,
        n_speakers=30,
        samples_per_speaker=10
    )
    
    return train_dataset, test_dataset

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--eval_interval', type=int, default=5, help='Evaluation interval')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Initialize embedding extractor
    extractor = EmbeddingExtractor(encoder_type='Wav2Vec2')
    
    # Create datasets
    train_dataset, test_dataset = create_datasets(extractor)
    
    # Initialize model
    model = FullyConnectedNeuralNetwork()
    
    # Initialize trainer
    trainer = BaselineTrainer(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Train the model
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval
    )
    
    # Evaluate on test set
    eer = calculate_eer(
        model=trainer,
        test_dataset=test_dataset,
        model_type='baseline'
    )
    print(f'Test EER: {eer:.4f}')

    # Visualization
    metrics = trainer.metrics
    plot_training_metrics(metrics, "Baseline")
    
    # Evaluation and saving
    ...

if __name__ == '__main__':
    main()