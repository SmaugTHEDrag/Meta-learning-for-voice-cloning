import torch
import argparse
from data.preprocessing import EmbeddingExtractor
from data.dataset import LibriSpeechDataset
from models.voice_models import FullyConnectedNeuralNetwork
from models.reptile import Reptile
from utils.eval_metrics import calculate_eer

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
    parser.add_argument('--inner_lr', type=float, default=0.02, help='Inner learning rate')
    parser.add_argument('--meta_lr', type=float, default=0.002, help='Meta learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--n_support', type=int, default=10, help='Number of support samples')
    parser.add_argument('--n_query', type=int, default=10, help='Number of query samples')
    parser.add_argument('--n_tasks', type=int, default=64, help='Number of tasks per epoch')
    parser.add_argument('--eval_interval', type=int, default=5, help='Evaluation interval')
    parser.add_argument('--inner_steps', type=int, default=5, help='Number of inner steps')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Initialize embedding extractor
    extractor = EmbeddingExtractor(encoder_type='Wav2Vec2')
    
    # Create datasets
    train_dataset, test_dataset = create_datasets(extractor)
    
    # Initialize model
    model = FullyConnectedNeuralNetwork()
    
    # Initialize Reptile trainer
    reptile = Reptile(
        model=model,
        inner_lr=args.inner_lr,
        meta_lr=args.meta_lr,
        weight_decay=args.weight_decay
    )
    
    # Train the model
    reptile.meta_train(
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        n_epochs=args.n_epochs,
        n_support=args.n_support,
        n_query=args.n_query,
        n_tasks=args.n_tasks,
        eval_interval=args.eval_interval,
        inner_steps=args.inner_steps
    )
    
    # Evaluate on test set
    eer = calculate_eer(
        model=reptile,
        test_dataset=test_dataset,
        model_type='reptile'
    )
    print(f'Test EER: {eer:.4f}')

if __name__ == '__main__':
    main()
