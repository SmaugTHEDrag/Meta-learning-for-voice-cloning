
import torch
import torch.nn as nn
import numpy as np
import librosa
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor, HubertModel
import torchaudio
from speechbrain.inference import SpeakerRecognition

# Class to normalize embedding dimensions using linear layers for different encoders
class EmbeddingNormalizer(nn.Module):
    def __init__(self, target_dim=256):
        super().__init__() # Initialize parent nn.Module
        self.target_dim = target_dim # Store target dimension
        # Define linear layers to project embeddings from different encoders to the target_dim
        self.ge2e_proj = nn.Linear(192, target_dim) # GE2E (192) -> target_dim
        self.voiceencoder_proj = nn.Linear(256, target_dim) # VoiceEncoder (256) -> target_dim
        self.wav2vec2_proj = nn.Linear(768, target_dim) # Wav2Vec2 (768) -> target_dim
        self.hubert_proj = nn.Linear(768, target_dim) # HuBERT (768) -> target_dim

    # Forward pass: converts NumPy to Tensor if needed and applies dynamic linear projection
    def forward(self, embedding, encoder_type):
        embedding = torch.from_numpy(embedding).float() if isinstance(embedding, np.ndarray) else embedding # Ensure input is a float tensor
        return getattr(self, f"{encoder_type.lower()}_proj")(embedding)  # HARD: Dynamically get and call the correct projection layer

# Class to extract and normalize speaker embeddings from audio files using various models
class EmbeddingExtractor:
    def __init__(self, encoder_type='VoiceEncoder', target_dim=256):
        self.encoder_type = encoder_type # Store encoder type
        self.target_dim = target_dim # Store target dimension
        self.normalizer = EmbeddingNormalizer(target_dim) # Initialize the normalizer
        # Initialize the specified pre-trained speaker encoder model
        if encoder_type == 'VoiceEncoder': self.encoder = VoiceEncoder() # Resemblyzer encoder
        elif encoder_type == 'GE2E': self.encoder = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp") # SpeechBrain GE2E
        elif encoder_type == 'Wav2Vec2': # Hugging Face Wav2Vec2
             self.processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h") # Wav2Vec2 audio processor
             self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h") # Wav2Vec2 model
        elif encoder_type == 'HuBERT': # Hugging Face HuBERT
             self.processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960") # HuBERT audio processor
             self.encoder = HubertModel.from_pretrained("facebook/hubert-base-ls960") # HuBERT model
        else: raise ValueError(f"Unsupported encoder: {encoder_type}") # Error for unknown type

    # Method to extract the raw (unnormalized) embedding from an audio file
    def extract_embedding(self, wav_path):
        """Extract original embedding (no normalization)"""
        try:
            if self.encoder_type == 'VoiceEncoder': # Process for VoiceEncoder
                wav = preprocess_wav(wav_path) # HARD: Model-specific preprocessing (load, resample, convert)
                return self.encoder.embed_utterance(wav) # Compute embedding
            elif self.encoder_type == 'GE2E': # Process for GE2E
                signal, _ = torchaudio.load(wav_path); signal = signal.mean(dim=0) if signal.dim() > 1 else signal # HARD: Load, handle stereo
                embedding = self.encoder.encode_batch(signal.unsqueeze(0)) # HARD: Add batch dim and encode
                return embedding.squeeze().detach().cpu().numpy() # HARD: Squeeze, detach, move to CPU, convert to NumPy
            elif self.encoder_type == 'Wav2Vec2': # Process for Wav2Vec2
                speech, _ = librosa.load(wav_path, sr=16000) # HARD: Load and resample to 16kHz
                inputs = self.processor(speech, return_tensors="pt", sampling_rate=16000, padding=True) # HARD: Process audio for model input
                with torch.no_grad(): outputs = self.encoder(**inputs) # HARD: Run model inference without gradients
                return outputs.last_hidden_state.mean(dim=1).squeeze().numpy() # HARD: Average hidden states for utterance embedding
            elif self.encoder_type == 'HuBERT': # Process for HuBERT (similar to Wav2Vec2)
                speech, _ = librosa.load(wav_path, sr=16000) # Load and resample
                inputs = self.processor(speech, return_tensors="pt", sampling_rate=16000, padding=True) # Process audio for model input
                with torch.no_grad(): outputs = self.encoder(**inputs) # Run model inference without gradients
                return outputs.last_hidden_state.mean(dim=1).squeeze().numpy() # Average hidden states for utterance embedding
        except Exception as e: # Catch errors during extraction
            print(f"Error extracting embedding: {e}"); return None # Print error and return None

    # Method to extract the raw embedding and then apply normalization
    def get_normalized_embedding(self, wav_path):
        raw_embed = self.extract_embedding(wav_path) # Get raw embedding
        if raw_embed is None: return None # Return None if raw extraction failed
        normalized_embed = self.normalizer(raw_embed, self.encoder_type) # HARD: Normalize the raw embedding using the normalizer instance
        return normalized_embed.detach().cpu().numpy() # Detach, move to CPU, convert to NumPy

# Initialize extractor with target_dim=256
# Initialize embedding extractor
print("Initializing embedding extractor...")
extractor = EmbeddingExtractor(encoder_type='VoiceEncoder', target_dim=256)  # Change to GE2E, Wav2Vec2 and HuBERT

"""# 5. LibriSpeech Dataset Class"""

import random
from pathlib import Path
import torch
from torch.utils.data import Dataset

# Define a custom Dataset class for the LibriSpeech dataset
class LibriSpeechDataset(Dataset):
    def __init__(self, root_path, extractor, n_speakers=None, samples_per_speaker=20, specific_speakers=None):
        """
        Initializes the LibriSpeech dataset

        Args:
            root_path (str or Path): Path to the LibriSpeech root directory.
            extractor: An object used to extract embeddings from audio files.
                       This object should have a method `get_normalized_embedding(wav_path)`.
            n_speakers (int, optional): Number of speakers to load if specific_speakers is None.
            samples_per_speaker (int, optional): Maximum number of audio samples per speaker to load. Defaults to 20.
            specific_speakers (list, optional): A list of specific speaker directory paths to load.
                                                 If provided, n_speakers is ignored.
        """
        # Convert root_path to a Path object for easier file system operations
        self.root_path = Path(root_path)
        # Check if the provided root_path exists
        if not self.root_path.exists():
            raise ValueError(f"Path {root_path} does not exist")

        # Store the embedding extractor object
        self.extractor = extractor

        # Get the list of speaker directories
        if specific_speakers is not None:
            # Use the provided list of specific speaker paths
            speakers = specific_speakers
        else:
            # Find all directories within the root path and sort them (assumes they are speaker IDs)
            speakers = sorted([d for d in self.root_path.iterdir() if d.is_dir()])
            # Raise an error if no speaker directories are found
            if len(speakers) == 0:
                raise ValueError(f"No speaker directories found in {root_path}")

            # If n_speakers is specified, take only the first n_speakers
            if n_speakers is not None:
                speakers = speakers[:n_speakers]

        # Print the number of speakers being loaded
        print(f"Loading {len(speakers)} speakers...")

        # Initialize lists to store extracted embeddings and their corresponding speaker IDs
        self.embeddings = []
        self.speaker_ids = []

        # Process each speaker directory
        for speaker_dir in speakers:
            try:
                # Get the speaker ID from the directory name
                speaker_id = speaker_dir.name
                # Find chapter directories within the speaker directory
                chapter_dirs = [d for d in speaker_dir.iterdir() if d.is_dir()]
                # Initialize a list to hold paths to audio files for this speaker
                wav_files = []

                # Collect .flac audio files from each chapter directory
                for chapter_dir in chapter_dirs:
                    # Use glob to find all .flac files in the chapter directory
                    wav_files.extend(list(chapter_dir.glob('*.flac')))

                # Limit the number of audio files per speaker to samples_per_speaker
                wav_files = wav_files[:samples_per_speaker]

                # Warning if no .flac files were found for this speaker
                if len(wav_files) == 0:
                    print(f"Warning: No .flac files found for speaker {speaker_id}")
                    continue # Skip to the next speaker

                # Print the number of files being processed for the current speaker
                print(f"Processing speaker {speaker_id}: {len(wav_files)} files")

                # Extract embeddings for each audio file
                for wav_path in wav_files:
                    try:
                        # Use the provided extractor to get the normalized speaker embedding
                        embedding = self.extractor.get_normalized_embedding(str(wav_path))
                        # If embedding extraction was successful (not None)
                        if embedding is not None:
                            # Convert the NumPy embedding to a PyTorch FloatTensor and append
                            self.embeddings.append(torch.FloatTensor(embedding))
                            # Append the corresponding speaker ID
                            self.speaker_ids.append(speaker_id)
                        else:
                             print(f"Warning: Could not extract embedding for {wav_path}")
                    except Exception as e:
                        # Catch and print errors during embedding extraction for a specific file
                        print(f"Error processing {wav_path}: {str(e)}")
                        continue # Skip to the next file for this speaker

            except Exception as e:
                # Catch and print errors during processing of a speaker directory
                print(f"Error processing speaker directory {speaker_dir}: {str(e)}")
                continue # Skip to the next speaker directory

        # Raise an error if no valid embeddings were extracted from the entire dataset
        if len(self.embeddings) == 0:
            raise ValueError("No valid embeddings extracted from the dataset")

        # Print the total number of samples (embeddings) loaded
        print(f"Total samples: {len(self.embeddings)}")

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.embeddings)

    def __getitem__(self, idx):
        # Return the embedding and speaker ID at the given index
        return self.embeddings[idx], self.speaker_ids[idx]

    def sample_task(self, n_support, n_query):
        """
        Creates a task for meta-learning. A task consists of a support set
        (used for quick adaptation) and a query set (used for evaluation).

        Args:
            n_support (int): Number of samples for the support set. Must be positive.
            n_query (int): Number of samples for the query set. Must be positive.

        Returns:
            tuple: (support_x, support_y, query_x, query_y)
                   support_x (Tensor): Embeddings for the support set.
                   support_y (Tensor): Target embeddings for the support set (clone of support_x).
                   query_x (Tensor): Embeddings for the query set.
                   query_y (Tensor): Target embeddings for the query set (clone of query_x).
        """
        # Validate that the number of support and query samples requested are positive
        if n_support <= 0 or n_query <= 0:
            raise ValueError("n_support and n_query must be positive")

        # Get a list of all unique speaker IDs present in the dataset
        unique_speakers = list(set(self.speaker_ids))
        # Raise an error if no speakers are available in the dataset
        if len(unique_speakers) == 0:
            raise ValueError("No speakers available")

        # Randomly select one speaker to form the task
        speaker = random.choice(unique_speakers)
        # Find the indices of all samples (embeddings) belonging to the selected speaker
        speaker_indices = [i for i, sid in enumerate(self.speaker_ids) if sid == speaker]

        # Check if the selected speaker has enough samples for the requested support and query sets
        if len(speaker_indices) < (n_support + n_query):
            # Print a warning if there aren't enough samples
            print(f"Warning: Speaker {speaker} has only {len(speaker_indices)} samples, "
                  f"less than requested {n_support + n_query}")

        # Adjust the total number of samples to take for this task
        # It's the minimum of the requested total (n_support + n_query) and the available samples for the speaker
        total_samples = min(n_support + n_query, len(speaker_indices))

        # Adjust the number of support samples
        # Ensure it's not more than half of the total available samples (a common split)
        n_support_actual = min(n_support, total_samples // 2)
        # Adjust the number of query samples
        # Ensure it's the remaining samples after taking the actual support samples
        n_query_actual = min(n_query, total_samples - n_support_actual)

        # Raise an error if, after adjustments, either support or query set would be empty
        if n_support_actual == 0 or n_query_actual == 0:
             raise ValueError(f"Not enough samples for speaker {speaker} to create support ({n_support}) and query ({n_query}) sets. Available: {len(speaker_indices)}. Actual: support={n_support_actual}, query={n_query_actual}")


        # Randomly select indices from the speaker's available samples
        selected_indices = random.sample(speaker_indices, total_samples)

        # Split the selected indices into support and query sets
        support_indices = selected_indices[:n_support_actual]
        query_indices = selected_indices[n_support_actual : n_support_actual + n_query_actual] # Use actual counts

        # Create tensors by stacking the embeddings at the selected indices
        support_x = torch.stack([self.embeddings[i] for i in support_indices])
        query_x = torch.stack([self.embeddings[i] for i in query_indices])

        # Return the support inputs, support targets (clone of support inputs),
        # query inputs, and query targets (clone of query inputs).
        # Cloning is used because the task is likely an autoencoding-style reconstruction.
        return support_x, support_x.clone(), query_x, query_x.clone()

# Set path
librispeech_path = "/content/drive/MyDrive/LibriSpeech/train-clean-100"

# Get all available speakers
all_speakers = sorted([d for d in Path(librispeech_path).iterdir() if d.is_dir()])
print(f"Total speakers available: {len(all_speakers)}")

# Shuffle speakers to ensure random splitting
random.shuffle(all_speakers)

# Select a subset large enough for the experiment
max_speakers = 50  # Adjust this number based on computational capabilities
selected_speakers = all_speakers[:max_speakers] if len(all_speakers) > max_speakers else all_speakers

# Calculate the number for train and test (80% train, 20% test)
num_train = int(0.8 * len(selected_speakers))
train_speakers = selected_speakers[:num_train]
test_speakers = selected_speakers[num_train:]

# Create dataset for training with specific speakers
print("\nPreparing training dataset...")
train_dataset = LibriSpeechDataset(
    librispeech_path,
    extractor,
    specific_speakers=train_speakers,
    samples_per_speaker=20
)

# Tạo dataset cho testing với specific speakers khác
print("\nPreparing test dataset...")
test_dataset = LibriSpeechDataset(
      librispeech_path,
      extractor,
      specific_speakers=test_speakers,
      samples_per_speaker=20
)

"""# Fully Connected Neural Network"""

class FullyConnectedNeuralNetwork(nn.Module):
    def __init__(self, input_dim=256, hidden_dim1=256, hidden_dim2=128):
          """
          - input_dim (int): Input size (default is 256, corresponding to voice embedding).
          - hidden_dim1 (int): Size of the first hidden layer (default is 256).
          - hidden_dim2 (int): Size of the second hidden layer (default is 128).
          """
          super().__init__()
          self.net = nn.Sequential(
              nn.Linear(input_dim, hidden_dim1),     # First fully connected
              nn.ReLU(),                             # ReLU activation
              nn.Linear(hidden_dim1, hidden_dim2),   # Second fully connected
              nn.ReLU(),                             # ReLU activation
              nn.Linear(hidden_dim2, input_dim)      # Final fully connected layer, output has the same size as input
          )

    def forward(self, x, params=None): # Function to pass data through the network.
          """
          - x (Tensor): Input to the model (voice embedding vector).
          - params (dict, optional): Custom weights when performing meta-learning.
          - out (Tensor): Output of the model.
          """
          if params is None:
              return self.net(x)                    # If no custom parameters, run through the network normally.
          x = x.view(x.size(0), -1)  # Flatten input if necessary

          # Pass data through each layer with custom weights
          h = F.linear(x, params['net.0.weight'], params['net.0.bias'])
          h = F.relu(h)
          h = F.linear(h, params['net.2.weight'], params['net.2.bias'])
          h = F.relu(h)
          out = F.linear(h, params['net.4.weight'], params['net.4.bias'])
          return out

"""# Autoencoder Baseline (No meta-learning)"""

# Initialize baseline model with the same architecture as MAML model
baseline_model = FullyConnectedNeuralNetwork(input_dim=256, hidden_dim1=256, hidden_dim2=128).to(device)

class BaselineTrainer:
    def __init__(self, model, lr=0.001, weight_decay=0.01):
        """
        Initialize the baseline trainer

        Args:
            model: Neural network model (e.g., FullyConnectedNeuralNetwork). This is the core model we will train.
            lr: Learning rate for the main Adam optimizer. Controls the step size during parameter updates.
            weight_decay: L2 regularization parameter for the optimizer. Helps prevent overfitting by penalizing large weights.
        """
        # Store the provided neural network model instance
        self.model = model
        # Define the main optimizer (Adam) to update the model's parameters.
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # Define the loss function (Mean Squared Error).
        self.criterion = nn.MSELoss()
        # Determine the device (CPU or GPU) where the model is currently located.
        self.device = next(model.parameters()).device
        # Set an inner learning rate. This specific learning rate will be used
        self.inner_lr = 0.02  # Same as MAML inner learning rate for fair comparison

    def train(self, train_dataset, val_dataset=None, n_epochs=100, batch_size=64, eval_interval=5):
        """
        Trains the baseline model on the entire training dataset.

        Args:
            train_dataset: The dataset object containing training data (embeddings and speaker IDs).
            val_dataset: Optional dataset object for validation. If provided, the model will be evaluated on it periodically.
            n_epochs: The total number of complete passes through the training dataset.
            batch_size: The number of samples processed in a single forward/backward pass.
            eval_interval: The number of epochs between each evaluation on the validation dataset.
        """
        # This section prepares the data for standard training.
        # It extracts all embeddings from the dataset and uses them as both inputs and targets (autoencoder setup).
        train_embeddings = []
        train_targets = []

        print("Preparing training data...")
        # Iterate through the training dataset to collect all samples.
        for i in range(len(train_dataset)):
            # Get the embedding and speaker ID for the current sample.
            embedding, _ = train_dataset[i]
            # Add the embedding to the list of inputs.
            train_embeddings.append(embedding)
            # Create the target by cloning the embedding itself (autoencoder objective).
            train_targets.append(embedding.clone())

        # Stack the list of embeddings and targets into single tensors.
        train_embeddings = torch.stack(train_embeddings)
        train_targets = torch.stack(train_targets)

        # Create a DataLoader to manage batching and shuffling of the training data.
        # It wraps the zipped inputs and targets.
        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            list(zip(train_embeddings, train_targets)),
            batch_size=batch_size,
            shuffle=True # Shuffle the data each epoch to improve training robustness.
        )

        # Dictionary to store training and validation metrics over time.
        metrics_history = {
            'epoch': [], # List to store epoch numbers.
            'train_loss': [], # List to store training loss for each epoch.
            'val_mse': [], # List to store validation MSE loss.
            'val_cosine_sim': [],  # List to store validation cosine similarity.
            'val_speaker_sim': [] # List to store validation speaker similarity.
        }

        print(f"Starting baseline training with {len(train_loader)} batches per epoch...")

        # Main training loop iterates through the specified number of epochs.
        for epoch in range(n_epochs):
            # Set the model to training mode. This enables features like dropout.
            self.model.train()
            # Initialize loss for the current epoch.
            epoch_loss = 0.0

            # Iterate through batches provided by the DataLoader.
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # Move input and target tensors to the correct device (CPU/GPU).
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Zero the gradients accumulated from the previous training step.
                self.optimizer.zero_grad()
                # Perform a forward pass: feed inputs through the model to get predictions.
                outputs = self.model(inputs)
                # Calculate the loss between the model's outputs and the targets.
                loss = self.criterion(outputs, targets)
                # Perform backpropagation: compute gradients of the loss with respect to model parameters.
                loss.backward()
                # Update model parameters using the optimizer and calculated gradients.
                self.optimizer.step()

                # Accumulate the loss for the current batch.
                epoch_loss += loss.item()

                # Print the loss periodically (every 10 batches) to monitor training progress.
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{n_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

            # Calculate the average training loss for the current epoch.
            avg_loss = epoch_loss / len(train_loader)
            # Record the epoch number and average training loss.
            metrics_history['epoch'].append(epoch + 1)
            metrics_history['train_loss'].append(avg_loss)

            # Perform validation evaluation at specified intervals or at the end of training.
            if ((epoch + 1) % eval_interval == 0) or (epoch == n_epochs - 1):
                print(f"Epoch {epoch+1}/{n_epochs}: Train Loss = {avg_loss:.4f}")

                # If a validation dataset is provided, evaluate the model on it.
                if val_dataset is not None:
                    # Call the evaluation method on the validation dataset.
                    # n_tasks=10 means it will sample 10 tasks (speakers) for evaluation.
                    val_metrics = self.evaluate_on_dataset(val_dataset, n_tasks=10)
                    # Store the validation metrics.
                    metrics_history['val_mse'].append(val_metrics['mse_loss'])
                    metrics_history['val_cosine_sim'].append(val_metrics['cosine_similarity'])
                    metrics_history['val_speaker_sim'].append(val_metrics['speaker_similarity'])

                    # Print the validation metrics.
                    print(f"Validation Metrics - MSE: {val_metrics['mse_loss']:.4f}, Cosine Similarity: {val_metrics['cosine_similarity']:.4f}, Speaker Similarity: {val_metrics['speaker_similarity']:.4f}")

        # Return the history of recorded metrics.
        return metrics_history

    def fine_tune(self, support_x, support_y, n_steps=50):
        """
        Simulates fine-tuning the model on a small support set for a few gradient steps.
        This mimics the inner loop of a MAML-like process but uses the standard model's
        parameters as a starting point.

        Args:
            support_x: Input embeddings for the support set.
            support_y: Target embeddings for the support set (usually a clone of support_x for this task).
            n_steps: The number of gradient descent steps to take during fine-tuning.
        """
        # Create a deep copy of the model's current parameters.
        # Fine-tuning is performed on this copy, leaving the original model parameters untouched.
        fine_tuned_params = {name: param.clone() for name, param in self.model.named_parameters()}

        # Move the support data to the correct device.
        support_x = support_x.to(self.device)
        support_y = support_y.to(self.device)

        # Perform the fine-tuning steps.
        for _ in range(n_steps):
            # Perform a forward pass using the support data and the *copied* parameters.
            pred = self.model(support_x, fine_tuned_params)
            # Calculate the loss on the support set using the predictions from the copied parameters.
            loss = F.mse_loss(pred, support_y)

            # Compute gradients of the loss with respect to the *copied* parameters.
            # create_graph=False because we only need the gradients for this inner update, not for an outer loop.
            # allow_unused=True handles cases where some parameters might not receive gradients in a specific step.
            grads = torch.autograd.grad(
                loss,
                fine_tuned_params.values(),
                create_graph=False,
                allow_unused=True
            )

            # Update the copied parameters manually using gradient descent with the inner learning rate.
            # This is a manual parameter update, not using the self.optimizer.
            fine_tuned_params = {
                name: param - self.inner_lr * grad
                for ((name, param), grad) in zip(fine_tuned_params.items(), grads)
            }

        # Return the parameters after fine-tuning. These adapted parameters are specialized for the specific task/speaker.
        return fine_tuned_params

    def evaluate(self, query_x, query_y, adapted_params=None):
        """
        Evaluates the model's performance on query data, optionally using adapted parameters.

        Args:
            query_x: Input embeddings for the query set.
            query_y: Target embeddings for the query set (usually a clone of query_x).
            adapted_params: Optional dictionary of parameters to use for the forward pass (e.g., from fine-tuning).
                            If None, the model's standard parameters are used.
        """
        # Disable gradient computation for evaluation to save memory and computation.
        with torch.no_grad():
            # Perform a forward pass using either the adapted parameters or the standard model parameters.
            if adapted_params is None:
                pred = self.model(query_x) # Use standard model parameters
            else:
                pred = self.model(query_x, adapted_params) # Use fine-tuned parameters

            # Calculate Mean Squared Error loss.
            mse_loss = F.mse_loss(pred, query_y).item()
            # Calculate Cosine Similarity between predicted and target embeddings.
            # mean() is used to get an average similarity if there are multiple samples in the query set.
            cos_sim = F.cosine_similarity(pred, query_y).mean().item()
            # Calculate Speaker Similarity by normalizing Cosine Similarity to a [0, 1] range.
            speaker_sim = (cos_sim + 1) / 2

            # Return a dictionary containing the calculated metrics.
            return {
                'mse_loss': mse_loss,
                'cosine_similarity': cos_sim,
                'speaker_similarity': speaker_sim
            }

    def evaluate_on_dataset(self, dataset, n_tasks=10, n_support=10, n_query=10):
        """
        Evaluates the model's performance on a dataset by sampling multiple tasks (speakers)
        and averaging the evaluation metrics. This method evaluates using the standard
        model parameters, NOT performing fine-tuning before evaluating each task.

        Args:
            dataset: The dataset object to evaluate on (e.g., validation or test set).
            n_tasks: The number of distinct tasks (speakers) to sample for evaluation.
            n_support: The number of support samples to request when sampling a task
                       (though these are not used for fine-tuning in this method).
            n_query: The number of query samples to request when sampling a task.
        """
        # Set the model to evaluation mode. Disables training-specific layers like dropout.
        self.model.eval()
        # List to store the metrics calculated for each sampled task.
        metrics_list = []

        # Disable gradient computation for evaluation.
        with torch.no_grad():
            # Loop to sample and evaluate multiple tasks.
            for i in range(n_tasks):
                try:
                    # Sample a task (speaker) from the dataset.
                    # We only care about the query set for this evaluation method.
                    _, _, query_x, query_y = dataset.sample_task(n_support, n_query)

                    # Move the query data to the correct device.
                    query_x = query_x.to(self.device)
                    query_y = query_y.to(self.device)

                    # Calculate metrics for the current task using the standard model parameters (no adapted_params).
                    metrics = self.evaluate(query_x, query_y)
                    # Add the metrics for this task to the list.
                    metrics_list.append(metrics)
                except Exception as e:
                    # Catch any errors during task evaluation and print a warning.
                    print(f"Error in task evaluation {i+1}/{n_tasks}: {str(e)}")
                    continue # Skip to the next task

        # If no valid tasks were evaluated, return default "bad" metric values.
        if len(metrics_list) == 0:
            print("Warning: No valid tasks evaluated.")
            return {'mse_loss': float('inf'), 'cosine_similarity': -1, 'speaker_similarity': 0}

        # Calculate the average metrics across all successfully evaluated tasks.
        avg_metrics = {
            'mse_loss': np.mean([m['mse_loss'] for m in metrics_list]),
            'cosine_similarity': np.mean([m['cosine_similarity'] for m in metrics_list]),
            'speaker_similarity': np.mean([m['speaker_similarity'] for m in metrics_list])
        }

        # Return the average metrics.
        return avg_metrics

    def generate_embedding_after_finetune(self, target_audio_path, reference_audio_paths, extractor, n_steps=50):
        """
        Generates an embedding for a single target audio file after first
        fine-tuning the model on a set of reference audio files from the same speaker.
        This simulates applying the fine-tuned model to a new sample from an encountered speaker.

        Args:
            target_audio_path (str): The file path to the audio sample you want to embed.
            reference_audio_paths (list): A list of file paths to reference audio samples from the *same* speaker as the target.
            extractor: An instance of the voice embedding extractor (e.g., EmbeddingExtractor).
            n_steps (int): The number of fine-tuning steps to perform on the reference files.

        Returns:
            numpy.ndarray: The resulting embedding for the target audio file after fine-tuning.
        """
        # Use the extractor to get the normalized embedding for the target audio file.
        target_embedding = extractor.get_normalized_embedding(target_audio_path)
        # Raise an error if embedding extraction failed.
        if target_embedding is None:
            raise ValueError(f"Failed to extract embedding from target audio: {target_audio_path}")

        # Determine the device the model is on and move the target embedding to that device.
        # unsqueeze(0) adds a batch dimension (required by the model).
        device = next(self.model.parameters()).device
        target_embedding_tensor = torch.FloatTensor(target_embedding).unsqueeze(0).to(device)

        # Extract embeddings from the reference audio files. These will form the support set.
        support_embeddings = []
        for ref_path in reference_audio_paths:
            # It's crucial to use the *normalized* embedding extraction here
            # to ensure the input dimensions match the model's expectation after normalization.
            embedding = extractor.get_normalized_embedding(ref_path)
            if embedding is not None:
                support_embeddings.append(torch.FloatTensor(embedding))

        # Raise an error if no valid reference embeddings could be extracted.
        if not support_embeddings:
            raise ValueError("No valid reference embeddings found")

        # Stack the list of support embeddings into a single tensor and move to the device.
        support_x = torch.stack(support_embeddings).to(device)
        # Create targets for the support set - cloning the inputs for the autoencoder objective.
        support_y = support_x.clone()

        # Fine-tune the model parameters using the reference audio (support set).
        # This returns adapted parameters specific to this speaker.
        adapted_params = self.fine_tune(support_x, support_y, n_steps=n_steps)

        # Use the fine-tuned model parameters to generate the embedding for the target audio.
        with torch.no_grad(): # Disable gradient computation.
            fine_tuned_embedding = self.model(target_embedding_tensor, adapted_params)
            # Remove the batch dimension (squeeze(0)) and move the tensor back to CPU before converting to NumPy.
            return fine_tuned_embedding.squeeze(0).cpu().numpy()

baseline_trainer = BaselineTrainer(baseline_model, lr=0.001, weight_decay=0.01)

# Use the same validation dataset as MAML for fair comparison
val_dataset = train_dataset

# Train the baseline model
print("Training baseline model...")
baseline_metrics = baseline_trainer.train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    n_epochs=100,
    batch_size=64,
    eval_interval=5
)

# Plot training metrics
plt.figure(figsize=(15, 5))

# Plot 1: Training Loss
plt.subplot(1, 3, 1)
plt.plot(baseline_metrics['epoch'], baseline_metrics['train_loss'])
plt.title('Baseline Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plot validation metrics if available
if 'val_mse' in baseline_metrics and len(baseline_metrics['val_mse']) > 0:
    epochs = baseline_metrics['epoch'][::5]  # For eval_interval=5

    # Plot 2: Validation Cosine Similarity
    plt.subplot(1, 3, 2)
    plt.plot(epochs, baseline_metrics['val_cosine_sim'])
    plt.title('Validation Cosine Similarity')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.ylim(-1, 1)  # Cosine similarity range is [-1, 1]

    # Plot 3: Validation Speaker Similarity
    plt.subplot(1, 3, 3)
    plt.plot(epochs, baseline_metrics['val_speaker_sim'])
    plt.title('Validation Speaker Similarity')
    plt.xlabel('Epoch')
    plt.ylabel('Similarity')
    plt.ylim(0, 1)  # Speaker similarity range is [0, 1]

plt.tight_layout()
plt.savefig('baseline_training_metrics.png')
plt.show()

# Evaluate baseline model with fine-tuning
print("\nEvaluating baseline model...")
n_test_tasks = 20
baseline_test_metrics = []

for i in range(n_test_tasks):
    support_x, support_y, query_x, query_y = test_dataset.sample_task(15, 5)

    # Move to device
    support_x = support_x.to(device)
    support_y = support_y.to(device)
    query_x = query_x.to(device)
    query_y = query_y.to(device)

    # Evaluate before fine-tuning
    metrics_before = baseline_trainer.evaluate(query_x, query_y)

    # Fine-tune and evaluate
    adapted_params = baseline_trainer.fine_tune(support_x, support_y, n_steps=50)
    metrics_after = baseline_trainer.evaluate(query_x, query_y, adapted_params)

    baseline_test_metrics.append({
        'before': metrics_before,
        'after': metrics_after
    })

    if (i + 1) % 5 == 0:
        print(f"\nTest task {i+1}/{n_test_tasks}")
        print("Metrics before/after fine-tuning:")
        for metric in metrics_before.keys():
            avg_before = np.mean([m['before'][metric] for m in baseline_test_metrics[:i+1]])
            avg_after = np.mean([m['after'][metric] for m in baseline_test_metrics[:i+1]])
            print(f"{metric}: {avg_before:.4f} -> {avg_after:.4f}")

import torch
import numpy as np
from speechbrain.utils.metric_stats import EER
import matplotlib.pyplot as plt
import torch.nn.functional as F
def calculate_eer_baseline(test_dataset, baseline_trainer, num_speakers=30, num_samples_per_speaker=10):
    """
    Calculate Equal Error Rate using embeddings already in the dataset with BaselineTrainer.

    Args:
        test_dataset: The test dataset containing embeddings and speaker IDs
        baseline_trainer: The BaselineTrainer model
        num_speakers: Number of speakers to use for evaluation
        num_samples_per_speaker: Number of samples per speaker

    Returns:
        Equal Error Rate
    """
    # Get device from model
    device = next(baseline_trainer.model.parameters()).device

    # Select random speakers from test dataset
    all_test_speakers = list(set(test_dataset.speaker_ids))
    if len(all_test_speakers) < num_speakers:
        print(f"Warning: Only {len(all_test_speakers)} speakers available, using all of them")
        selected_speakers = all_test_speakers
    else:
        selected_speakers = np.random.choice(all_test_speakers, num_speakers, replace=False)

    positive_scores = []
    negative_scores = []

    for target_speaker in selected_speakers:
        # Get all embeddings of this speaker
        speaker_indices = [i for i, sid in enumerate(test_dataset.speaker_ids) if sid == target_speaker]

        # Ensure we have enough samples
        if len(speaker_indices) < num_samples_per_speaker:
            continue

        # Get actual embedding tensors for this speaker
        speaker_embeddings = [test_dataset.embeddings[i] for i in speaker_indices[:num_samples_per_speaker]]

        # Use 5 embeddings as support set and 1 as target
        indices = np.random.choice(len(speaker_embeddings), 6, replace=False)
        support_indices = indices[:5]
        target_index = indices[5]
        support_embeddings = [speaker_embeddings[i] for i in support_indices]
        target_embedding = speaker_embeddings[target_index]

        # Move to device
        support_x = torch.stack(support_embeddings).to(device)
        support_y = support_x.clone()  # We want the model to preserve embeddings
        target_embedding = target_embedding.to(device)

        # Fine-tune model on support set using BaselineTrainer
        adapted_params = baseline_trainer.fine_tune(support_x, support_y, n_steps=50)

        # Get model prediction for target embedding
        with torch.no_grad():
            fine_tuned_embedding = baseline_trainer.model(target_embedding.unsqueeze(0), adapted_params)
            fine_tuned_embedding = fine_tuned_embedding.squeeze(0)

        # Calculate cosine similarity for positive pair (same speaker)

        cos_sim_positive = F.cosine_similarity(
            target_embedding.unsqueeze(0),
            fine_tuned_embedding.unsqueeze(0)
        ).item()
        positive_scores.append(cos_sim_positive)

        # Create negative pair (different speaker)
        other_speaker = np.random.choice([s for s in selected_speakers if s != target_speaker])
        other_indices = [i for i, sid in enumerate(test_dataset.speaker_ids) if sid == other_speaker]

        if not other_indices:
            continue

        other_embedding = test_dataset.embeddings[np.random.choice(other_indices)].to(device)

        # Calculate cosine similarity for negative pair (different speaker)
        cos_sim_negative = F.cosine_similarity(
            target_embedding.unsqueeze(0),
            other_embedding.unsqueeze(0)
        ).item()
        negative_scores.append(cos_sim_negative)

    if not positive_scores or not negative_scores:
        raise ValueError("Not enough data to calculate EER")

    # Convert to tensors for EER calculation
    positive_scores_tensor = torch.tensor(positive_scores)
    negative_scores_tensor = torch.tensor(negative_scores)

    # Calculate EER
    eer_score, threshold = EER(positive_scores_tensor, negative_scores_tensor)

    # Visualize score distribution
    plt.figure(figsize=(10, 6))
    plt.hist(positive_scores, bins=30, alpha=0.5, label="Positive (Same Speaker)")
    plt.hist(negative_scores, bins=30, alpha=0.5, label="Negative (Different Speaker)")
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.3f}')
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title(f"Speaker Verification Score Distribution (Baseline) - EER: {eer_score*100:.2f}%")
    plt.savefig('baseline_eer_distribution.png')
    plt.show()

    return eer_score

import numpy as np
import torch
import random
import os

# Hàm cố định seed cho tất cả thư viện
def set_seed(seed=42):
    # Python
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Nếu dùng multi-GPU
    torch.backends.cudnn.deterministic = True  # Đảm bảo tính deterministic
    torch.backends.cudnn.benchmark = False
    # OS
    os.environ["PYTHONHASHSEED"] = str(seed)

# Cố định seed trước khi chạy đánh giá
set_seed(42)  # Seed có thể chọn bất kỳ (ví dụ: 42)

# Run evaluation
eer_baseline_scores = []
for _ in range(5):  # Chạy 5 lần
    eer_baseline = calculate_eer_baseline(
        test_dataset,
        baseline_trainer,
        num_speakers=20,
        num_samples_per_speaker=20
    )
    eer_baseline_scores.append(eer_baseline)

# In kết quả
print(f"Mean EER: {np.mean(eer_baseline_scores)*100:.2f}% ± {np.std(eer_baseline_scores)*100:.2f}%")

# Assuming you have already trained your MAML model
# and have access to the voice embedding extractor

# Path to the target audio file for which you want to generate an embedding
target_audio_path = "/content/drive/MyDrive/LibriSpeech/train-clean-100/103/1240/103-1240-0000.flac"

# Paths to reference audio files from the same speaker
reference_audio_paths = [
    "/content/drive/MyDrive/LibriSpeech/train-clean-100/103/1240/103-1240-0001.flac",
    "/content/drive/MyDrive/LibriSpeech/train-clean-100/103/1241/103-1241-0002.flac",
    "/content/drive/MyDrive/LibriSpeech/train-clean-100/103/1241/103-1241-0003.flac",
    "/content/drive/MyDrive/LibriSpeech/train-clean-100/103/1241/103-1241-0004.flac",
    "/content/drive/MyDrive/LibriSpeech/train-clean-100/103/1241/103-1241-0005.flac",
    "/content/drive/MyDrive/LibriSpeech/train-clean-100/103/1241/103-1241-0006.flac",
    "/content/drive/MyDrive/LibriSpeech/train-clean-100/103/1241/103-1241-0007.flac",
    "/content/drive/MyDrive/LibriSpeech/train-clean-100/103/1241/103-1241-0008.flac"
]

# Number of fine-tuning steps (default is 50 in the function)
n_steps = 50

# Generate the fine-tuned embedding
baseline_embedding = baseline_trainer.generate_embedding_after_finetune(
    target_audio_path=target_audio_path,
    reference_audio_paths=reference_audio_paths,
    extractor=extractor,  # Your voice embedding extractor
    n_steps=n_steps
)

# Now you can use fine_tuned_embedding for further processing
# For example, save it to a file
np.save("baseline_embedding.npy", baseline_embedding)

import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from IPython.display import Audio
import torchaudio
from speechbrain.inference.vocoders import HIFIGAN
import pystoi  # Import the pystoi library
from pystoi import stoi  # Import the stoi function from pystoi

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

# 2. Load embedding
embedding_path = "/content/baseline_embedding.npy"
try:
    embedding = np.load(embedding_path, allow_pickle=True)
    print("Embedding loaded. Shape:", embedding.shape)
except Exception as e:
    print("Lỗi khi load embedding:", e)
    raise

# 3. Chuẩn bị dữ liệu
if isinstance(embedding, np.ndarray):
    embedding_tensor = torch.FloatTensor(embedding)
elif isinstance(embedding, torch.Tensor):
    embedding_tensor = embedding.float()
else:
    raise ValueError("Định dạng embedding không hỗ trợ")

if len(embedding_tensor.shape) == 1:
    embedding_tensor = embedding_tensor.unsqueeze(0)

# 4. Khởi tạo model và vocoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmbeddingToMelMLP().to(device)
model.eval()

# Load vocoder HiFi-GAN
hifi_gan = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-ljspeech",
    savedir="tmp_vocoder"
).to(device).eval()

# 5. Tạo mel-spectrogram từ embedding
with torch.no_grad():
    mel_output = model(embedding_tensor.to(device))

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
if reference_audio_path:
    audio_ref, fs = librosa.load(reference_audio_path, sr=22050)

    # Đảm bảo cùng độ dài
    min_len = min(len(audio_ref), len(audio_synth))
    audio_ref = audio_ref[:min_len]
    audio_synth = audio_synth[:min_len]

    # Tính STOI
    stoi_score = stoi(audio_ref, audio_synth, fs_sig=22050, extended=False)
    print(f"STOI score: {stoi_score:.4f} (1.0 là tốt nhất)")
else:
    print("Không có audio gốc để tính STOI")

"""# 7. MAML Algorithm Implementation"""

# Khởi tạo model
model = FullyConnectedNeuralNetwork(input_dim=256, hidden_dim1=256, hidden_dim2=128).to(device)

class MAML:
    def __init__(self, model, inner_lr=0.02, meta_lr=0.002, weight_decay=0.01):
        """
        - model: MAML neural network model
        - inner_lr (float): Learning rate for updates within each task.
        - meta_lr (float): Learning rate for meta-optimization.
        - weight_decay (float): Weight regularization parameter.
        """
        self.model = model
        self.inner_lr = inner_lr
        self.meta_optimizer = optim.Adam(model.parameters(),
                                       lr=meta_lr,
                                       weight_decay=weight_decay)

    def adapt(self, support_x, support_y, n_steps=5, first_order=False):
        """
        Perform adaptation/fine-tuning with gradient descent steps on support set.
        Implements proper second-order derivatives when first_order=False.

        - support_x: Input data for support set
        - support_y: Target data for support set
        - n_steps: Number of gradient updates
        - first_order: If True, skip second-order derivatives for speed

        Returns dictionary of adapted parameters
        """
        # Create dictionary of fast weights
        fast_weights = {name: param.clone() for name, param in self.model.named_parameters()}

        for step in range(n_steps):
            # Forward pass with current weights
            support_pred = self.model(support_x, fast_weights)
            support_loss = F.mse_loss(support_pred, support_y)

            # Calculate gradients
            grads = torch.autograd.grad(
                support_loss,
                fast_weights.values(),
                create_graph=not first_order,  # Important for second-order optimization
                allow_unused=True
            )

            # Update fast weights
            fast_weights = {
                name: param - self.inner_lr * (grad if grad is not None else 0.0)
                for (name, param), grad in zip(fast_weights.items(), grads)
            }

        return fast_weights

    def calculate_metrics(self, x, y, params=None):
        """
        Calculate evaluation metrics: MSE Loss, Cosine Similarity and Speaker Similarity.

        - x: Input data
        - y: Target data
        - params: Custom parameters (if any)

        Returns dictionary with evaluation metrics
        """
        with torch.no_grad():
            # Predict with fine-tuned weights
            pred = self.model(x, params) if params else self.model(x)

            # Calculate metrics
            mse_loss = F.mse_loss(pred, y).item()
            cos_sim = F.cosine_similarity(pred, y).mean().item()
            speaker_sim = (cos_sim + 1) / 2

            return {
                'mse_loss': mse_loss,
                'cosine_similarity': cos_sim,
                'speaker_similarity': speaker_sim
            }

    def meta_train(self, train_dataset, val_dataset=None, n_epochs=100, n_support=10, n_query=10, batch_size=64, n_tasks=64, eval_interval=5):
        """
        Train MAML with proper second-order derivatives.

        - train_dataset: Dataset containing training tasks.
        - val_dataset: Dataset containing validation tasks (optional).
        - n_epochs (int): Number of training epochs.
        - n_support (int): Number of samples in support set.
        - n_query (int): Number of samples in query set.
        - batch_size (int): Batch size for training.
        - n_tasks (int): Number of tasks to sample each epoch.
        - eval_interval (int): Epochs between evaluations.
        """
        metrics_history = {
            'epoch': [],
            'meta_loss': [],
            'val_mse': [],
            'val_cosine_sim': [],
            'val_speaker_sim': []
        }

        for epoch in range(n_epochs):
            meta_losses = []
            task_count = 0

            # Calculate required batches
            n_batches = (n_tasks + batch_size - 1) // batch_size

            for _ in range(n_batches):
                self.meta_optimizer.zero_grad()
                batch_meta_loss = 0
                actual_batch_size = min(batch_size, n_tasks - task_count)

                # Calculate loss for each task in batch
                for _ in range(actual_batch_size):
                    # Sample task from dataset
                    support_x, support_y, query_x, query_y = train_dataset.sample_task(n_support, n_query)

                    # Move data to appropriate device
                    device = next(self.model.parameters()).device
                    support_x = support_x.to(device)
                    support_y = support_y.to(device)
                    query_x = query_x.to(device)
                    query_y = query_y.to(device)

                    # Perform adaptation on support set
                    # Keep second-order derivatives for accurate MAML
                    adapted_params = self.adapt(support_x, support_y, n_steps=5, first_order=False)

                    # Calculate loss on query set
                    query_pred = self.model(query_x, adapted_params)
                    task_loss = F.mse_loss(query_pred, query_y)

                    # Scale and backpropagate loss
                    scaled_loss = task_loss / actual_batch_size
                    scaled_loss.backward()

                    # Update statistics
                    batch_meta_loss += task_loss.item()
                    task_count += 1

                # Update meta-parameters after accumulating gradients
                self.meta_optimizer.step()

                # Calculate average loss for current batch
                batch_meta_loss = batch_meta_loss / actual_batch_size
                meta_losses.append(batch_meta_loss)

                print(f'Epoch {epoch+1}/{n_epochs}, Loss: {batch_meta_loss:.4f}')

            # Calculate average loss for epoch
            epoch_meta_loss = sum(meta_losses) / len(meta_losses)
            metrics_history['epoch'].append(epoch + 1)
            metrics_history['meta_loss'].append(epoch_meta_loss)

            # Evaluate model at intervals
            if ((epoch + 1) % eval_interval == 0) or (epoch == n_epochs - 1):
                print(f'Epoch {epoch+1}/{n_epochs}: Train Loss = {epoch_meta_loss:.4f}')

                # If validation dataset exists, evaluate on it
                if val_dataset is not None:
                    val_metrics = self.evaluate_on_dataset(val_dataset, n_tasks=10, n_support=n_support, n_query=n_query)
                    metrics_history['val_mse'].append(val_metrics['mse_loss'])
                    metrics_history['val_cosine_sim'].append(val_metrics['cosine_similarity'])
                    metrics_history['val_speaker_sim'].append(val_metrics['speaker_similarity'])

                    print(f"Validation Metrics - MSE: {val_metrics['mse_loss']:.4f}, "
                          f"Cosine Similarity: {val_metrics['cosine_similarity']:.4f}, "
                          f"Speaker Similarity: {val_metrics['speaker_similarity']:.4f}")

        return metrics_history

    def evaluate_on_dataset(self, dataset, n_tasks=10, n_support=10, n_query=10):
        """
        Evaluate model on a dataset.

        - dataset: Dataset containing tasks to evaluate.
        - n_tasks: Number of tasks to use for evaluation.
        - n_support: Number of samples in support set.
        - n_query: Number of samples in query set.

        Returns average metrics.
        """
        metrics_list = []

        # Evaluate on multiple tasks
        for _ in range(n_tasks):
            # Sample task from dataset
            support_x, support_y, query_x, query_y = dataset.sample_task(n_support, n_query)

            # Move data to appropriate device
            device = next(self.model.parameters()).device
            support_x = support_x.to(device)
            support_y = support_y.to(device)
            query_x = query_x.to(device)
            query_y = query_y.to(device)

            # Adaptation/fine-tune on support set
            # Use first_order=True for evaluation to increase speed
            adapted_params = self.adapt(support_x, support_y, n_steps=10, first_order=True)

            # Evaluate on query set
            metrics = self.calculate_metrics(query_x, query_y, adapted_params)
            metrics_list.append(metrics)

        # Calculate average metrics
        avg_metrics = {
            'mse_loss': np.mean([m['mse_loss'] for m in metrics_list]),
            'cosine_similarity': np.mean([m['cosine_similarity'] for m in metrics_list]),
            'speaker_similarity': np.mean([m['speaker_similarity'] for m in metrics_list])
        }

        return avg_metrics

    def generate_embedding_after_finetune(self, target_audio_path, reference_audio_paths, extractor, n_steps=50):
        """
        Generate embedding for a target audio file after fine-tuning on reference files.
        Follows MAML principle: adapt on support set, predict on new data.

        Args:
            target_audio_path (str): Path to the target audio file
            reference_audio_paths (list): List of paths to reference audio files (same speaker)
            extractor: Voice embedding extractor
            n_steps (int): Number of fine-tuning steps

        Returns:
            numpy.ndarray: The fine-tuned embedding for the target audio
        """
        # Extract embedding from target audio (query sample)
        target_embedding = extractor.get_normalized_embedding(target_audio_path)
        if target_embedding is None:
            raise ValueError(f"Failed to extract embedding from target audio: {target_audio_path}")

        device = next(self.model.parameters()).device
        target_embedding_tensor = torch.FloatTensor(target_embedding).unsqueeze(0).to(device)

        # Extract reference embeddings (support set)
        support_embeddings = []
        for ref_path in reference_audio_paths:
            embedding = extractor.get_normalized_embedding(ref_path)
            if embedding is not None:
                support_embeddings.append(torch.FloatTensor(embedding))

        if not support_embeddings:
            raise ValueError("No valid reference embeddings found")

        # Create tensors for adaptation
        support_x = torch.stack(support_embeddings).to(device)
        support_y = support_x.clone()  # We want the model to preserve embeddings

        # Adapt model to the support set (MAML's inner loop)
        adapted_params = self.adapt(support_x, support_y, n_steps=n_steps, first_order=True)

        # Generate embedding for target using adapted model (MAML's inference)
        with torch.no_grad():
            fine_tuned_embedding = self.model(target_embedding_tensor, adapted_params)
            return fine_tuned_embedding.squeeze(0).cpu().numpy()

# Initialize MAML trainer
maml = MAML(model, inner_lr=0.02, meta_lr=0.002, weight_decay=0.01)

# Start meta-training
print("\nStarting meta-training with performance evaluation...")
metrics_history = maml.meta_train(
        train_dataset=train_dataset,
        val_dataset=test_dataset,  # Use test_dataset for validation
        n_epochs=100,
        n_support=15,
        n_query=5,
        batch_size=64,
        n_tasks=64,
        eval_interval=5  # Evaluate every 5 epochs
    )

# Plot training metrics
plt.figure(figsize=(15, 5))

# Plot 1: Meta Loss
plt.subplot(1, 3, 1)
plt.plot(metrics_history['epoch'], metrics_history['meta_loss'])
plt.title('Meta Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plot validation metrics
if 'val_mse' in metrics_history and len(metrics_history['val_mse']) > 0:
    eval_epochs = metrics_history['epoch'][::5]  # For eval_interval=5

    # Plot 2: Validation Cosine Similarity
    plt.subplot(1, 3, 2)
    plt.plot(eval_epochs, metrics_history['val_cosine_sim'])
    plt.title('Validation Cosine Similarity')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.ylim(-1, 1)  # Cosine similarity range is [-1, 1]

    # Plot 3: Validation Speaker Similarity
    plt.subplot(1, 3, 3)
    plt.plot(eval_epochs, metrics_history['val_speaker_sim'])
    plt.title('Validation Speaker Similarity')
    plt.xlabel('Epoch')
    plt.ylabel('Similarity')
    plt.ylim(0, 1)  # Speaker similarity range is [0, 1]

plt.tight_layout()
plt.savefig('training_metrics.png')
plt.show()

# Evaluate model on test dataset
print("\nEvaluating model...")
n_test_tasks = 20
test_metrics = []

device = next(model.parameters()).device

for i in range(n_test_tasks):
    support_x, support_y, query_x, query_y = test_dataset.sample_task(15, 5)

    # Move data to device
    support_x = support_x.to(device)
    support_y = support_y.to(device)
    query_x = query_x.to(device)
    query_y = query_y.to(device)

    # Evaluate before fine-tuning
    metrics_before = maml.calculate_metrics(query_x, query_y)

    # Fine-tune and evaluate
    adapted_params = maml.adapt(support_x, support_y, n_steps=50, first_order=True)
    metrics_after = maml.calculate_metrics(query_x, query_y, adapted_params)

    test_metrics.append({
        'before': metrics_before,
        'after': metrics_after
    })

    if (i + 1) % 5 == 0:
        print(f"\nTest task {i+1}/{n_test_tasks}")
        print("Metrics before/after fine-tuning:")
        for metric in metrics_before.keys():
            avg_before = np.mean([m['before'][metric] for m in test_metrics[:i+1]])
            avg_after = np.mean([m['after'][metric] for m in test_metrics[:i+1]])
            print(f"{metric}: {avg_before:.4f} -> {avg_after:.4f}")

import torch
import numpy as np
from speechbrain.utils.metric_stats import EER
import matplotlib.pyplot as plt
import torch.nn.functional as F

def calculate_eer(test_dataset, maml, num_speakers=30, num_samples_per_speaker=10):
    """
    Calculate Equal Error Rate using embeddings already in the dataset.

    Args:
        test_dataset: The test dataset containing embeddings and speaker IDs
        maml: The MAML model
        num_speakers: Number of speakers to use for evaluation
        num_samples_per_speaker: Number of samples per speaker

    Returns:
        Equal Error Rate
    """
    # Get device from model
    device = next(maml.model.parameters()).device

    # Select random speakers from test dataset
    all_test_speakers = list(set(test_dataset.speaker_ids))
    if len(all_test_speakers) < num_speakers:
        print(f"Warning: Only {len(all_test_speakers)} speakers available, using all of them")
        selected_speakers = all_test_speakers
    else:
        selected_speakers = np.random.choice(all_test_speakers, num_speakers, replace=False)

    positive_scores = []
    negative_scores = []

    for target_speaker in selected_speakers:
        # Get all embeddings of this speaker
        speaker_indices = [i for i, sid in enumerate(test_dataset.speaker_ids) if sid == target_speaker]

        # Ensure we have enough samples
        if len(speaker_indices) < num_samples_per_speaker:
            continue

        # Get actual embedding tensors for this speaker
        speaker_embeddings = [test_dataset.embeddings[i] for i in speaker_indices[:num_samples_per_speaker]]

        # Use 5 embeddings as support set and 1 as target
        # Chọn ngẫu nhiên 6 indices
        indices = np.random.choice(len(speaker_embeddings), 6, replace=False)
        support_indices = indices[:5]
        target_index = indices[5]
        support_embeddings = [speaker_embeddings[i] for i in support_indices]
        target_embedding = speaker_embeddings[target_index]

        # Move to device
        support_x = torch.stack(support_embeddings).to(device)
        support_y = support_x.clone()  # We want the model to preserve embeddings
        target_embedding = target_embedding.to(device)

        # Fine-tune model on support set
        adapted_params = maml.adapt(support_x, support_y, n_steps=50, first_order=True)

        # Get model prediction for target embedding
        with torch.no_grad():
            fine_tuned_embedding = maml.model(target_embedding.unsqueeze(0), adapted_params)
            fine_tuned_embedding = fine_tuned_embedding.squeeze(0)

        # Calculate cosine similarity for positive pair (same speaker)
        cos_sim_positive = F.cosine_similarity(
            target_embedding.unsqueeze(0),
            fine_tuned_embedding.unsqueeze(0)
        ).item()
        positive_scores.append(cos_sim_positive)

        # Create negative pair (different speaker)
        other_speaker = np.random.choice([s for s in selected_speakers if s != target_speaker])
        other_indices = [i for i, sid in enumerate(test_dataset.speaker_ids) if sid == other_speaker]

        if not other_indices:
            continue

        other_embedding = test_dataset.embeddings[np.random.choice(other_indices)].to(device)

        # Calculate cosine similarity for negative pair (different speaker)
        cos_sim_negative = F.cosine_similarity(
            target_embedding.unsqueeze(0),
            other_embedding.unsqueeze(0)
        ).item()
        negative_scores.append(cos_sim_negative)

    if not positive_scores or not negative_scores:
        raise ValueError("Not enough data to calculate EER")

    # Convert to tensors for EER calculation
    positive_scores_tensor = torch.tensor(positive_scores)
    negative_scores_tensor = torch.tensor(negative_scores)

    # Calculate EER
    eer_score, threshold = EER(positive_scores_tensor, negative_scores_tensor)

    # Visualize score distribution
    plt.figure(figsize=(10, 6))
    plt.hist(positive_scores, bins=30, alpha=0.5, label="Positive (Same Speaker)")
    plt.hist(negative_scores, bins=30, alpha=0.5, label="Negative (Different Speaker)")
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.3f}')
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title(f"Speaker Verification Score Distribution - EER: {eer_score*100:.2f}%")
    plt.savefig('eer_distribution.png')
    plt.show()

    return eer_score

import numpy as np
import torch
import random
import os

# Hàm cố định seed cho tất cả thư viện
def set_seed(seed=42):
    # Python
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Nếu dùng multi-GPU
    torch.backends.cudnn.deterministic = True  # Đảm bảo tính deterministic
    torch.backends.cudnn.benchmark = False
    # OS
    os.environ["PYTHONHASHSEED"] = str(seed)

# Cố định seed trước khi chạy đánh giá
set_seed(42)  # Seed có thể chọn bất kỳ (ví dụ: 42)

# Run evaluation
eer_scores = []
for _ in range(5):  # Chạy 5 lần
    eer = calculate_eer(test_dataset, maml, num_speakers=50, num_samples_per_speaker=20)
    eer_scores.append(eer)
print(f"Mean EER: {np.mean(eer_scores)*100:.2f}% ± {np.std(eer_scores)*100:.2f}%")

# Assuming you have already trained your MAML model
# and have access to the voice embedding extractor

# Path to the target audio file for which you want to generate an embedding
target_audio_path = "/content/drive/MyDrive/LibriSpeech/train-clean-100/103/1240/103-1240-0000.flac"

# Paths to reference audio files from the same speaker
reference_audio_paths = [
    "/content/drive/MyDrive/LibriSpeech/train-clean-100/103/1240/103-1240-0001.flac",
    "/content/drive/MyDrive/LibriSpeech/train-clean-100/103/1241/103-1241-0002.flac",
    "/content/drive/MyDrive/LibriSpeech/train-clean-100/103/1241/103-1241-0003.flac",
    "/content/drive/MyDrive/LibriSpeech/train-clean-100/103/1241/103-1241-0004.flac",
    "/content/drive/MyDrive/LibriSpeech/train-clean-100/103/1241/103-1241-0005.flac",
    "/content/drive/MyDrive/LibriSpeech/train-clean-100/103/1241/103-1241-0006.flac",
    "/content/drive/MyDrive/LibriSpeech/train-clean-100/103/1241/103-1241-0007.flac",
    "/content/drive/MyDrive/LibriSpeech/train-clean-100/103/1241/103-1241-0008.flac"
]

# Number of fine-tuning steps (default is 50 in the function)
n_steps = 50

# Generate the fine-tuned embedding
fine_tuned_embedding = maml.generate_embedding_after_finetune(
    # Your trained MAML model
    target_audio_path=target_audio_path,
    reference_audio_paths=reference_audio_paths,
    extractor=extractor,  # Your voice embedding extractor
    n_steps=n_steps
)

# Now you can use fine_tuned_embedding for further processing
# For example, save it to a file
np.save("fine_tuned_embedding.npy", fine_tuned_embedding)

import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from IPython.display import Audio
import torchaudio
from speechbrain.inference.vocoders import HIFIGAN
import pystoi  # Import the pystoi library
from pystoi import stoi  # Import the stoi function from pystoi

# 1. Define the model to convert embedding -> mel-spectrogram
class EmbeddingToMelMLP(nn.Module):
    def __init__(self, embedding_dim=256, mel_dim=80, hidden_dim=512, time_steps=100):
        super().__init__() # Initialize the parent nn.Module class
        self.time_steps = time_steps # Store the desired number of time steps in the output mel-spectrogram
        self.base = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),     # First fully connected layer
            nn.ReLU(),                             # ReLU activation function
            nn.Linear(hidden_dim, hidden_dim),   # Second fully connected layer
            nn.ReLU()                             # ReLU activation function
        )
        # Final layer to project the hidden representation to a flattened mel-spectrogram
        self.time_distributed = nn.Linear(hidden_dim, mel_dim * time_steps)

    def forward(self, x): # Function to pass data through the network.
        """
        Forward pass of the EmbeddingToMelMLP model.

        Args:
            x (Tensor): Input speaker embedding tensor. Expected shape (batch_size, embedding_dim).

        Returns:
            Tensor: Generated mel-spectrogram tensor. Shape (batch_size, mel_dim, time_steps).
        """
        h = self.base(x) # Pass the input embedding through the base layers
        # Project to flattened mel-spectrogram and reshape to (batch_size, mel_dim, time_steps)
        return self.time_distributed(h).view(-1, 80, self.time_steps)

# 2. Load embedding
embedding_path = "/content/fine_tuned_embedding.npy" # Path to the saved embedding file
try:
    embedding = np.load(embedding_path, allow_pickle=True) # Load the embedding using numpy
    print("Embedding loaded. Shape:", embedding.shape) # Print the shape of the loaded embedding
except Exception as e:
    print("Error loading embedding:", e) # Print error if loading fails
    raise # Re-raise the exception

# 3. Prepare data
# Check if the loaded embedding is a numpy array or a torch tensor and convert if necessary
if isinstance(embedding, np.ndarray):
    embedding_tensor = torch.FloatTensor(embedding) # Convert numpy array to PyTorch FloatTensor
elif isinstance(embedding, torch.Tensor):
    embedding_tensor = embedding.float() # Ensure the tensor is of float type
else:
    raise ValueError("Unsupported embedding format") # Raise error for unsupported types

# Add a batch dimension if the embedding is a single vector (1D)
if len(embedding_tensor.shape) == 1:
    embedding_tensor = embedding_tensor.unsqueeze(0) # Adds a dimension at the beginning (batch size 1)

# 4. Initialize model and vocoder
# Set the device to CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmbeddingToMelMLP().to(device) # Create an instance of the model and move it to the selected device
model.eval() # Set the model to evaluation mode (disables dropout, etc.)

# Load the pre-trained HiFi-GAN vocoder from SpeechBrain
hifi_gan = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-ljspeech", # Specify the pre-trained model source
    savedir="tmp_vocoder" # Directory to save the downloaded model files
).to(device).eval() # Move the vocoder to the device and set to evaluation mode

# 5. Generate mel-spectrogram from embedding
with torch.no_grad(): # Disable gradient computation for inference
    # Move the embedding tensor to the device and pass it through the model
    mel_output = model(embedding_tensor.to(device))

    # Normalize and scale the mel-spectrogram for HiFi-GAN input
    # Normalize to [0, 1] range
    mel_output = (mel_output - mel_output.min()) / (mel_output.max() - mel_output.min())
    # Scale to [-4, 4] range as expected by the HiFi-GAN model
    mel_output = mel_output * 8 - 4

# 6. Display mel-spectrogram
# Remove batch dimension, move to CPU, and convert to numpy array for plotting
mel_spec = mel_output.squeeze().cpu().numpy()
plt.figure(figsize=(10, 4)) # Create a new matplotlib figure
plt.imshow(mel_spec, aspect='auto', origin='lower') # Display the mel-spectrogram as an image
plt.colorbar() # Add a color bar
plt.title("Generated Mel-Spectrogram") # Set the title of the plot
plt.show() # Show the plot

# 7. Convert mel-spectrogram to audio using HiFi-GAN
with torch.no_grad(): # Disable gradient computation
    # Ensure mel_output has a batch dimension (required by decode_batch)
    if len(mel_output.shape) == 2:
        mel_output = mel_output.unsqueeze(0)
    # Decode the mel-spectrogram into a waveform using the HiFi-GAN vocoder
    waveforms = hifi_gan.decode_batch(mel_output)
    # Remove batch dimension, move to CPU, and convert to numpy array
    audio_synth = waveforms.squeeze().cpu().numpy()

# 8. Load reference audio for comparison (if available)
# Path to the original reference audio file (replace with your actual path)
reference_audio_path = "/content/drive/MyDrive/LibriSpeech/train-clean-100/103/1240/103-1240-0000.flac"
if reference_audio_path: # Check if a reference path is provided
    # Load the reference audio using librosa, resampling to 22050 Hz
    audio_ref, fs = librosa.load(reference_audio_path, sr=22050)

    # Ensure both waveforms have the same length by truncating
    min_len = min(len(audio_ref), len(audio_synth))
    audio_ref = audio_ref[:min_len]
    audio_synth = audio_synth[:min_len]

    # Calculate the STOI (Short-Time Objective Intelligibility) score
    stoi_score = stoi(audio_ref, audio_synth, fs_sig=22050, extended=False)
    # Print the calculated STOI score (1.0 is best)
    print(f"STOI score: {stoi_score:.4f} (1.0 is best)")
else:
    # Print a message if no reference audio path is provided
    print("No reference audio provided to calculate STOI")

"""# 8. Reptile Model Class"""

# Khởi tạo model
model = FullyConnectedNeuralNetwork(input_dim=256, hidden_dim1=256, hidden_dim2=128).to(device)

"""# 9. Reptile Algorithm Implementation"""

class Reptile:
    def __init__(self, model, inner_lr=0.02, meta_lr=0.002, weight_decay=0.01):
        """
        - model: FullyConnectedNeuralNetwork deep learning model
        - inner_lr (float): Learning rate for updates within each task.
        - meta_lr (float): Learning rate for the overall meta-learning process.
        - weight_decay (float): Weight regularization coefficient to prevent overfitting.
        """
        self.model = model
        self.inner_lr = inner_lr  # Learning rate within each sub-task
        self.meta_lr = meta_lr    # Meta learning rate
        # Does not use PyTorch's meta_optimizer like in MAML
        # because Reptile updates parameters manually

    def adapt(self, support_x, support_y, n_steps=5, first_order=True):
        """
        Performs gradient descent on a specific task to learn new parameters.
        """
        # Copy current parameters
        params = {name: param.clone() for name, param in self.model.named_parameters()}

        # Perform n gradient descent steps
        for _ in range(n_steps):
            # Predict on the support set
            support_pred = self.model(support_x, params)
            support_loss = F.mse_loss(support_pred, support_y)

            # Compute gradients of the loss with respect to each parameter (first-order only)
            grads = torch.autograd.grad(
                support_loss,
                params.values(),
                create_graph=False,  # Disable second-order gradients
                allow_unused=True
            )

            # Update parameters using gradient descent (skip grad=None)
            params = {
                name: param - self.inner_lr * grad if grad is not None else param
                for ((name, param), grad) in zip(params.items(), grads)
            }

        return params

    def calculate_metrics(self, x, y, params=None):
        """
        Calculates evaluation metrics: MSE Loss, Cosine Similarity, and Speaker Similarity.

        - x: Input data
        - y: Desired output data
        - params: Optional custom parameters

        Returns a dictionary containing the evaluation metrics
        """
        with torch.no_grad():
            # Predict with fine-tuned weights (if provided)
            pred = self.model(x, params) if params else self.model(x)

            # Calculate evaluation metrics
            mse_loss = F.mse_loss(pred, y).item()
            cos_sim = F.cosine_similarity(pred, y).mean().item()
            speaker_sim = (cos_sim + 1) / 2

            return {
                'mse_loss': mse_loss,
                'cosine_similarity': cos_sim,  # Add cosine similarity metric separately
                'speaker_similarity': speaker_sim
            }

    def meta_train(self, train_dataset, val_dataset=None, n_epochs=100, n_support=10, n_query=10, batch_size=64, n_tasks=64, eval_interval=5, inner_steps=5):
        """
        Trains the model using the Reptile algorithm.

        - train_dataset: Dataset containing training tasks.
        - val_dataset: Optional dataset containing validation tasks.
        - n_epochs (int): Number of training epochs.
        - n_support (int): Number of samples in the support set.
        - n_query (int): Number of samples in the query set.
        - batch_size (int): Batch size for training.
        - n_tasks (int): Number of tasks to sample per epoch.
        - eval_interval (int): Number of epochs between evaluations.
        - inner_steps (int): Number of update steps within each task.
        """
        metrics_history = {
            'epoch': [],
            'meta_loss': [],
            'val_mse': [],
            'val_cosine_sim': [],  # Track cosine similarity
            'val_speaker_sim': []
        }

        for epoch in range(n_epochs):
            meta_loss = 0
            n_batches = n_tasks // batch_size + (1 if n_tasks % batch_size != 0 else 0)

            # Calculate loss and update parameters
            for batch in range(n_batches):
                batch_meta_loss = 0
                actual_batch_size = min(batch_size, n_tasks - batch * batch_size)

                # Save original parameters before processing batch
                original_params = {name: param.clone() for name, param in self.model.named_parameters()}

                # Accumulate parameter changes from each task
                accumulated_params_updates = {name: torch.zeros_like(param) for name, param in original_params.items()}

                for _ in range(actual_batch_size):
                    # Sample a task from the dataset
                    support_x, support_y, query_x, query_y = train_dataset.sample_task(n_support, n_query)

                    # Move data to GPU if needed
                    device = next(self.model.parameters()).device # Get device from model
                    support_x = support_x.to(device)
                    support_y = support_y.to(device)
                    query_x = query_x.to(device)
                    query_y = query_y.to(device)

                    # Update model on the support set (inner loop)
                    adapted_params = self.adapt(support_x, support_y, n_steps=inner_steps)

                    # Calculate the difference compared to the original parameters
                    for name in original_params:
                        param_update = adapted_params[name] - original_params[name]
                        accumulated_params_updates[name].add_(param_update)

                    # Evaluate on the query set (for monitoring only, does not affect update)
                    query_pred = self.model(query_x, adapted_params)
                    task_loss = F.mse_loss(query_pred, query_y)
                    batch_meta_loss += task_loss.item()

                # Average the updates from all tasks in the batch
                for name in accumulated_params_updates:
                    accumulated_params_updates[name] /= actual_batch_size

                # Update parameters according to the Reptile algorithm
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        # Reptile update: θ ← θ + ε(θ_task - θ)
                        param.add_(accumulated_params_updates[name] * self.meta_lr)

                # Calculate average loss for the batch (for monitoring)
                batch_meta_loss = batch_meta_loss / actual_batch_size
                meta_loss += batch_meta_loss * actual_batch_size

                # Print info for each batch
                if (batch + 1) % max(1, n_batches // 3) == 0 or batch == n_batches - 1:
                    print(f'Epoch {epoch+1}/{n_epochs}, Loss: {batch_meta_loss:.4f}')

            meta_loss = meta_loss / n_tasks
            metrics_history['epoch'].append(epoch + 1)
            metrics_history['meta_loss'].append(meta_loss)

            # Evaluate model after every eval_interval epochs or the last epoch
            if ((epoch + 1) % eval_interval == 0) or (epoch == n_epochs - 1):
                print(f'Epoch {epoch+1}/{n_epochs}: Train Loss = {meta_loss:.4f}')

                # If validation dataset exists, evaluate on it
                if val_dataset is not None:
                    val_metrics = self.evaluate_on_dataset(val_dataset, n_tasks=10, n_support=n_support, n_query=n_query, inner_steps=inner_steps)
                    metrics_history['val_mse'].append(val_metrics['mse_loss'])
                    metrics_history['val_cosine_sim'].append(val_metrics['cosine_similarity'])  # Save cosine similarity
                    metrics_history['val_speaker_sim'].append(val_metrics['speaker_similarity'])

                    print(f"Validation Metrics - MSE: {val_metrics['mse_loss']:.4f}, Cosine Similarity: {val_metrics['cosine_similarity']:.4f}, Speaker Similarity: {val_metrics['speaker_similarity']:.4f}")

        return metrics_history

    def evaluate_on_dataset(self, dataset, n_tasks=10, n_support=10, n_query=10, inner_steps=10):
        """
        Evaluates the model on a dataset.

        - dataset: Dataset containing tasks for evaluation.
        - n_tasks: Number of tasks used for evaluation.
        - n_support: Number of samples in the support set.
        - n_query: Number of samples in the query set.
        - inner_steps: Number of update steps within each task.

        Returns the average of the metrics.
        """
        metrics_list = []

        for _ in range(n_tasks):
            # Sample a task from the dataset
            support_x, support_y, query_x, query_y = dataset.sample_task(n_support, n_query)

            # Move data to GPU if needed
            device = next(self.model.parameters()).device # Get device from model
            support_x = support_x.to(device)
            support_y = support_y.to(device)
            query_x = query_x.to(device)
            query_y = query_y.to(device)

            # Fine-tune the model on the support set
            adapted_params = self.fine_tune(support_x, support_y, n_steps=inner_steps)

            # Evaluate on the query set
            metrics = self.calculate_metrics(query_x, query_y, adapted_params)
            metrics_list.append(metrics)

        # Calculate the average of the metrics
        avg_metrics = {
            'mse_loss': np.mean([m['mse_loss'] for m in metrics_list]),
            'cosine_similarity': np.mean([m['cosine_similarity'] for m in metrics_list]),  # Calculate average cosine similarity
            'speaker_similarity': np.mean([m['speaker_similarity'] for m in metrics_list])
        }

        return avg_metrics

    def fine_tune(self, support_x, support_y, n_steps=50):
        """
        Fine-tunes the model on new data.

        - support_x: Input data (embedding of the new voice).
        - support_y: Corresponding label (desired embedding).
        - n_steps (int): Number of update steps on the support set.
        - return params: Parameters after fine-tuning.
        """
        # Call the adapt method to perform fine-tuning
        return self.adapt(support_x, support_y, n_steps=n_steps)

    def evaluate(self, query_x, query_y, adapted_params=None):
        """
        - query_x: Test data (embedding of the new voice).
        - query_y: Actual label (desired embedding).
        - adapted_params: Fine-tuned parameters, if any.
        Returns:
        - MSE Loss: The mean squared error between the predicted embedding (pred) and the actual embedding (query_y). Lower MSE means the two embeddings are more similar.
        - Speaker Similarity: The similarity between the input and output voice. Value is in the range [-1, 1], where 1 means completely similar.
        """
        return self.calculate_metrics(query_x, query_y, adapted_params)

    def generate_embedding_after_finetune(self, target_audio_path, reference_audio_paths, extractor, n_steps=50):
        """
        Generate embedding for a target audio file after fine-tuning on reference files.

        Args:
            target_audio_path (str): Path to the target audio file
            reference_audio_paths (list): List of paths to reference audio files (same speaker)
            extractor: Voice embedding extractor
            n_steps (int): Number of fine-tuning steps

        Returns:
            numpy.ndarray: The fine-tuned embedding for the target audio
        """
        # Extract embedding from target audio
        target_embedding = extractor.get_normalized_embedding(target_audio_path)
        if target_embedding is None:
            raise ValueError(f"Failed to extract embedding from target audio: {target_audio_path}")

        device = next(self.model.parameters()).device
        target_embedding_tensor = torch.FloatTensor(target_embedding).unsqueeze(0).to(device)

        # Extract and validate reference embeddings
        support_embeddings = []
        for ref_path in reference_audio_paths:
            # Need to use get_normalized_embedding instead of extract_embedding
            # to ensure dimensions match the expected input
            embedding = extractor.get_normalized_embedding(ref_path)
            if embedding is not None:
                support_embeddings.append(torch.FloatTensor(embedding))

        if not support_embeddings:
            raise ValueError("No valid reference embeddings found")

        # Create tensors for fine-tuning
        support_x = torch.stack(support_embeddings).to(device)
        support_y = support_x.clone()  # In this case, we want the model to preserve the embeddings

        # Fine-tune the model on reference audio
        adapted_params = self.fine_tune(support_x, support_y, n_steps=n_steps)

        # Generate the fine-tuned embedding
        with torch.no_grad():
            fine_tuned_embedding = self.model(target_embedding_tensor, adapted_params)
            return fine_tuned_embedding.squeeze(0).cpu().numpy()

# Thay thế MAML bằng Reptile
reptile = Reptile(model, inner_lr=0.02, meta_lr=0.002, weight_decay=0.01)

# Tạo validation dataset từ một phần của train_dataset
val_dataset = train_dataset  # Hoặc tạo một val_dataset riêng

print("\nStarting meta-training with Reptile algorithm...")
metrics_history = reptile.meta_train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    n_epochs=100,
    n_support=15,
    n_query=5,
    batch_size=64,
    n_tasks=64,
    eval_interval=5,  # Đánh giá sau mỗi 5 epoch
    inner_steps=5     # Số bước inner optimization trong Reptile
)

# Plot về training loss và validation metrics
plt.figure(figsize=(15, 5))

# Plot 1: Meta Loss
plt.subplot(1, 3, 1)
plt.plot(metrics_history['epoch'], metrics_history['meta_loss'])
plt.title('Meta Loss (Reptile)')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Tạo plot về các validation metrics
if 'val_mse' in metrics_history and len(metrics_history['val_mse']) > 0:
    epochs = metrics_history['epoch'][::5]  # Cho eval_interval=5

    # Plot 2: Validation Cosine Similarity
    plt.subplot(1, 3, 2)
    plt.plot(epochs, metrics_history['val_cosine_sim'])
    plt.title('Validation Cosine Similarity')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.ylim(-1, 1)  # Cosine similarity range is [-1, 1]

    # Plot 3: Validation Speaker Similarity
    plt.subplot(1, 3, 3)
    plt.plot(epochs, metrics_history['val_speaker_sim'])
    plt.title('Validation Speaker Similarity')
    plt.xlabel('Epoch')
    plt.ylabel('Similarity')
    plt.ylim(0, 1)  # Speaker similarity range is [0, 1]

plt.tight_layout()
plt.savefig('reptile_training_metrics.png')
plt.show()

# Đánh giá mô hình trên test_dataset
print("\nEvaluating Reptile model...")
n_test_tasks = 20
test_metrics = []

for i in range(n_test_tasks):
    support_x, support_y, query_x, query_y = test_dataset.sample_task(15, 5)

    # Đưa dữ liệu đến thiết bị (device)
    support_x = support_x.to(device)
    support_y = support_y.to(device)
    query_x = query_x.to(device)
    query_y = query_y.to(device)

    # Đánh giá trước khi fine-tuning
    metrics_before = reptile.evaluate(query_x, query_y)

    # Fine-tune và đánh giá
    adapted_params = reptile.fine_tune(support_x, support_y, n_steps=50)
    metrics_after = reptile.evaluate(query_x, query_y, adapted_params)

    test_metrics.append({
         'before': metrics_before,
          'after': metrics_after
    })

    if (i + 1) % 5 == 0:
        print(f"\nTest task {i+1}/{n_test_tasks}")
        print("Metrics before/after fine-tuning:")
        for metric in metrics_before.keys():
           avg_before = np.mean([m['before'][metric] for m in test_metrics])
           avg_after = np.mean([m['after'][metric] for m in test_metrics])
           print(f"{metric}: {avg_before:.4f} -> {avg_after:.4f}")

def calculate_eer_reptile(test_dataset, reptile, num_speakers=30, num_samples_per_speaker=10):
    """
    Calculate Equal Error Rate using embeddings already in the dataset with Reptile.

    Args:
        test_dataset: The test dataset containing embeddings and speaker IDs
        reptile: The Reptile model
        num_speakers: Number of speakers to use for evaluation
        num_samples_per_speaker: Number of samples per speaker

    Returns:
        Equal Error Rate
    """
    # Get device from model
    device = next(reptile.model.parameters()).device

    # Select random speakers from test dataset
    all_test_speakers = list(set(test_dataset.speaker_ids))
    if len(all_test_speakers) < num_speakers:
        print(f"Warning: Only {len(all_test_speakers)} speakers available, using all of them")
        selected_speakers = all_test_speakers
    else:
        selected_speakers = np.random.choice(all_test_speakers, num_speakers, replace=False)

    positive_scores = []
    negative_scores = []

    for target_speaker in selected_speakers:
        # Get all embeddings of this speaker
        speaker_indices = [i for i, sid in enumerate(test_dataset.speaker_ids) if sid == target_speaker]

        # Ensure we have enough samples
        if len(speaker_indices) < num_samples_per_speaker:
            continue

        # Get actual embedding tensors for this speaker
        speaker_embeddings = [test_dataset.embeddings[i] for i in speaker_indices[:num_samples_per_speaker]]

        # Use 5 embeddings as support set and 1 as target
        indices = np.random.choice(len(speaker_embeddings), 6, replace=False)
        support_indices = indices[:5]
        target_index = indices[5]
        support_embeddings = [speaker_embeddings[i] for i in support_indices]
        target_embedding = speaker_embeddings[target_index]

        # Move to device
        support_x = torch.stack(support_embeddings).to(device)
        support_y = support_x.clone()  # We want the model to preserve embeddings
        target_embedding = target_embedding.to(device)

        # Fine-tune model on support set using Reptile
        adapted_params = reptile.fine_tune(support_x, support_y, n_steps=50)

        # Get model prediction for target embedding
        with torch.no_grad():
            fine_tuned_embedding = reptile.model(target_embedding.unsqueeze(0), adapted_params)
            fine_tuned_embedding = fine_tuned_embedding.squeeze(0)

        # Calculate cosine similarity for positive pair (same speaker)
        cos_sim_positive = F.cosine_similarity(
            target_embedding.unsqueeze(0),
            fine_tuned_embedding.unsqueeze(0)
        ).item()
        positive_scores.append(cos_sim_positive)

        # Create negative pair (different speaker)
        other_speaker = np.random.choice([s for s in selected_speakers if s != target_speaker])
        other_indices = [i for i, sid in enumerate(test_dataset.speaker_ids) if sid == other_speaker]

        if not other_indices:
            continue

        other_embedding = test_dataset.embeddings[np.random.choice(other_indices)].to(device)

        # Calculate cosine similarity for negative pair (different speaker)
        cos_sim_negative = F.cosine_similarity(
            target_embedding.unsqueeze(0),
            other_embedding.unsqueeze(0)
        ).item()
        negative_scores.append(cos_sim_negative)

    if not positive_scores or not negative_scores:
        raise ValueError("Not enough data to calculate EER")

    # Convert to tensors for EER calculation
    positive_scores_tensor = torch.tensor(positive_scores)
    negative_scores_tensor = torch.tensor(negative_scores)

    # Calculate EER
    eer_score, threshold = EER(positive_scores_tensor, negative_scores_tensor)

    # Visualize score distribution
    plt.figure(figsize=(10, 6))
    plt.hist(positive_scores, bins=30, alpha=0.5, label="Positive (Same Speaker)")
    plt.hist(negative_scores, bins=30, alpha=0.5, label="Negative (Different Speaker)")
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.3f}')
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title(f"Speaker Verification Score Distribution (Reptile) - EER: {eer_score*100:.2f}%")
    plt.savefig('reptile_eer_distribution.png')
    plt.show()

    return eer_score

import numpy as np
import torch
import random
import os

# Hàm cố định seed cho tất cả thư viện
def set_seed(seed=42):
    # Python
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Nếu dùng multi-GPU
    torch.backends.cudnn.deterministic = True  # Đảm bảo tính deterministic
    torch.backends.cudnn.benchmark = False
    # OS
    os.environ["PYTHONHASHSEED"] = str(seed)

# Cố định seed trước khi chạy đánh giá
set_seed(42)  # Seed có thể chọn bất kỳ (ví dụ: 42)

# Run evaluation
eer_reptile_scores = []
for _ in range(5):  # Chạy 5 lần
    eer_reptile = calculate_eer_reptile(test_dataset, reptile, num_speakers=50, num_samples_per_speaker=20)
    eer_reptile_scores.append(eer_reptile)
print(f"Mean EER: {np.mean(eer_reptile_scores)*100:.2f}% ± {np.std(eer_reptile_scores)*100:.2f}%")

# Generate embedding after fine-tuning
target_audio_path = "/content/drive/MyDrive/LibriSpeech/train-clean-100/103/1240/103-1240-0000.flac"

# Paths to reference audio files from the same speaker
reference_audio_paths = [
    "/content/drive/MyDrive/LibriSpeech/train-clean-100/103/1240/103-1240-0001.flac",
    "/content/drive/MyDrive/LibriSpeech/train-clean-100/103/1241/103-1241-0002.flac",
    "/content/drive/MyDrive/LibriSpeech/train-clean-100/103/1241/103-1241-0003.flac",
    "/content/drive/MyDrive/LibriSpeech/train-clean-100/103/1241/103-1241-0004.flac",
    "/content/drive/MyDrive/LibriSpeech/train-clean-100/103/1241/103-1241-0005.flac",
    "/content/drive/MyDrive/LibriSpeech/train-clean-100/103/1241/103-1241-0006.flac",
    "/content/drive/MyDrive/LibriSpeech/train-clean-100/103/1241/103-1241-0007.flac",
    "/content/drive/MyDrive/LibriSpeech/train-clean-100/103/1241/103-1241-0008.flac"
]

# Generate the fine-tuned embedding using Reptile
fine_tuned_embedding = reptile.generate_embedding_after_finetune(
    target_audio_path=target_audio_path,
    reference_audio_paths=reference_audio_paths,
    extractor=extractor,
    n_steps=50
)

# Save the embedding
np.save("reptile_fine_tuned_embedding.npy", fine_tuned_embedding)
print("Fine-tuned embedding saved as reptile_fine_tuned_embedding.npy")

import torch
import numpy as np
from speechbrain.utils.metric_stats import EER

# Load embedding từ file hoặc pipeline trực tiếp
embedding_original = extractor.get_normalized_embedding(target_audio_path) # Call the extract_embedding method
embedding_cloning = "/content/reptile_fine_tuned_embedding.npy"  # Lấy từ pipeline, không cần lưu file

import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from IPython.display import Audio
import torchaudio
from speechbrain.inference.vocoders import HIFIGAN
import pystoi  # Import the pystoi library
from pystoi import stoi  # Import the stoi function from pystoi

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

# 2. Load embedding
embedding_path = "/content/reptile_fine_tuned_embedding.npy"
try:
    embedding = np.load(embedding_path, allow_pickle=True)
    print("Embedding loaded. Shape:", embedding.shape)
except Exception as e:
    print("Lỗi khi load embedding:", e)
    raise

# 3. Chuẩn bị dữ liệu
if isinstance(embedding, np.ndarray):
    embedding_tensor = torch.FloatTensor(embedding)
elif isinstance(embedding, torch.Tensor):
    embedding_tensor = embedding.float()
else:
    raise ValueError("Định dạng embedding không hỗ trợ")

if len(embedding_tensor.shape) == 1:
    embedding_tensor = embedding_tensor.unsqueeze(0)

# 4. Khởi tạo model và vocoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmbeddingToMelMLP().to(device)
model.eval()

# Load vocoder HiFi-GAN
hifi_gan = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-ljspeech",
    savedir="tmp_vocoder"
).to(device).eval()

# 5. Tạo mel-spectrogram từ embedding
with torch.no_grad():
    mel_output = model(embedding_tensor.to(device))

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
if reference_audio_path:
    audio_ref, fs = librosa.load(reference_audio_path, sr=22050)

    # Đảm bảo cùng độ dài
    min_len = min(len(audio_ref), len(audio_synth))
    audio_ref = audio_ref[:min_len]
    audio_synth = audio_synth[:min_len]

    # Tính STOI
    stoi_score = stoi(audio_ref, audio_synth, fs_sig=22050, extended=False)
    print(f"STOI score: {stoi_score:.4f} (1.0 là tốt nhất)")
else:
    print("Không có audio gốc để tính STOI")