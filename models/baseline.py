import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .voice_models import FullyConnectedNeuralNetwork
from config import Config

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