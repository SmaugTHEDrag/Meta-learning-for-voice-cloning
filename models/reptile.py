import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy
from .voice_models import FullyConnectedNeuralNetwork
from config import Config

class Reptile:
    def __init__(self, model, inner_lr=0.02, meta_lr=0.002, weight_decay=0.01):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.weight_decay = weight_decay
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=meta_lr, weight_decay=weight_decay)
        
    def adapt(self, support_x, support_y, n_steps=5, first_order=True):
        params = {k: v.clone() for k, v in self.model.named_parameters()}
        
        for _ in range(n_steps):
            # Forward pass
            outputs = self.model(support_x, params)
            loss = nn.MSELoss()(outputs, support_y)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, params.values(), create_graph=not first_order)
            
            # Update parameters
            params = {k: v - self.inner_lr * g for (k, v), g in zip(params.items(), grads)}
            
        return params
    
    def calculate_metrics(self, x, y, params=None):
        with torch.no_grad():
            outputs = self.model(x, params)
            mse = nn.MSELoss()(outputs, y)
            cosine_sim = nn.CosineSimilarity()(outputs, y).mean()
            return {
                'mse': mse.item(),
                'cosine_similarity': cosine_sim.item()
            }
    
    def meta_train(self, train_dataset, val_dataset=None, n_epochs=100, n_support=10, n_query=10, 
                  batch_size=64, n_tasks=64, eval_interval=5, inner_steps=5):
        for epoch in range(n_epochs):
            self.model.train()
            total_loss = 0
            
            for _ in range(n_tasks):
                # Sample a task
                support_x, support_y, query_x, query_y = train_dataset.sample_task(n_support, n_query)
                
                # Adapt to the task
                adapted_params = self.adapt(support_x, support_y, n_steps=inner_steps)
                
                # Reptile update
                self.meta_optimizer.zero_grad()
                for p, adapted_p in zip(self.model.parameters(), adapted_params.values()):
                    p.grad = p - adapted_p
                self.meta_optimizer.step()
                
                # Calculate loss for monitoring
                with torch.no_grad():
                    outputs = self.model(query_x)
                    loss = nn.MSELoss()(outputs, query_y)
                    total_loss += loss.item()
            
            # Print training progress
            if (epoch + 1) % eval_interval == 0:
                avg_loss = total_loss / n_tasks
                print(f'Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}')
                
                if val_dataset is not None:
                    val_metrics = self.evaluate_on_dataset(val_dataset)
                    print(f'Validation Metrics: {val_metrics}')
    
    def evaluate_on_dataset(self, dataset, n_tasks=10, n_support=10, n_query=10, inner_steps=10):
        self.model.eval()
        metrics_list = []
        
        for _ in range(n_tasks):
            support_x, support_y, query_x, query_y = dataset.sample_task(n_support, n_query)
            adapted_params = self.adapt(support_x, support_y, n_steps=inner_steps)
            metrics = self.calculate_metrics(query_x, query_y, adapted_params)
            metrics_list.append(metrics)
        
        # Average metrics across tasks
        avg_metrics = {
            k: np.mean([m[k] for m in metrics_list])
            for k in metrics_list[0].keys()
        }
        
        return avg_metrics
    
    def fine_tune(self, support_x, support_y, n_steps=50):
        return self.adapt(support_x, support_y, n_steps=n_steps)
    
    def evaluate(self, query_x, query_y, adapted_params=None):
        return self.calculate_metrics(query_x, query_y, adapted_params)
    
    def generate_embedding_after_finetune(self, target_audio_path, reference_audio_paths, extractor, n_steps=50):
        # Extract embeddings from reference audios
        ref_embeddings = []
        for ref_path in reference_audio_paths:
            embedding = extractor.get_normalized_embedding(ref_path)
            ref_embeddings.append(embedding)
        
        # Stack reference embeddings
        support_x = torch.stack(ref_embeddings)
        support_y = support_x.clone()  # Target is the same as input for voice conversion
        
        # Adapt the model to the reference speaker
        adapted_params = self.adapt(support_x, support_y, n_steps=n_steps)
        
        # Extract and process target audio
        target_embedding = extractor.get_normalized_embedding(target_audio_path)
        target_embedding = target_embedding.unsqueeze(0)  # Add batch dimension
        
        # Generate converted embedding
        with torch.no_grad():
            converted_embedding = self.model(target_embedding, adapted_params)
        
        return converted_embedding.squeeze()