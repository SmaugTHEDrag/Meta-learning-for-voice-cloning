import torch

class Config:
    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset paths
    LIBRISPEECH_PATH = "/content/drive/MyDrive/LibriSpeech/train-clean-100"
    
    # Model parameters
    EMBEDDING_DIM = 256
    HIDDEN_DIM1 = 256
    HIDDEN_DIM2 = 128
    
    # Training hyperparameters
    META_LR = 0.002
    INNER_LR = 0.02
    WEIGHT_DECAY = 0.01
    N_EPOCHS = 100
    BATCH_SIZE = 64
    N_SUPPORT = 15
    N_QUERY = 5
    N_TASKS = 64
    
    # Evaluation
    N_TEST_TASKS = 20
    N_EVAL_SPEAKERS = 50
    N_SAMPLES_PER_SPEAKER = 20