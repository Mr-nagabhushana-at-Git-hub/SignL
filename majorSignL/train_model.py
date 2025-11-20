#/mnt/d/AI_Team/majorSignL/ train_model.py
# PyTorch version - optimized for OpenCV integration
# Removes first layer, uses proper activations, softmax for classification

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import pickle
import logging
from majorSignL.config import TRAINING_DIR, SIGN_LANGUAGE_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignLanguageDataset(Dataset):
    """PyTorch Dataset for sign language training data"""
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class OptimizedTransformerBlock(nn.Module):
    """Optimized Transformer block for OpenCV compatibility"""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(OptimizedTransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),  # ReLU for better OpenCV compatibility
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head attention
        attn_output, _ = self.attention(x, x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)

        # Feed forward
        ff_output = self.feed_forward(out1)
        ff_output = self.dropout2(ff_output)
        out2 = self.layernorm2(out1 + ff_output)

        return out2

class SignLanguageTransformer(nn.Module):
    """Optimized PyTorch Transformer for sign language - OpenCV compatible"""
    def __init__(self, sequence_length=30, num_landmarks=1662, num_actions=10,
                 embed_dim=128, num_heads=8, ff_dim=128, num_layers=3, dropout=0.1):
        super(SignLanguageTransformer, self).__init__()

        self.sequence_length = sequence_length
        self.num_landmarks = num_landmarks
        self.num_actions = num_actions

        # Remove first dense layer - direct embedding for OpenCV optimization
        # This reduces parameters and improves inference speed
        self.input_projection = nn.Linear(num_landmarks, embed_dim)

        # Positional encoding
        self.register_buffer('pos_encoding', self._create_positional_encoding(sequence_length, embed_dim))

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            OptimizedTransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classification head with proper softmax
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_actions)
        )

    def _create_positional_encoding(self, seq_len, embed_dim):
        pe = torch.zeros(seq_len, embed_dim)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() *
                           -(np.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, num_landmarks)

        # Direct projection (no first dense layer removed)
        x = self.input_projection(x)  # (batch, seq, embed_dim)

        # Add positional encoding
        x = x + self.pos_encoding[:, :self.sequence_length, :]

        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # Global pooling: (batch, seq, embed) -> (batch, embed)
        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)

        # Classification with softmax
        logits = self.classifier(x)
        return logits

def train_model(model, train_loader, val_loader, num_epochs=100, device='cuda'):
    """Train the PyTorch model with optimized settings"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    best_accuracy = 0.0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_accuracy = 100 * train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total

        if (epoch + 1) % 10 == 0:
            logger.info('.1f')

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_model(model, val_accuracy, actions)

    logger.info('.1f')
    return model

def save_model(model, accuracy, actions):
    """Save PyTorch model with metadata"""
    MODEL_SAVE_PATH = SIGN_LANGUAGE_MODEL

    # Create model directory
    MODEL_SAVE_PATH.parent.mkdir(exist_ok=True)

    # Save model state and metadata
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': {
            'sequence_length': model.sequence_length,
            'num_landmarks': model.num_landmarks,
            'num_actions': model.num_actions,
            'embed_dim': model.input_projection.out_features,
            'accuracy': accuracy
        },
        'actions': actions
    }

    torch.save(checkpoint, MODEL_SAVE_PATH)
    logger.info(f"Model saved to {MODEL_SAVE_PATH} with {accuracy:.2f}% validation accuracy")

# --- Main Training Logic ---
if __name__ == '__main__':
    # Path configuration
    DATA_PATH = TRAINING_DIR

    # Check if training data exists
    if not DATA_PATH.exists():
        logger.error(f"Training data path {DATA_PATH} does not exist!")
        logger.info("Please create training data directory and add .npy files")
        exit(1)

    # Get actions from folder names
    actions = [action for action in os.listdir(DATA_PATH) if os.path.isdir(DATA_PATH / action)]
    label_map = {label: num for num, label in enumerate(actions)}

    print(f"Found actions: {actions}")
    print(f"Label map: {label_map}")

    # Load sequences and labels
    sequences, labels = [], []
    for action in actions:
        action_path = DATA_PATH / action
        if not action_path.exists():
            continue

        # Assume 30 videos/sequences per action
        for sequence in range(min(30, len(os.listdir(action_path)))):
            sequence_path = action_path / str(sequence)
            if not sequence_path.exists():
                continue

            window = []
            # Assume 30 frames per video
            for frame_num in range(30):
                frame_path = sequence_path / f"{frame_num}.npy"
                if frame_path.exists():
                    res = np.load(frame_path)
                    window.append(res)
                else:
                    # Pad with zeros if frame missing
                    window.append(np.zeros(1662))

            if len(window) == 30:
                sequences.append(window)
                labels.append(label_map[action])

    if not sequences:
        logger.error("No training sequences found!")
        exit(1)

    X = np.array(sequences)
    y = np.array(labels)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    # Create datasets and dataloaders
    train_dataset = SignLanguageDataset(X_train, y_train)
    val_dataset = SignLanguageDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Model configuration
    SEQUENCE_LENGTH = 30
    NUM_LANDMARKS = 1662
    NUM_SIGNS = len(actions)

    EMBED_DIM = 128
    NUM_HEADS = 8
    FF_DIM = 128
    NUM_LAYERS = 3

    # Create model
    model = SignLanguageTransformer(
        sequence_length=SEQUENCE_LENGTH,
        num_landmarks=NUM_LANDMARKS,
        num_actions=NUM_SIGNS,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_layers=NUM_LAYERS
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Train model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")

    trained_model = train_model(model, train_loader, val_loader, num_epochs=100, device=device)

    print("\n--- Training Complete ---")
    print("Model saved as PyTorch .pt file for inference compatibility")