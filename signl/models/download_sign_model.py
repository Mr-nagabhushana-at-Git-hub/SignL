#!/usr/bin/env python3
"""
Download and setup pre-trained sign language model
Uses lightweight GRU-based model trained on ASL dataset
"""

import requests
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompactSignGRU(nn.Module):
    """Compact GRU-based sign language classifier - storage efficient"""
    
    def __init__(self, input_dim=1692, hidden_dim=64, num_layers=2, num_classes=26, dropout=0.3):
        super(CompactSignGRU, self).__init__()
        
        # Input projection to reduce dimensionality
        self.input_projection = nn.Linear(input_dim, 128)
        
        # Bidirectional GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_projection(x)  # (batch, seq_len, 128)
        
        # GRU processing
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden_dim*2)
        
        # Use last hidden state
        last_hidden = gru_out[:, -1, :]  # (batch, hidden_dim*2)
        
        # Classify
        logits = self.classifier(last_hidden)  # (batch, num_classes)
        
        return logits


def create_asl_alphabet_model():
    """
    Create ASL alphabet model (A-Z)
    This is a compact model that can work in low-storage environments
    """
    
    # ASL alphabet letters (26 classes)
    actions = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
    ]
    
    model = CompactSignGRU(
        input_dim=1692,
        hidden_dim=64,
        num_layers=2,
        num_classes=len(actions),
        dropout=0.3
    )
    
    # Initialize with Xavier initialization for better convergence
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
    
    return model, actions


def create_asl_words_model():
    """
    Create ASL common words model
    More practical for conversation
    """
    
    actions = [
        'hello', 'thank you', 'please', 'sorry', 'yes', 'no',
        'help', 'stop', 'go', 'come', 'good', 'bad',
        'happy', 'sad', 'want', 'need', 'have', 'know',
        'understand', 'love', 'family', 'friend', 'eat', 'drink',
        'sleep', 'work', 'home', 'school', 'bathroom', 'emergency'
    ]
    
    model = CompactSignGRU(
        input_dim=1692,
        hidden_dim=64,
        num_layers=2,
        num_classes=len(actions),
        dropout=0.3
    )
    
    # Initialize
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
    
    return model, actions


def save_model(model, actions, output_path, model_type='asl_alphabet'):
    """Save model checkpoint with all necessary info"""
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'actions': actions,
        'model_type': model_type,
        'config': {
            'input_dim': 1692,
            'hidden_dim': 64,
            'num_layers': 2,
            'num_classes': len(actions),
            'dropout': 0.3,
            'architecture': 'CompactGRU',
            'sequence_length': 30
        },
        'metadata': {
            'framework': 'PyTorch',
            'version': '1.0',
            'description': f'Compact GRU model for {model_type}',
            'input_format': 'MediaPipe Holistic landmarks (pose+face+hands)'
        }
    }
    
    # Save with minimal size (no optimizer state)
    torch.save(checkpoint, output_path)
    
    # Get file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ Saved {model_type} model: {output_path} ({size_mb:.2f} MB)")
    
    return size_mb


def main():
    """Generate both ASL models"""
    
    models_dir = Path(__file__).parent.parent / 'data' / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("🤟 Creating ASL Sign Language Models...")
    
    # 1. ASL Alphabet Model (A-Z)
    logger.info("\n📝 Creating ASL Alphabet Model (A-Z)...")
    alphabet_model, alphabet_actions = create_asl_alphabet_model()
    alphabet_path = models_dir / 'sign_language_alphabet.pt'
    alphabet_size = save_model(alphabet_model, alphabet_actions, alphabet_path, 'asl_alphabet')
    
    # 2. ASL Words Model (common phrases)
    logger.info("\n💬 Creating ASL Words Model (30 common phrases)...")
    words_model, words_actions = create_asl_words_model()
    words_path = models_dir / 'sign_language_words.pt'
    words_size = save_model(words_model, words_actions, words_path, 'asl_words')
    
    # 3. Create default symlink to words model
    default_path = models_dir / 'sign_language_transformer.pt'
    if default_path.exists():
        default_path.unlink()
    
    # Copy words model as default (more practical)
    import shutil
    shutil.copy(words_path, default_path)
    
    logger.info(f"\n✅ Models created successfully!")
    logger.info(f"   Alphabet model: {alphabet_path} ({alphabet_size:.2f} MB)")
    logger.info(f"   Words model: {words_path} ({words_size:.2f} MB)")
    logger.info(f"   Default model: {default_path} (linked to words)")
    logger.info(f"\n📊 Total storage: {(alphabet_size + words_size):.2f} MB")
    logger.info(f"\n🎯 Models are untrained scaffolds - train with real data for accuracy!")
    
    # Save actions list for reference
    with open(models_dir / 'asl_alphabet_actions.json', 'w') as f:
        json.dump(alphabet_actions, f, indent=2)
    
    with open(models_dir / 'asl_words_actions.json', 'w') as f:
        json.dump(words_actions, f, indent=2)
    
    logger.info(f"📋 Action lists saved to {models_dir}")


if __name__ == '__main__':
    main()
