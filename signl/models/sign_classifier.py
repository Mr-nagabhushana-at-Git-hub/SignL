# src/majorSignL/models/sign_classifier.py
# PyTorch GPU-only Sign Language Classifier

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time

# Enable cuDNN autotuning for maximum performance
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

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

class TransformerBlock(nn.Module):
    """PyTorch Transformer block for sign language classification"""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
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
    """Complete PyTorch-based Sign Language Transformer"""
    
    def __init__(self, sequence_length=30, num_landmarks=1662, num_actions=10, 
                 embed_dim=128, num_heads=8, ff_dim=128, num_layers=3, dropout=0.1):
        super(SignLanguageTransformer, self).__init__()
        
        self.sequence_length = sequence_length
        self.num_landmarks = num_landmarks
        self.num_actions = num_actions
        self.embed_dim = embed_dim
        
        # Input embedding layer
        self.input_embedding = nn.Linear(num_landmarks, embed_dim)
        
        # Positional encoding
        self.register_buffer('pos_encoding', self._create_positional_encoding(sequence_length, embed_dim))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Global average pooling and classification head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_actions)
        )
        
    def _create_positional_encoding(self, seq_len, embed_dim):
        """Create positional encoding for transformer"""
        pe = torch.zeros(seq_len, embed_dim)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           -(np.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Add batch dimension
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, num_landmarks)
        batch_size, seq_len, _ = x.shape
        
        # Input embedding
        x = self.input_embedding(x)  # (batch, seq, embed_dim)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Global pooling: (batch, seq, embed) -> (batch, embed)
        x = x.transpose(1, 2)  # (batch, embed, seq)
        x = self.global_pool(x).squeeze(-1)  # (batch, embed)
        
        # Classification
        logits = self.classifier(x)  # (batch, num_actions)
        
        return logits

class SignClassifier:
    """PyTorch GPU-accelerated Sign Language Classifier"""
    
    def __init__(self, model_path: Optional[Path] = None):
        """Initialize PyTorch-based sign classifier"""
        
        # Device setup - prioritize CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🔥 SignClassifier using device: {self.device}")
        
        # Model and state
        self.model = None
        self.actions = ['hello', 'thanks', 'yes', 'no', 'please', 'sorry', 'help', 'good', 'bad', 'love']
        self.label_map = {action: i for i, action in enumerate(self.actions)}
        self.sequence_length = 30
        self.num_landmarks = 1692  # Updated: MediaPipe Holistic with refined face landmarks (132+1434+63+63)
        self.current_sequence = []
        self.confidence_threshold = 0.5  # Lower for better detection
        
        # Performance tracking
        self.last_prediction_time = 0
        self.prediction_count = 0
        
        # Load model if available or setup demo
        if model_path and model_path.exists():
            self.load_model(model_path)
        else:
            logger.warning("🤟 No trained model found - initializing with default actions")
            # Try to load anyway, it might exist after restart
            self._setup_demo_model()
    
    def _setup_demo_model(self):
        """Setup a demo model for testing without trained weights"""
        try:
            # Use default action list
            num_actions = len(self.actions)
            
            self.model = CompactSignGRU(
                input_dim=self.num_landmarks,
                hidden_dim=64,
                num_layers=2,
                num_classes=num_actions,
                dropout=0.3
            ).to(self.device)
            self.model.eval()  # Demo mode
            logger.info(f"✅ Demo model initialized with {num_actions} actions")
        except Exception as e:
            logger.error(f"❌ Failed to setup demo model: {e}")
    
    def load_model(self, model_path: Path) -> bool:
        """Load trained PyTorch model - supports both Transformer and CompactGRU"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract model configuration
            config = checkpoint.get('config', {})
            num_actions = config.get('num_actions', len(self.actions))
            architecture = config.get('architecture', 'Transformer')
            
            # Create model based on architecture type
            if architecture == 'CompactGRU':
                self.model = CompactSignGRU(
                    input_dim=config.get('input_dim', 1692),
                    hidden_dim=config.get('hidden_dim', 64),
                    num_layers=config.get('num_layers', 2),
                    num_classes=num_actions,
                    dropout=config.get('dropout', 0.3)
                ).to(self.device)
                logger.info(f"🎯 Loading CompactGRU model...")
            else:
                self.model = SignLanguageTransformer(
                    sequence_length=self.sequence_length,
                    num_landmarks=self.num_landmarks,
                    num_actions=num_actions,
                    embed_dim=config.get('embed_dim', 128),
                    num_heads=config.get('num_heads', 8),
                    ff_dim=config.get('ff_dim', 128)
                ).to(self.device)
                logger.info(f"🎯 Loading Transformer model...")
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Load actions
            self.actions = checkpoint.get('actions', self.actions)
            self.label_map = {action: i for i, action in enumerate(self.actions)}
            
            model_type = checkpoint.get('model_type', 'unknown')
            logger.info(f"✅ Loaded {architecture} model ({model_type}) with {num_actions} actions on {self.device}")
            logger.info(f"🤟 Available signs: {', '.join(self.actions[:10])}{'...' if len(self.actions) > 10 else ''}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load PyTorch model: {e}")
            import traceback
            traceback.print_exc()
            self._setup_demo_model()  # Fallback to demo
            return False
    
    def extract_landmarks(self, results) -> Optional[np.ndarray]:
        """Extract landmarks from MediaPipe results - maintains compatibility"""
        try:
            landmarks = np.zeros(self.num_landmarks, dtype=np.float32)
            
            # Pose landmarks (33 * 4 = 132)
            if results.pose_landmarks:
                pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] 
                                for lm in results.pose_landmarks.landmark], dtype=np.float32).flatten()
                landmarks[:132] = pose
            
            # Face landmarks (468 refined landmarks = 1404 → now 478 = 1434)
            if results.face_landmarks:
                face = np.array([[lm.x, lm.y, lm.z] 
                                for lm in results.face_landmarks.landmark], dtype=np.float32).flatten()
                landmarks[132:132+len(face)] = face  # Dynamic sizing for compatibility
            
            # Left hand landmarks (21 * 3 = 63)
            if results.left_hand_landmarks:
                left_hand = np.array([[lm.x, lm.y, lm.z] 
                                     for lm in results.left_hand_landmarks.landmark], dtype=np.float32).flatten()
                landmarks[1566:1629] = left_hand  # Updated offset: 132 + 1434
            
            # Right hand landmarks (21 * 3 = 63)
            if results.right_hand_landmarks:
                right_hand = np.array([[lm.x, lm.y, lm.z] 
                                      for lm in results.right_hand_landmarks.landmark], dtype=np.float32).flatten()
                landmarks[1629:1692] = right_hand  # Updated offset: 132 + 1434 + 63
            
            return landmarks
            
        except Exception as e:
            logger.error(f"Error extracting landmarks: {e}")
            return None
    
    def update_sequence(self, mediapipe_results) -> Optional[Dict]:
        """Update sequence and predict - GPU accelerated"""
        if not self.model:
            return {
                "predicted_sign": "No Model",
                "confidence": 0.0,
                "sequence_progress": 0.0,
                "status": "Model not loaded"
            }
        
        landmarks = self.extract_landmarks(mediapipe_results)
        if landmarks is None:
            return None
        
        # Add to current sequence
        self.current_sequence.append(landmarks)
        
        # Keep only the last sequence_length frames
        if len(self.current_sequence) > self.sequence_length:
            self.current_sequence = self.current_sequence[-self.sequence_length:]
        
        # Predict when we have full sequence
        if len(self.current_sequence) == self.sequence_length:
            return self.predict_sign()
        
        return {
            "predicted_sign": "Collecting...",
            "confidence": 0.0,
            "sequence_progress": len(self.current_sequence) / self.sequence_length,
            "status": f"Collecting frames {len(self.current_sequence)}/{self.sequence_length}"
        }
    
    def predict_sign(self) -> Dict:
        """GPU-accelerated sign prediction"""
        start_time = time.time()
        
        try:
            if not self.model or len(self.current_sequence) != self.sequence_length:
                return {"predicted_sign": "Incomplete", "confidence": 0.0, "status": "Sequence incomplete"}
            
            # Convert to PyTorch tensor and move to GPU
            sequence = np.array(self.current_sequence, dtype=np.float32)
            input_tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)  # Add batch dim
            
            # GPU prediction
            with torch.no_grad():
                logits = self.model(input_tensor)
                probabilities = F.softmax(logits, dim=-1)
                confidence, predicted_class = torch.max(probabilities, dim=-1)
            
            # Convert back to CPU for final processing
            predicted_idx = predicted_class.item()
            confidence_score = confidence.item()
            
            # Get predicted sign name
            if predicted_idx < len(self.actions):
                predicted_sign = self.actions[predicted_idx]
            else:
                predicted_sign = "Unknown"
            
            # Apply confidence threshold
            if confidence_score < self.confidence_threshold:
                predicted_sign = "Uncertain"
            
            # Performance tracking
            prediction_time = (time.time() - start_time) * 1000
            self.last_prediction_time = prediction_time
            self.prediction_count += 1
            
            # All predictions for debugging
            all_predictions = {}
            probs_cpu = probabilities.cpu().numpy()[0]
            for i, action in enumerate(self.actions):
                if i < len(probs_cpu):
                    all_predictions[action] = float(probs_cpu[i])
            
            return {
                "predicted_sign": predicted_sign,
                "confidence": confidence_score,
                "all_predictions": all_predictions,
                "sequence_progress": 1.0,
                "status": "Active prediction",
                "prediction_time_ms": prediction_time,
                "device": str(self.device)
            }
            
        except Exception as e:
            logger.error(f"GPU prediction error: {e}")
            return {
                "predicted_sign": "Error", 
                "confidence": 0.0,
                "status": f"Prediction error: {str(e)}",
                "device": str(self.device)
            }
    
    def reset_sequence(self):
        """Reset the current sequence"""
        self.current_sequence = []
        logger.debug("Sign sequence reset")
    
    def get_model_info(self) -> Dict:
        """Get comprehensive model information"""
        return {
            "model_loaded": self.model is not None,
            "actions": self.actions,
            "sequence_length": self.sequence_length,
            "confidence_threshold": self.confidence_threshold,
            "current_sequence_length": len(self.current_sequence),
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "backend": "PyTorch GPU",
            "status": "Ready" if self.model else "No Model",
            "last_prediction_time_ms": self.last_prediction_time,
            "total_predictions": self.prediction_count
        }
    
    def set_confidence_threshold(self, threshold: float):
        """Set the confidence threshold for predictions"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Set confidence threshold to {self.confidence_threshold}")
    
    def warm_up_gpu(self):
        """Warm up GPU with dummy prediction"""
        if self.model and torch.cuda.is_available():
            try:
                dummy_sequence = torch.randn(1, self.sequence_length, self.num_landmarks).to(self.device)
                with torch.no_grad():
                    _ = self.model(dummy_sequence)
                logger.info("🔥 GPU warmed up successfully")
            except Exception as e:
                logger.warning(f"GPU warm-up failed: {e}")

