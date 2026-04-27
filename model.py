import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torchaudio

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.2):
        """
        Two-layer MLP with ReLU activation and dropout
        
        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Dimension of hidden layer
            output_dim (int): Dimension of output features
            dropout_rate (float): Dropout probability (default: 0.2)
        """
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        Forward pass through the MLP
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        return self.layers(x)


class SelfSupervisedAudioAlignmentModel(nn.Module):
    def __init__(self, audio_model, language_model, alignment_layer=-1, compute_alignment=True):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.audio_model = audio_model
        self.language_model = language_model
        self.alignment_layer = alignment_layer
        self.compute_alignment = compute_alignment

        # Freeze language model
        self.language_model.eval()
        for param in self.language_model.parameters():
            param.requires_grad = False

        self.audio_projection_mlp = MLP(384, 1024, 512)
        self.language_projection_mlp = MLP(2048, 1024, 512)

    def compute_alignment_loss(self, audio_features, language_features):
        """Compute CKA similarity-based alignment loss"""
        audio_features = audio_features - audio_features.mean(0, keepdim=True)
        language_features = language_features - language_features.mean(0, keepdim=True)

        audio_gram = torch.matmul(audio_features.T, audio_features)
        language_gram = torch.matmul(language_features.T, language_features)

        dot_product = torch.sum(audio_gram * language_gram)
        norm_audio = torch.sqrt(torch.sum(audio_gram * audio_gram))
        norm_language = torch.sqrt(torch.sum(language_gram * language_gram))

        cka = dot_product / (norm_audio * norm_language)
        return 1.0 - cka

    def forward(self, audio_input, input_ids, attention_mask):
        audio_input = torch.tensor(audio_input, dtype=torch.float32, device=self.device)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.float32, device=self.device)

        mam_loss, _, _, _ = self.audio_model.forward(audio_input)

        audio_features = self.audio_model.forward_feature(audio_input).to(torch.float32)
        audio_projection = self.audio_projection_mlp(audio_features)

        with torch.no_grad():
            outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            language_features = outputs.hidden_states[self.alignment_layer]
            language_features = torch.mean(language_features, dim=1).to(torch.float32)

        language_projection = self.language_projection_mlp(language_features)

        if self.compute_alignment:
            alignment_loss = self.compute_alignment_loss(audio_projection, language_projection)
        else:
            alignment_loss = torch.tensor(0.0, device=self.device)

        return {
            "alignment_loss": alignment_loss,
            "mam_loss": mam_loss
        }
