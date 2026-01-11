import torch
import torch.nn as nn
import torchvision.models as models


class BaselineStoryModel(nn.Module):
    """
    BASELINE MODEL
    - Image encoder (CNN)
    - Text encoder (BiLSTM)
    - Simple concatenation-based fusion
    - No cross-modal attention
    """

    def __init__(
        self,
        vocab_size,
        embed_dim=512,
        txt_embed_dim=256,
        hidden_dim=512,
        num_layers=1,
        dropout=0.3
    ):
        super().__init__()

        # -------------------------
        # IMAGE ENCODER (ResNet-18)
        # -------------------------
        resnet = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # remove FC
        self.img_proj = nn.Linear(512, embed_dim)

        # Freeze CNN (baseline simplicity)
        for p in self.cnn.parameters():
            p.requires_grad = False

        # -------------------------
        # TEXT ENCODER
        # -------------------------
        self.embedding = nn.Embedding(vocab_size, txt_embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=txt_embed_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # -------------------------
        # FUSION (SIMPLE CONCAT)
        # -------------------------
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # -------------------------
        # OUTPUT HEAD (TEXT)
        # -------------------------
        self.output_fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images, input_ids):
        """
        Args:
            images: (B, 5, 3, 224, 224)
            input_ids: (B, L)

        Returns:
            logits: (B, L, vocab_size)
        """

        B, T, C, H, W = images.shape

        # -------------------------
        # IMAGE FEATURES
        # -------------------------
        images = images.view(B * T, C, H, W)
        img_feat = self.cnn(images).squeeze(-1).squeeze(-1)  # (B*T, 512)
        img_feat = self.img_proj(img_feat)                    # (B*T, D)
        img_feat = img_feat.view(B, T, -1).mean(dim=1)       # (B, D)

        # -------------------------
        # TEXT FEATURES
        # -------------------------
        txt_emb = self.embedding(input_ids)                  # (B, L, E)
        txt_out, _ = self.lstm(txt_emb)                      # (B, L, H)

        # -------------------------
        # FUSION
        # -------------------------
        img_feat = img_feat.unsqueeze(1).expand(-1, txt_out.size(1), -1)
        fused = torch.cat([img_feat, txt_out], dim=-1)       # (B, L, D+H)
        fused = self.fusion(fused)

        # -------------------------
        # OUTPUT
        # -------------------------
        logits = self.output_fc(fused)                       # (B, L, vocab)

        return logits
|

# ============================================================================
# CELL 8: MEMORY-EFFICIENT MODEL (Fixes OOM)
# ============================================================================

import torch
import torch.nn as nn
from torchvision import models

class CrossModalStoryModel(nn.Module):
    """
    Memory-efficient story generation model
    Decodes one step at a time without storing intermediate outputs
    """
    def __init__(self, vocab_size, embed_dim=256, num_heads=4, hidden_dim=256,
                 num_layers=1, dropout=0.1, max_seq_len=200):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # ============================================================
        # 1. IMAGE ENCODER: ResNet50 (Pre-trained, Frozen)
        # ============================================================
        self.image_encoder = models.resnet50(pretrained=True)
        # Freeze encoder to save memory
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        self.image_encoder.fc = nn.Linear(2048, embed_dim)

        # ============================================================
        # 2. TEXT ENCODER: Unidirectional LSTM (Single Layer)
        # ============================================================
        self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.text_lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,  # Single layer to save memory
            batch_first=True,
            dropout=0 if num_layers == 1 else dropout,
            bidirectional=False  # Unidirectional to save memory
        )

        # ============================================================
        # 3. SIMPLIFIED FUSION (Just concatenation)
        # ============================================================
        # Average image features and concatenate with text
        self.fusion_proj = nn.Linear(embed_dim + hidden_dim, embed_dim)

        # ============================================================
        # 4. DECODER: Single-step LSTM
        # ============================================================
        self.decoder_lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, images, texts):
        """
        Args:
            images: (batch_size, 5, 3, 224, 224)
            texts: (batch_size, seq_len)

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size = images.shape[0]
        seq_len = texts.shape[1]

        # ============================================================
        # STAGE 1: IMAGE ENCODING (Frozen)
        # ============================================================
        with torch.no_grad():
            images_flat = images.view(-1, 3, 224, 224)
            img_features = self.image_encoder(images_flat)  # (batch*5, embed_dim)
            img_features = img_features.view(batch_size, 5, self.embed_dim)

            # Average 5 images into 1 context vector
            img_context = img_features.mean(dim=1)  # (batch, embed_dim)

        # ============================================================
        # STAGE 2: TEXT ENCODING
        # ============================================================
        text_embed = self.word_embed(texts)  # (batch, seq_len, embed_dim)
        text_lstm_out, (text_h, text_c) = self.text_lstm(text_embed)
        # text_h: (num_layers, batch, hidden_dim)

        # ============================================================
        # STAGE 3: FUSION - Create context for each position
        # ============================================================
        # Concatenate image context with each text position
        img_context_expanded = img_context.unsqueeze(1).expand(-1, seq_len, -1)
        # (batch, seq_len, embed_dim + embed_dim) → (batch, seq_len, embed_dim)
        fused = torch.cat([text_lstm_out, img_context_expanded], dim=2)
        fused = self.fusion_proj(fused)
        fused = self.dropout(fused)

        # ============================================================
        # STAGE 4: DECODING (Memory-Efficient)
        # ============================================================
        # Initialize hidden state
        hidden = (text_h, text_c)

        # Generate logits step-by-step (don't store intermediate results)
        logits_list = []

        for t in range(seq_len):
            # Decode one token at a time
            decoder_input = fused[:, t:t+1]  # (batch, 1, embed_dim)
            decoder_output, hidden = self.decoder_lstm(decoder_input, hidden)
            # decoder_output: (batch, 1, hidden_dim)

            logit = self.output_proj(decoder_output)  # (batch, 1, vocab_size)
            logits_list.append(logit)

        # Concatenate all logits
        logits = torch.cat(logits_list, dim=1)  # (batch, seq_len, vocab_size)

        return logits

print("✓ Memory-efficient CrossModalStoryModel defined")
