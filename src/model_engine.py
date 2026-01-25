import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- 1. Define the Attention Mechanism ---
class AlphaAttention(nn.Module):
    def __init__(self, tcn_dim=128, rag_dim=384):
        super().__init__()
        self.query = nn.Linear(tcn_dim, 128)
        self.key = nn.Linear(rag_dim, 128)
        self.value = nn.Linear(rag_dim, 128)

    def forward(self, tcn_feat, rag_feat):
        # Attention Calculation
        Q = self.query(tcn_feat)
        K = self.key(rag_feat)
        V = self.value(rag_feat)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / 11.3
        weights = F.softmax(scores, dim=-1)
        context = torch.matmul(weights, V)
        return context, weights

# --- 2. Define the Main TCN Model ---
class SentimetrixTCN(nn.Module):
    def __init__(self, input_size=17):
        super().__init__()
        # Temporal Convolutional Network (The "Time" Brain)
        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=2, dilation=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=4, dilation=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Flatten time to 1 vector
        )
        
        # Fusion Layer (The "Reasoning" Brain)
        self.fusion = AlphaAttention()
        
        # Classifier (The "Decision" Maker)
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3) # Output: 0=Sell, 1=Hold, 2=Buy
        )

    def forward(self, x_price, x_rag):
        # x_price shape: [Batch, 17, 20]
        # x_rag shape:   [Batch, Rules, 384]
        
        p_feat = self.tcn(x_price).transpose(1, 2) # -> [Batch, 1, 128]
        r_context, weights = self.fusion(p_feat, x_rag)
        
        # Concatenate Price features + RAG Context
        combined = torch.cat([p_feat, r_context], dim=-1).squeeze(1)
        
        return self.classifier(combined), weights

# --- 3. Helper Function for Inference ---
def predict_signal(model, df, rules, embedder, scaler=None):
    """
    Runs the TCN model on the last 20 days of data + RAG rules.
    Returns: Class Index (0,1,2), Confidence %, and Attention Weights
    """
    # Prepare Numerical Data (Last 20 days)
    # Ensure we exclude 'target'/'returns' if they exist, to match training
    numeric_df = df.select_dtypes(include=[np.number])
    if 'target' in numeric_df.columns: numeric_df = numeric_df.drop(columns=['target'])
    if 'returns' in numeric_df.columns: numeric_df = numeric_df.drop(columns=['returns'])
    
    recent_data = numeric_df.values[-20:] 

    if scaler:
        recent_data = scaler.transform(recent_data)
    
    # Handle insufficient data (e.g., new stocks)
    if len(recent_data) < 20: 
        return 1, 50.0, np.zeros(len(rules)) # Return Neutral
        
    # Create Tensors
    x_price = torch.tensor(recent_data.T, dtype=torch.float32).unsqueeze(0)
    
    # Prepare Textual Data
    rule_embeddings = embedder.encode(rules)
    x_rag = torch.tensor(rule_embeddings, dtype=torch.float32).unsqueeze(0)
    
    # Run Inference
    model.eval()
    with torch.no_grad():
        logits, attn_weights = model(x_price, x_rag)
        probs = torch.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)
        
    return predicted_class.item(), confidence.item() * 100, attn_weights.squeeze().numpy()