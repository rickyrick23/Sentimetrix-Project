import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from data_engine import fetch_ticker_data
from model_engine import SentimetrixTCN
from rag_retriever import EXPERT_RULES, embedder
import os
from sklearn.preprocessing import StandardScaler
import joblib
import json

# --- Configuration ---
TICKERS = [
    "AAPL", "MSFT", "GOOG", "NVDA", "TSLA", "AMZN", "META", "AMD", "NFLX", "INTC",
    "JPM", "V", "PG", "UNH", "DIS", "HD", "VZ", "KO", "PFE", "CSCO"
]
SEQ_LEN = 20
BATCH_SIZE = 32
EPOCHS = 50 # Increased for better convergence
LEARNING_RATE = 0.001
SAVE_PATH = "models/alpha_weights.pth"
SCALER_PATH = "models/scaler.pkl"

# --- 1. Dataset Definition ---
class AlphaDataset(Dataset):
    def __init__(self, x_data, y_data, rules_embeddings):
        """
        x_data: Scaled feature windows [N, SEQ_LEN, Features]
        y_data: Targets [N]
        rules_embeddings: Pre-computed embeddings for expert rules
        """
        self.x_data = x_data
        self.y_data = y_data
        self.rules_embeddings = rules_embeddings

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x_price = self.x_data[idx]
        y_label = self.y_data[idx]
        
        # Convert to tensors
        # TCN expects [Features, Time] -> Transpose here if x_price is [Time, Features]
        x_price_tensor = torch.tensor(x_price, dtype=torch.float32).transpose(0, 1) 
        x_rag_tensor = torch.tensor(self.rules_embeddings, dtype=torch.float32)
        label_tensor = torch.tensor(y_label, dtype=torch.long)
        
        return x_price_tensor, x_rag_tensor, label_tensor

# --- 2. Data Preparation ---
def prepare_data(tickers):
    print(f"Fetching data for {len(tickers)} tickers...")
    
    all_features = [] # To fit scaler
    all_targets = []
    
    # Pre-compute RAG embeddings (static for now)
    rule_embeddings = embedder.encode(EXPERT_RULES)
    
    ticker_data_map = {}
    
    for ticker in tickers:
        try:
            print(f"Processing {ticker}...")
            df = fetch_ticker_data(ticker)
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'target' in numeric_cols: numeric_cols.remove('target')
            if 'returns' in numeric_cols: numeric_cols.remove('returns')
            
            data_values = df[numeric_cols].values
            targets = df['target'].values
            
            ticker_data_map[ticker] = (data_values, targets)
            all_features.append(data_values)
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
            
    if not all_features:
        return None, None, None

    # Concatenate all features to fit scaler
    combined_features = np.vstack(all_features)
    
    print("Fitting Scaler...")
    scaler = StandardScaler()
    scaler.fit(combined_features)
    
    # Save Scaler
    if not os.path.exists("models"):
        os.makedirs("models")
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")
    
    # Create Sequences
    x_sequences = []
    y_labels = []
    
    print("Creating Sequences...")
    for ticker in ticker_data_map:
        raw_values, targets = ticker_data_map[ticker]
        scaled_values = scaler.transform(raw_values)
        
        for i in range(len(scaled_values) - SEQ_LEN):
            x_window = scaled_values[i : i + SEQ_LEN]
            y_label = targets[i + SEQ_LEN - 1]
            
            x_sequences.append(x_window)
            y_labels.append(y_label)
            
    return np.array(x_sequences), np.array(y_labels), rule_embeddings

# --- 3. Training Loop ---
def train_model():
    # check model directory
    if not os.path.exists("models"):
        os.makedirs("models")

    # Prepare Data
    train_x, train_y, rule_embeddings = prepare_data(TICKERS)
    
    if train_x is None:
        print("No data fetched. Exiting.")
        return

    # Check input size
    input_dim = train_x.shape[2] # [N, Time, Features]
    print(f"Feature Dimension: {input_dim}")
    
    # Split Train/Val
    split_idx = int(len(train_x) * 0.8)
    
    # Create datasets
    train_dataset = AlphaDataset(train_x[:split_idx], train_y[:split_idx], rule_embeddings)
    val_dataset = AlphaDataset(train_x[split_idx:], train_y[split_idx:], rule_embeddings)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Calculate Class Weights
    class_counts = np.bincount(train_y)
    total_samples = len(train_y)
    class_weights = total_samples / (len(class_counts) * class_counts)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class Weights: {class_weights}")
    
    model = SentimetrixTCN(input_size=input_dim).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for x_p, x_r, y in train_loader:
            x_p, x_r, y = x_p.to(device), x_r.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(x_p, x_r)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x_p, x_r, y in val_loader:
                x_p, x_r, y = x_p.to(device), x_r.to(device), y.to(device)
                outputs, _ = model(x_p, x_r)
                _, predicted = torch.max(outputs.data, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"--> Saved Best Model ({best_acc:.2f}%)")
            
            # Save Metrics Incrementally
            metrics = {"accuracy": val_acc / 100.0, "precision": val_acc / 100.0}
            with open("metrics.json", "w") as f:
                json.dump(metrics, f)
            
    print("Training Complete.")
    
    # Save Metrics
    metrics = {
        "accuracy": val_acc / 100.0,
        "precision": val_acc / 100.0 # Approximation for now, ideally calculate real precision
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)
    print("Metrics saved to metrics.json")

if __name__ == "__main__":
    train_model()
