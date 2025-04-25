from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Load PhoBERT
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
phobert = AutoModel.from_pretrained("vinai/phobert-base")

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
phobert.to(device)
phobert.eval()

# Function to get embedding for 1 text (based on [CLS] token)
def get_phobert_embedding(text):
    inputs = tokenizer.encode_plus(text, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = phobert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # get embedding of [CLS]
    return cls_embedding.squeeze().cpu().numpy()