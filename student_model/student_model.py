import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class BertStudentMLP(nn.Module):
    """
    A simple MLP student model that mimics BERT embeddings.
    Takes tokenized input and produces embeddings similar to BERT's [CLS] token.
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, output_dim=768, dropout=0.1):
        """
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of the token embeddings
            hidden_dim: Dimension of the hidden layer
            output_dim: Dimension of the output (should match BERT's output dimension, typically 768)
            dropout: Dropout probability
        """
        super(BertStudentMLP, self).__init__()
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Simple MLP layers
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Layer normalization (similar to BERT)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass of the student model
        
        Args:
            input_ids: Tensor of token ids
            attention_mask: Optional attention mask (not used in this simple implementation)
            
        Returns:
            Tensor of shape (batch_size, output_dim) - mimicking BERT's [CLS] embedding
        """
        # Embed tokens
        embeds = self.embedding(input_ids)
        
        # Average embeddings (simple approach)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeds.size())
            embeds = embeds * mask_expanded
            # Average over sequence length (dim=1) considering the mask
            sum_embeddings = embeds.sum(dim=1)
            sum_mask = attention_mask.sum(dim=1, keepdim=True)
            avg_embeddings = sum_embeddings / (sum_mask + 1e-10)
        else:
            # Simple average if no mask provided
            avg_embeddings = embeds.mean(dim=1)
        
        # Pass through MLP
        x = F.relu(self.fc1(avg_embeddings))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        return x


class BertEmbeddingDataset(Dataset):
    """
    Dataset for training the student model using BERT embeddings
    """
    def __init__(self, texts, embeddings, tokenizer, max_length=128):
        """
        Args:
            texts: List of input texts
            embeddings: List of corresponding BERT embeddings
            tokenizer: Tokenizer to use for encoding the texts
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.embeddings = embeddings
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        embedding = self.embeddings[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'embedding': torch.tensor(embedding, dtype=torch.float)
        }