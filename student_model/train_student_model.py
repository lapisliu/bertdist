import argparse
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from pyspark.sql import SparkSession
from transformers import BertTokenizer

from student_model import BertStudentMLP, BertEmbeddingDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a student model to mimic BERT embeddings"
    )
    parser.add_argument(
        "--bert-embeddings", "-b",
        required=True,
        help="HDFS path to the BERT embeddings Parquet files"
    )
    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        help="Directory to save the trained model"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=30522,  # Default vocab size for bert-base-uncased
        help="Vocabulary size for the student model"
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Embedding dimension for the student model"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension for the student model"
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=768,  # BERT base hidden size
        help="Output dimension (should match BERT's output dimension)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )
    
    return parser.parse_args()


def setup_distributed(rank, world_size):
    """
    Setup distributed training
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", 
                           rank=rank, 
                           world_size=world_size)
    
    # Set device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        

def load_data_from_hdfs(bert_embeddings_path):
    """
    Load BERT embeddings from HDFS using Spark
    """
    spark = SparkSession.builder \
        .appName("Load BERT embeddings") \
        .getOrCreate()
    
    # Read the Parquet files
    df = spark.read.parquet(bert_embeddings_path)
    
    # Convert to Python objects
    rows = df.select("text", "bert_cls_embedding").collect()
    texts = [row["text"] for row in rows]
    embeddings = [row["bert_cls_embedding"] for row in rows]
    
    spark.stop()
    
    return texts, embeddings


def train_model(rank, world_size, args):
    """
    Train the student model using distributed training
    """
    # Setup distributed training
    setup_distributed(rank, world_size)
    
    # Load data
    texts, embeddings = load_data_from_hdfs(args.bert_embeddings)
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Create dataset
    dataset = BertEmbeddingDataset(texts, embeddings, tokenizer)
    
    # Split dataset for training and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, 
                                      num_replicas=world_size, 
                                      rank=rank,
                                      shuffle=True)
    
    val_sampler = DistributedSampler(val_dataset,
                                    num_replicas=world_size,
                                    rank=rank,
                                    shuffle=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize student model
    model = BertStudentMLP(
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim
    )
    
    # Move model to correct device
    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_sampler.set_epoch(epoch)
        train_loss = 0.0
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]") if rank == 0 else train_loader
        
        for batch in train_progress:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_embeddings = batch['embedding'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            
            # Compute loss
            loss = criterion(outputs, target_embeddings)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if rank == 0 and isinstance(train_progress, tqdm):
                train_progress.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]") if rank == 0 else val_loader
            
            for batch in val_progress:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                target_embeddings = batch['embedding'].to(device)
                
                # Forward pass
                outputs = model(input_ids, attention_mask)
                
                # Compute loss
                loss = criterion(outputs, target_embeddings)
                
                val_loss += loss.item()
                
                if rank == 0 and isinstance(val_progress, tqdm):
                    val_progress.set_postfix({'loss': loss.item()})
        
        val_loss /= len(val_loader)
        
        # Print progress
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                # Save model
                os.makedirs(args.output_dir, exist_ok=True)
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }
                torch.save(checkpoint, os.path.join(args.output_dir, 'best_student_model.pt'))
                print(f"Saved best model with validation loss: {val_loss:.4f}")
    
    # Cleanup distributed
    dist.destroy_process_group()


def main():
    # Parse arguments
    args = parse_args()
    
    # Get number of available GPUs or use CPUs
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
    else:
        world_size = 1  # Use 1 process in CPU-only mode
    
    if args.local_rank == -1:
        # Launch distributed processes
        torch.multiprocessing.spawn(
            train_model,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    else:
        # Launched with torch.distributed.launch
        train_model(args.local_rank, world_size, args)


if __name__ == "__main__":
    main()