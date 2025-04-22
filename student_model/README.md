# BERT Distillation with Spark and PyTorch

This project implements knowledge distillation from BERT to a smaller, more efficient MLP student model using Apache Spark for data processing and PyTorch for model training.

## Overview

The implementation follows these steps:
1. Data is processed using Apache Spark
2. A pre-trained BERT model extracts embeddings
- `bert_inference_cpu.py`: Script to run BERT inference on text data
- `teacher_result_check.py`: Script to inspect BERT embeddings
- `run_distillation.sh`: Bash script to run the entire pipeline

## Setup

1. Make sure you have the required dependencies installed:
```bash
pip install -r requirements.txt
```

2. Set up your environment variable:
```bash
export PYSPARK_PYTHON=$(which python)
```

## Running the Pipeline

The entire distillation process can be run using the provided shell script:

```bash
chmod +x run_distillation.sh
./run_distillation.sh
```

Or you can run each step individually:

### 1. Prepare input data

Upload your text data to HDFS:
```bash
hadoop fs -mkdir -p /user/$(whoami)/bertdist/inputs
hadoop fs -put sample_input.txt /user/$(whoami)/bertdist/inputs/
```

### 2. Run BERT teacher inference

Process the input data with BERT to generate embeddings:
```bash
spark-submit \
    --master yarn \
    --deploy-mode cluster \
    bert_inference_cpu.py \
    --input-path /user/$(whoami)/bertdist/inputs \
    --output-path /user/$(whoami)/bertdist/bert_embeddings
```

### 3. Train the student model

Train the MLP student model using distributed training:
```bash
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    train_student_model.py \
    --bert-embeddings /user/$(whoami)/bertdist/bert_embeddings \
    --output-dir ./outputs/student_model \
    --epochs 10
```

### 4. Run inference with the student model

Use the trained student model to generate embeddings:
```bash
spark-submit \
    --master yarn \
    --deploy-mode cluster \
    student_inference.py \
    --input-path /user/$(whoami)/bertdist/inputs \
    --output-path /user/$(whoami)/bertdist/student_embeddings \
    --model-path ./outputs/student_model/best_student_model.pt
```

### 5. Evaluate the student model

Compare the student model with BERT:
```bash
spark-submit \
    --master yarn \
    --deploy-mode client \
    evaluate_student.py \
    --bert-embeddings /user/$(whoami)/bertdist/bert_embeddings \
    --student-embeddings /user/$(whoami)/bertdist/student_embeddings
```

## Student Model Architecture

The student model is a simple MLP with the following architecture:
1. Word embedding layer (vocab_size → embedding_dim)
2. Average pooling across sequence length
3. Fully connected layer with ReLU (embedding_dim → hidden_dim)
4. Dropout layer
5. Fully connected layer with ReLU (hidden_dim → hidden_dim)
6. Dropout layer
7. Fully connected layer (hidden_dim → output_dim)
8. Layer normalization

The output dimension matches BERT's output (768 for bert-base-uncased), allowing direct comparison.

## Training Details

- Loss function: Mean Squared Error (MSE) between BERT embeddings and student embeddings
- Optimizer: Adam
- Distributed training: Uses PyTorch's DistributedDataParallel (DDP)
- Best model selection: Model with lowest validation loss is saved

## Validation

The student model is validated using:
1. Cosine similarity between BERT and student embeddings
2. Semantic clustering comparison (using K-means)
3. Adjusted Rand Index and Normalized Mutual Information to measure cluster agreement

## Notes

- This implementation uses CPU-only mode but can be extended to use GPUs
- For larger datasets, consider increasing the number of Spark executors and partitions
- The student model size can be adjusted by changing embedding_dim and hidden_dim parameters from text data
3. A smaller student model (MLP) is trained to mimic BERT's embeddings
4. The student model is evaluated against the teacher (BERT)

## Files

- `student_model.py`: Implementation of the student MLP model
- `train_student_model.py`: Script for training the student model with distributed training
- `student_inference.py`: Script to run inference with the trained student model
- `evaluate_student.py`: Script to evaluate and compare student vs. BERT embe