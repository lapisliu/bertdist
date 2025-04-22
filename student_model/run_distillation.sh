#!/bin/bash
# Script to run the entire BERT distillation pipeline

set -e  # Exit on error

# Configuration
HADOOP_USER=$(whoami)
HDFS_BASE_DIR="/user/${HADOOP_USER}/bertdist"
BERT_INPUTS_DIR="${HDFS_BASE_DIR}/inputs"
BERT_OUTPUTS_DIR="${HDFS_BASE_DIR}/bert_embeddings"
STUDENT_MODEL_DIR="${HDFS_BASE_DIR}/student_model"
STUDENT_OUTPUTS_DIR="${HDFS_BASE_DIR}/student_embeddings"
LOCAL_OUTPUTS_DIR="./outputs"

# Make sure PYSPARK_PYTHON is set correctly
export PYSPARK_PYTHON=$(which python)

# Step 1: Prepare input data
echo "===== Step 1: Preparing input data ====="
# Check if sample data exists
hadoop fs -test -e ${BERT_INPUTS_DIR}/sample_input.txt || {
    echo "Uploading sample input data..."
    hadoop fs -mkdir -p ${BERT_INPUTS_DIR}
    hadoop fs -put sample_input.txt ${BERT_INPUTS_DIR}/
}

# Step 2: Run BERT teacher inference
echo "===== Step 2: Running BERT teacher inference ====="
echo "This may take a while depending on the size of your dataset..."
spark-submit \
    --master yarn \
    --deploy-mode cluster \
    bert_inference_cpu.py \
    --input-path ${BERT_INPUTS_DIR} \
    --output-path ${BERT_OUTPUTS_DIR}

# Step 3: Verify BERT outputs
echo "===== Step 3: Verifying BERT outputs ====="
spark-submit \
    --master yarn \
    --deploy-mode client \
    teacher_result_check.py \
    --parquet-path ${BERT_OUTPUTS_DIR}

# Step 4: Train the student model
echo "===== Step 4: Training student model ====="
echo "This will distribute training across available nodes..."
mkdir -p ${LOCAL_OUTPUTS_DIR}

# Using torch.distributed.launch
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    train_student_model.py \
    --bert-embeddings ${BERT_OUTPUTS_DIR} \
    --output-dir ${LOCAL_OUTPUTS_DIR}/student_model \
    --epochs 10 \
    --batch-size 64

# Copy student model to HDFS
echo "Copying student model to HDFS..."
hadoop fs -mkdir -p ${STUDENT_MODEL_DIR}
hadoop fs -put ${LOCAL_OUTPUTS_DIR}/student_model/best_student_model.pt ${STUDENT_MODEL_DIR}/

# Step 5: Run student model inference
echo "===== Step 5: Running student model inference ====="
spark-submit \
    --master yarn \
    --deploy-mode cluster \
    student_inference.py \
    --input-path ${BERT_INPUTS_DIR} \
    --output-path ${STUDENT_OUTPUTS_DIR} \
    --model-path ${LOCAL_OUTPUTS_DIR}/student_model/best_student_model.pt

# Step 6: Evaluate student model
echo "===== Step 6: Evaluating student model ====="
spark-submit \
    --master yarn \
    --deploy-mode client \
    evaluate_student.py \
    --bert-embeddings ${BERT_OUTPUTS_DIR} \
    --student-embeddings ${STUDENT_OUTPUTS_DIR} \
    --num-clusters 5

echo "===== Distillation pipeline completed! ====="
echo "BERT embeddings: ${BERT_OUTPUTS_DIR}"
echo "Student model: ${STUDENT_MODEL_DIR}/best_student_model.pt"
echo "Student embeddings: ${STUDENT_OUTPUTS_DIR}"