import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, ArrayType, FloatType, StringType

import torch
from transformers import BertTokenizer
import pandas as pd

from student_model import BertStudentMLP


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run student model inference over text in HDFS."
    )
    parser.add_argument(
        "--input-path", "-i",
        required=True,
        help="HDFS path (or local) to input text files; one line per record."
    )
    parser.add_argument(
        "--output-path", "-o",
        required=True,
        help="HDFS path (or local) to write Parquet output."
    )
    parser.add_argument(
        "--model-path", "-m",
        required=True,
        help="Path to the trained student model checkpoint."
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
    return parser.parse_args()


def student_inference_udf(iterator):
    """
    Spark UDF to run inference with the student model
    """
    args = parse_args()
    
    # Load once per partition
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Initialize model
    model = BertStudentMLP(
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim
    )
    
    # Load trained model weights
    checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    for pdf in iterator:
        texts = pdf["text"].tolist()
        
        student_embeddings = []
        batch_size = 32  # Process in small batches
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            with torch.no_grad():
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128
                )
                
                outputs = model(inputs['input_ids'], inputs['attention_mask'])
                student_embeddings.extend(outputs.tolist())
        
        yield pd.DataFrame({
            "text": texts,
            "student_embedding": student_embeddings
        })


def main():
    args = parse_args()
    
    # Start Spark
    spark = SparkSession.builder \
        .appName("Student Model Inference") \
        .getOrCreate()
    
    # Read input
    df = spark.read.text(args.input_path).withColumnRenamed("value", "text")
    
    # Define output schema
    schema = StructType([
        StructField("text", StringType(), True),
        StructField("student_embedding", ArrayType(FloatType()), True),
    ])
    
    # mapInPandas UDF
    processed_df = df.mapInPandas(student_inference_udf, schema=schema)
    
    # Write parquet
    processed_df.write.mode("overwrite").parquet(args.output_path)
    
    spark.stop()


if __name__ == "__main__":
    main()