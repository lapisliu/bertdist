import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, ArrayType, FloatType, StringType

import torch
from transformers import BertTokenizer, BertModel
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run BERT CLS embedding inference over text in HDFS (CPU-only)."
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
    return parser.parse_args()

def bert_inference_udf_cpu(iterator):
    # load once per partition
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").eval()
    
    for pdf in iterator:
        texts = pdf["text"].tolist()
        with torch.no_grad():
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].tolist()
        
        yield pd.DataFrame({
            "text": texts,
            "bert_cls_embedding": cls_embeddings
        })

def main():
    args = parse_args()
    
    # start Spark
    spark = SparkSession.builder \
        .appName("BERT Teacher Inference - CPU Only") \
        .getOrCreate()
    
    # read input
    df = spark.read.text(args.input_path).withColumnRenamed("value", "text")
    
    # define output schema
    schema = StructType([
        StructField("text", StringType(), True),
        StructField("bert_cls_embedding", ArrayType(FloatType()), True),
    ])
    
    # mapInPandas UDF
    processed_df = df.mapInPandas(bert_inference_udf_cpu, schema=schema)
    
    # write parquet
    processed_df.write.mode("overwrite").parquet(args.output_path)
    
    spark.stop()

if __name__ == "__main__":
    main()

