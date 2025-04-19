import argparse
from pyspark.sql import SparkSession

def parse_args():
    parser = argparse.ArgumentParser(description="Inspect a Parquet dataset of BERT embeddings")
    parser.add_argument(
        "--parquet-path", "-p",
        required=True,
        help="HDFS or local path to the Parquet directory to inspect"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    spark = SparkSession.builder \
        .appName("Check Parquet") \
        .getOrCreate()

    # Read from the path passed in
    df = spark.read.parquet(args.parquet_path)

    # Show a few rows
    print("\n=== Sample Rows ===")
    df.show(truncate=False)

    # Print schema
    print("\n=== Schema ===")
    df.printSchema()

    # Get one example embedding
    sample = df.select("bert_cls_embedding").first()
    if sample and sample["bert_cls_embedding"] is not None:
        vec = sample["bert_cls_embedding"]
        print(f"\nFirst embedding vector length: {len(vec)}")
        print(f"Sample embedding (first 5 dims): {vec[:5]}")
    else:
        print("\nNo embeddings found in the dataset.")

    spark.stop()

if __name__ == "__main__":
    main()

