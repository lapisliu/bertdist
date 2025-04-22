import argparse
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import size, col

"""
usage:
spark-submit \
    --master yarn \
    --deploy-mode cluster \
    data_preprocess_spark_hdfs.py \
    --input hdfs:///user/$(whoami)/v1.1/test-00000-of-00001.parquet \
    --output-dir hdfs:///user/$(whoami)/output_qa
     

"""

def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter Parquet QA pairs by answer length and write to HDFS using Spark"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="HDFS path to the input Parquet file or directory"
    )
    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        help="HDFS output directory (will contain 'queries' and 'answers' subfolders)"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    spark = SparkSession.builder.appName("QA_Filter_HDFS").getOrCreate()

    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
    Path = spark._jvm.org.apache.hadoop.fs.Path

    hdfs_output = Path(args.output_dir)
    if not fs.exists(hdfs_output):
        fs.mkdirs(hdfs_output)
        print(f"Created HDFS directory: {args.output_dir}")

    df = spark.read.parquet(args.input)

    df_filtered = df.filter(size(col("answers")) == 1)

    queries_rdd = df_filtered.select("query").rdd.map(lambda row: row[0])
    answers_rdd = df_filtered.select("answers").rdd.map(lambda row: row[0])
    
    input_file_name = os.path.splitext(os.path.basename(args.input))[0]

    out_queries = os.path.join(args.output_dir, f"{input_file_name}_queries.csv")
    out_answers = os.path.join(args.output_dir, f"{input_file_name}_answers.csv")

    queries_rdd.coalesce(1).saveAsTextFile(out_queries)
    answers_rdd.coalesce(1).saveAsTextFile(out_answers)

    print(f"Wrote {queries_rdd.count()} queries to {out_queries}")
    print(f"Wrote {answers_rdd.count()} answers to {out_answers}")

    spark.stop()

if __name__ == "__main__":
    main()



