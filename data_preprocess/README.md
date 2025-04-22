data_preprocess_spark_hdfs.py usage:

mkdir -p ms_marco/v1.1
cd ms_marco/v1.1
curl -L   "https://huggingface.co/datasets/microsoft/ms_marco/resolve/main/v1.1/train-00000-of-00001.parquet"   --output train-00000-of-00001.parquet

curl -L   "https://huggingface.co/datasets/microsoft/ms_marco/resolve/main/v1.1/test-00000-of-00001.parquet"   --output test-00000-of-00001.parquet

curl -L   "https://huggingface.co/datasets/microsoft/ms_marco/resolve/main/v1.1/validation-00000-of-00001.parquet"   --output validation-00000-of-00001.parquet

cd ~/bertdist/data_preprocess

hdfs dfs -put ms_marco/v1.1 /user/$(whoami)/

if you want to get the split of test, simply put 
spark-submit \
    --master yarn \
    --deploy-mode cluster \
    data_preprocess_spark_hdfs.py \
    --input hdfs:///user/$(whoami)/v1.1/test-00000-of-00001.parquet \
    --output-dir hdfs:///user/$(whoami)/output_qa
