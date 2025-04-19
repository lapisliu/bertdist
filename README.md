# bertdist

## Setup (CPU only for now)
I have only tested this on CPU. To set up the environment, run the following commands:

```bash
conda create -n bertdist_cpu python=3.11
conda activate bertdist_cpu
# Make sure it's activated
# `which pip` should give you something like 
# /home/username/.conda/envs/bertdist_cpu/bin/pip 
pip install -r requirements.txt
```

## Runing
Change the `PYSPARK_PYTHON` environment variable.
```bash
export PYSPARK_PYTHON=$(which python)
```

Test BERT inference with a small dataset on the master node:
```bash
# create the dataset folder if not exists
hadoop fs -mkdir -p datasets/queries/
# upload a small dataset
hadoop fs -put sample_input.txt datasets/queries/
```

Execute on the master node (for only testing small datasets):
```bash
spark-submit \
  --master local[2] \
  --deploy-mode client \
  bert_inference_cpu.py \
  --input-path datasets/queries/ \
  --output-path outputs/bert_cls_embeddings_cpu/
```

Check the results
```bash
hadoop fs -ls outputs/bert_cls_embeddings_cpu/

spark-submit \
  --master local[2] \
  --deploy-mode client \
  teacher_result_check.py \
  --parquet-path outputs/bert_cls_embeddings_cpu/
```