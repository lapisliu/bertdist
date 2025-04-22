import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from pyspark.sql import SparkSession
from pyspark.sql.functions import col


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate student model embeddings against BERT embeddings"
    )
    parser.add_argument(
        "--bert-embeddings", "-b",
        required=True,
        help="HDFS path to the BERT embeddings Parquet files"
    )
    parser.add_argument(
        "--student-embeddings", "-s",
        required=True,
        help="HDFS path to the student model embeddings Parquet files"
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=5,
        help="Number of clusters for the semantic clustering validation"
    )
    return parser.parse_args()


def compute_cosine_similarities(bert_embeddings, student_embeddings):
    """
    Compute cosine similarities between BERT and student embeddings
    """
    similarities = []
    
    for bert_emb, student_emb in zip(bert_embeddings, student_embeddings):
        # Reshape to 2D for sklearn
        bert_emb_2d = np.array(bert_emb).reshape(1, -1)
        student_emb_2d = np.array(student_emb).reshape(1, -1)
        
        # Compute cosine similarity
        sim = cosine_similarity(bert_emb_2d, student_emb_2d)[0][0]
        similarities.append(sim)
    
    return similarities


def perform_clustering(embeddings, n_clusters=5):
    """
    Perform K-means clustering on embeddings
    """
    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings)
    
    # Fit K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_array)
    
    return cluster_labels


def cluster_agreement(bert_clusters, student_clusters):
    """
    Compute cluster agreement between BERT and student model clusters
    """
    # Create contingency table
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    # Compute metrics
    ari = adjusted_rand_score(bert_clusters, student_clusters)
    nmi = normalized_mutual_info_score(bert_clusters, student_clusters)
    
    return ari, nmi


def main():
    args = parse_args()
    
    # Start Spark
    spark = SparkSession.builder \
        .appName("Evaluate Student Model") \
        .getOrCreate()
    
    # Read BERT embeddings
    bert_df = spark.read.parquet(args.bert_embeddings)
    bert_data = bert_df.select("text", "bert_cls_embedding").collect()
    
    # Read student embeddings
    student_df = spark.read.parquet(args.student_embeddings)
    student_data = student_df.select("text", "student_embedding").collect()
    
    # Extract texts and embeddings
    bert_texts = [row["text"] for row in bert_data]
    bert_embeddings = [row["bert_cls_embedding"] for row in bert_data]
    
    student_texts = [row["text"] for row in student_data]
    student_embeddings = [row["student_embedding"] for row in student_data]
    
    # Ensure the texts match
    assert bert_texts == student_texts, "The text datasets don't match!"
    
    # Compute cosine similarities
    similarities = compute_cosine_similarities(bert_embeddings, student_embeddings)
    
    # Calculate average similarity
    avg_similarity = np.mean(similarities)
    min_similarity = np.min(similarities)
    max_similarity = np.max(similarities)
    
    print("\n=== Embedding Similarity ===")
    print(f"Average cosine similarity: {avg_similarity:.4f}")
    print(f"Minimum cosine similarity: {min_similarity:.4f}")
    print(f"Maximum cosine similarity: {max_similarity:.4f}")
    
    # Perform clustering validation
    bert_clusters = perform_clustering(bert_embeddings, args.num_clusters)
    student_clusters = perform_clustering(student_embeddings, args.num_clusters)
    
    # Compute cluster agreement
    ari, nmi = cluster_agreement(bert_clusters, student_clusters)
    
    print("\n=== Clustering Validation ===")
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Normalized Mutual Information: {nmi:.4f}")
    
    # Show a few examples with similarities
    print("\n=== Sample Comparisons ===")
    for i in range(min(5, len(bert_texts))):
        print(f"Text: {bert_texts[i]}")
        print(f"Cosine similarity: {similarities[i]:.4f}")
        print(f"BERT cluster: {bert_clusters[i]}, Student cluster: {student_clusters[i]}")
        print()
    
    spark.stop()


if __name__ == "__main__":
    main()