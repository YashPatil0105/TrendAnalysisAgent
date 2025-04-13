# # File: TrendAnalysisAgent/src/topic_model.py

# from bertopic import BERTopic
# import umap
# import hdbscan

# def build_topic_model(documents):
#     """
#     Train a BERTopic model on the provided documents.
#     Using custom UMAP and HDBSCAN parameters suited for a 100+ document dataset.
#     """
#     # Configure UMAP with moderate neighbors (for a larger dataset)
#     umap_model = umap.UMAP(
#         n_neighbors=15,
#         n_components=2,
#         min_dist=0.0,
#         metric='cosine',
#         random_state=42
#     )
#     # Configure HDBSCAN; adjust min_cluster_size to avoid excessively fragmented clusters
#     hdbscan_model = hdbscan.HDBSCAN(
#         min_cluster_size=5,
#         min_samples=1,
#         metric='euclidean',
#         prediction_data=True
#     )
#     topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, verbose=True)
#     topics, probs = topic_model.fit_transform(documents)
#     return topic_model, topics, probs

# def visualize_topics(topic_model):
#     """
#     Return the interactive visualization of topics.
#     (Call .show() or .write_html() on the returned object in your main app.)
#     """
#     return topic_model.visualize_topics()

# def print_topic_info(topic_model):
#     """
#     Print out user-friendly information for each topic.
#     Outlier documents (topic -1) will be labeled as 'Unclassified/Miscellaneous.'
#     """
#     topic_freq = topic_model.get_topic_freq()
#     print("Topic Frequencies:")
#     print(topic_freq)

#     unique_topics = topic_freq["Topic"].tolist()
#     for topic in unique_topics:
#         if topic == -1:
#             print("Topic -1: Unclassified/Miscellaneous")
#             continue
#         topic_words = topic_model.get_topic(topic)
#         if not topic_words or topic_words == False:
#             print(f"Topic {topic} has no meaningful words.")
#         else:
#             # Generate a simple label from the top 5 words
#             top_words = [word for word, _ in topic_words[:5]]
#             print(f"Topic {topic} - Top Keywords: {', '.join(top_words)}")

# File: TrendAnalysisAgent/src/topic_model.py

from bertopic import BERTopic
import umap
import hdbscan

def build_topic_model(documents, min_cluster_size=5):
    """
    Train a BERTopic model on the provided documents using custom parameters.
    Optionally, you could add a `domain_filter` parameter to pre-filter documents.
    """
    # UMAP configuration (adjusted for larger datasets)
    umap_model = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    # HDBSCAN configuration to suit a dataset with 100+ documents
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric='euclidean',
        prediction_data=True
    )
    topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, verbose=True)
    topics, probs = topic_model.fit_transform(documents)
    return topic_model, topics, probs

def generate_topic_label(topic_words):
    """
    Generate a friendly label for a topic given its top keywords.
    For a real system, you might use more advanced NLP or a lookup table.
    """
    if not topic_words or topic_words is False:
        return "Unclassified/Miscellaneous"
    # For simplicity, join the top 3 words as the label.
    top_words = [word for word, _ in topic_words[:3]]
    label = " ".join(top_words).title()
    return label

def print_topic_info(topic_model):
    """
    Print out user-friendly information for each topic.
    Outlier documents (topic -1) are labeled as 'Unclassified/Miscellaneous'.
    """
    topic_freq = topic_model.get_topic_freq()
    print("Topic Frequencies:")
    print(topic_freq)
    
    unique_topics = topic_freq["Topic"].tolist()
    for topic in unique_topics:
        if topic == -1:
            label = "Unclassified/Miscellaneous"
        else:
            topic_words = topic_model.get_topic(topic)
            label = generate_topic_label(topic_words)
        count = topic_freq[topic_freq["Topic"] == topic]["Count"].values[0]
        print(f"Topic {topic} ({label}) - {count} documents")
        
def get_topic_summary(topic_model):
    """
    Create a JSON-friendly summary of topics, including friendly labels,
    top keywords, and document counts.
    """
    topic_freq = topic_model.get_topic_freq()
    summary = []
    unique_topics = topic_freq["Topic"].tolist()
    for topic in unique_topics:
        if topic == -1:
            label = "Unclassified/Miscellaneous"
            top_keywords = []
        else:
            topic_words = topic_model.get_topic(topic)
            label = generate_topic_label(topic_words)
            # Prepare list of top keywords with their weights
            top_keywords = [{"keyword": word, "weight": float(weight)} for word, weight in topic_words[:5]]
        count = topic_freq[topic_freq["Topic"] == topic]["Count"].values[0]
        summary.append({
            "topic_id": topic,
            "label": label,
            "top_keywords": top_keywords,
            "document_count": int(count)
        })
    return summary

def visualize_topics_interactive(topic_model):
    """
    Return the interactive topic visualization.
    """
    return topic_model.visualize_topics()
