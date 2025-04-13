import json
from faker import Faker
import random

fake = Faker()
topics = [
    "Natural Language Processing", "Computer Vision", "Reinforcement Learning",
    "Quantum Computing", "Healthcare AI", "Federated Learning", "Graph Neural Networks",
    "Climate Modeling", "Anomaly Detection", "Generative Models", "Bias in AI",
    "Multi-Agent Systems", "Edge Computing", "Sustainable AI", "Autonomous Vehicles"
]

sample_data = []

for i in range(120):  # Generate 120 mock entries
    topic = random.choice(topics)
    title = f"{topic}: {fake.sentence(nb_words=6).rstrip('.')}"
    summary = fake.paragraph(nb_sentences=4)
    sample_data.append({
        "title": title,
        "summary": summary
    })

# Save the JSON file
with open("data/summaries.json", "w", encoding="utf-8") as f:
    json.dump(sample_data, f, indent=2, ensure_ascii=False)

print("✅ Generated 120 mock summaries in data/summaries.json")
