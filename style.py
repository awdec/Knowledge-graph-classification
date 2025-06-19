from collections import defaultdict, Counter
import os, csv
from sklearn.feature_extraction.text import CountVectorizer

label_map = {"AAAI": 0, "NeurIPS": 1, "CVPR": 2}
root_dir = "./data_output/cureus"

# 用于存储各类别的所有文本
class_texts = defaultdict(list)

for label_name, label_id in label_map.items():
    subdir = os.path.join(root_dir, label_name)
    for fname in os.listdir(subdir):
        if not fname.endswith(".csv"):
            continue
        with open(os.path.join(subdir, fname), 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)  # 跳过 header
            for row in reader:
                if len(row) >= 3:
                    class_texts[label_id].extend([row[0], row[1], row[2]])

# 使用 CountVectorizer 分析词频
vectorizer = CountVectorizer(stop_words='english', max_features=50)
for label_id, texts in class_texts.items():
    X = vectorizer.fit_transform(texts)
    total_counts = X.sum(axis=0).A1
    vocab = vectorizer.get_feature_names_out()
    word_freq = sorted(zip(vocab, total_counts), key=lambda x: -x[1])
    print(f"\nTop words for class {label_id}:")
    for word, freq in word_freq[:15]:
        print(f"  {word}: {freq}")
