import pandas as pd

# Load the dataset
df = pd.read_json("train.json")

# Step 1: Count how many times each label appears
label_counts = df['label'].value_counts()
print(label_counts)

# Step 2: Keep only labels with at least 1 articles
valid_labels = label_counts[label_counts >= 1].index
print(valid_labels)

# Sanity check
if len(valid_labels) < 250:
    raise ValueError(f"Only found {len(valid_labels)} labels with at least 3 articles — need at least 250.")

# Step 3: Sample 250 labels randomly
sampled_labels = pd.Series(valid_labels).sample(n=250, random_state=42)

# Step 4: From each label, take up to 4 articles
df_sampled = df[df['label'].isin(sampled_labels)].groupby('label').head(4)

# Step 5: Save result
df_sampled.to_json("cluster_sample.json", orient="records", indent=2)


print(f"✅ Done! Sampled {len(df_sampled)} articles from {len(sampled_labels)} clusters.")

# Φόρτωσε το JSON
df = pd.read_json('cluster_sample.json')
# Καθάρισε τα URLs από τα escape slashes (\/ -> /)
def clean_urls(nested_list):
    flat_list = [item for sublist in nested_list for item in sublist]  # flatten
    return [url.replace("\\/", "/") for url in flat_list]


# Ομαδοποίησε ανά label και καθάρισε τα urls
grouped_dict = df.groupby('label')['urls'].apply(
    lambda nested_urls: clean_urls(nested_urls)
).to_dict()









