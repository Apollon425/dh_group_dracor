import pandas as pd
corpus_metadata = pd.read_csv("https://dracor.org/api/corpora/ger/metadata/csv")

print(corpus_metadata.head())
