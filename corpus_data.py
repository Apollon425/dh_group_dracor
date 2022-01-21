from re import T
import pandas as pd
import requests
import urllib.request, json 

#CORPUS_METADATA = pd.read_csv("https://dracor.org/api/corpora/ger/metadata/csv")
CORPUS_METADATA = pd.read_csv("data_files//corpus_metadata.csv")

API_INFO = "https://dracor.org/api/corpora/ger"
PLAY_INFO = "https://dracor.org/api/corpora/ger/play/gottsched-der-sterbende-cato/metrics"
PLAY_TEXT = "https://dracor.org/api/corpora/ger/play/gottsched-der-sterbende-cato/spoken-text"

# with urllib.request.urlopen(PLAY_TEXT) as url:
#     data = json.loads(url.read().decode())
#     print(data)


with urllib.request.urlopen(PLAY_TEXT) as response:
   text = response.read()
   encoding = response.headers.get_content_charset('utf-8')
   text = text.decode(encoding)
   print(text)
   with open('der_sterbende_cato.json', 'w', encoding='utf-8') as f:
    json.dump(text, f, ensure_ascii=False, indent=4)



#  DATA_FILES_PATH = "C://Users//richa//corpus_metadata.csv"

#print(CORPUS_METADATA.head())

# for col in CORPUS_METADATA:
#     print(col)

#print(list(CORPUS_METADATA))

def write_csv_to_file(df: pd.DataFrame , path: str, separator: str, encoding: str) -> None:
    df.to_csv(path, sep=separator, encoding=encoding, index=False)


# if __name__ == "__main__":
#     pass   
