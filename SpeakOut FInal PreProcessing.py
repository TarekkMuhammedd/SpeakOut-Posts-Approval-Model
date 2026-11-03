import pandas as pd

def GetValuesCounts(data):
    columns = [col for col in data.columns]
    for col in columns:
        print(f"--- {col} ---")
        print(data[col].value_counts(), "\n")

data = pd.read_json(r"D:\Original SpeakOut Dataset.json")

data.drop_duplicates(inplace=True)

output_path = "Final_SpeakOut_Posts_Dataset.json"
data.to_json(output_path, orient='records', indent=4, force_ascii=False)
