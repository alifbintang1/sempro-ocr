import json
import pandas as pd
from io import StringIO

with open("ocr_table_output/json/page_002.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# handle list
if isinstance(data, list):
    data = data[0]

html = data["table_res_list"][0]["pred_html"]

df = pd.read_html(StringIO(html))[0]
df.to_csv("extracted_table.csv", index=False)  # Save to CSV
print(df.head())