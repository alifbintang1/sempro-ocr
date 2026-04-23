import json
import pandas as pd
from io import StringIO

with open("ocr_table_output/json/page_004.json", "r", encoding="utf-8") as f:
    data = json.load(f)

if isinstance(data, list):
    data = data[0]

html = data["table_res_list"][0]["pred_html"]

df = pd.read_html(StringIO(html), header=None)[0]

# tampilkan semua kolom
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 200)

print(df)