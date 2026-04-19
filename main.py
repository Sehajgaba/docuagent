
import pdfplumber
import json
from sentence_transformers import SentenceTransformer
with pdfplumber.open("RIL-Integrated-Annual-Report-2024-25.pdf") as pdf:
    first_page = pdf.pages[0]
    table_data = first_page.extract_table()

# Create the empty dictionary
financial_json = {}

# Loop through the data, skipping the first row (headers)
for row in table_data[1:]:
    # Notice the indentation. This code lives INSIDE the loop.
    key = row[0]
    value = row[1]
    financial_json[key] = value

# Print the final dictionary
print(financial_json)