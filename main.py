import pdfplumber
import re
from sentence_transformers import SentenceTransformer, util

print("Loading AI Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Extracting PDF...")
with pdfplumber.open("RIL-Integrated-Annual-Report-2024-25.pdf") as pdf:
    target_page = pdf.pages[56] 
    raw_text = target_page.extract_text()

vector_database = []

# Regex Pattern: 
# 1. Finds words/spaces: ([A-Za-z\s\-\&]+?)
# 2. Ignores optional 1-or-2 digit note numbers: (?:\d{1,2}\s+)?
# 3. Grabs Indian-formatted numbers with commas: (\d{1,3}(?:,\d{2,3})+)
pattern = re.compile(r'([A-Za-z\s\-\&]+?)\s+(?:\d{1,2}\s+)?(\d{1,3}(?:,\d{2,3})+)')

# Split text into lines and parse
for line in raw_text.split('\n'):
    matches = pattern.findall(line)
    
    # Because of the two-pane mash, there might be two matches per line!
    for match in matches:
        key = match[0].strip()
        value = match[1].strip()
        
        # Filter out table headers and garbage
        if len(key) > 3 and "As at" not in key and "Total" not in key:
            search_text = f"{key} is {value} crore"
            
            vector_database.append({
                "text": search_text,
                "vector": model.encode(search_text)
            })

print(f"Successfully cleaned and vectorized {len(vector_database)} financial line items.\n")

# --- THE AI SEARCH ENGINE ---
user_query = "How much is the Equity Share capital?"
print(f"User Question: '{user_query}'\n")

query_vector = model.encode(user_query)

best_score = 0
best_answer = ""

for item in vector_database:
    similarity_score = util.cos_sim(query_vector, item["vector"]).item()
    
    if similarity_score > best_score:
        best_score = similarity_score
        best_answer = item["text"]

print("--- AI SEARCH RESULTS ---")
print(f"Top Match: {best_answer}")
print(f"Confidence Score: {round(best_score * 100, 2)}%")