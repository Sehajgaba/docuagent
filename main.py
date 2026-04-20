
import pdfplumber
import json
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
with pdfplumber.open("RIL-Integrated-Annual-Report-2024-25.pdf") as pdf:
    first_page = pdf.pages[56]
    table_data = first_page.extract_table()

# Create the empty dictionary
vector_database = []

# Loop through the data, skipping the first row (headers)
for row in table_data[1:]:
    key = row[0]
    value = row[1]
    search_text = f"{key}: {value}"
    embedding = model.encode(search_text)
    
    # Store both the text and the math together!
    row_data = {
        "text": search_text,
        "vector": embedding
    }
    vector_database.append(row_data)
# Print the final dictionary
print(len(vector_database))

user_query = "What is the total cash?"
query_vector = model.encode(user_query)

# We will keep track of the highest score and the best text
best_score = 0
best_answer = ""

# Loop through our database to find the match
for item in vector_database:
    # Compare the user's question to the current row's vector
    # util.cos_sim returns a complex tensor, .item() converts it to a standard Python decimal
    similarity_score = util.cos_sim(query_vector, item["vector"]).item()
    
    # If this score is the highest we've seen, save it!
    if similarity_score > best_score:
        best_score = similarity_score
        best_answer = item["text"]

print(f"Top Match: {best_answer}")
print(f"Confidence Score: {best_score}")