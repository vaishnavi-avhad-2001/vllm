from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the T5 tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

text_to_summarize = """summarize: Rahul Gandhi has replied to Goa CM Manohar Parrikar's letter, 
which accused the Congress President of using his "visit to an ailing man for political gains". 
"He's under immense pressure from the PM after our meeting and needs to demonstrate his loyalty by attacking me," 
Gandhi wrote in his letter. Parrikar had clarified he didn't discuss Rafale deal with Rahul."""

# Tokenize and generate the summarized text
input_ids = tokenizer.encode(text_to_summarize, return_tensors="pt", max_length=1024, truncation=True)
summary_ids = model.generate(input_ids)

# Decode the generated summary   
generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Print the generated summary
print("Generated summary:", generated_summary)
