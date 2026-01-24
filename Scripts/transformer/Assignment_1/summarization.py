#Q1 Summarization pipeline: summarizing a paragraph inputted by the user.

from transformers import pipeline

summ = pipeline("summarization", model="facebook/bart-large-cnn") # initializing the pipeline

original = input("Enter a paragraph which you want to be summarized: ") # asking for the user's input

text = summ(original, max_length=150, min_length=60, truncation=True) # storing the generated summary in a variable

print("\n\nSummary:", text[0]['summary_text'])
print("\nLength of original text:", len(original.split()))
print("\nLength of summary text:", len(text[0]['summary_text'].split()))