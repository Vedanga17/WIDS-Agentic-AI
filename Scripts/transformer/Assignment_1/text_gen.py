#Q2 Text generation pipeline: given an initial phrase by the user, generating a short continuation of that.

from transformers import pipeline

text_gen = pipeline("text-generation", model="gpt2") # initializing the pipeline

starting_line = input("Enter a starting line for the text you want the model to generate: ") 

# storing the result in a variable
output = text_gen(starting_line, max_length=100, num_return_sequences=2, max_new_tokens=50, truncation=True) 

for i, out in enumerate(output):
    print("\nGenerated Text", i+1, ":", out['generated_text'])
