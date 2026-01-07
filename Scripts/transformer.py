from transformers import pipeline

#Q1
summ = pipeline("summarization", model="facebook/bart-large-cnn")

original = """India's set the pace in women's cricket in 2025, consistently dictating terms rather than reacting. Their intent was underlined early with a record-breaking 435/5 against Ireland Women at Rajkot in January, which is the highest ODI total by India and among the top five in women's cricket history. India\'s batting depth, especially beyond the top order, kept constant pressure 
on teams, making 300-plus scores normal rather than ambitious and forcing opponents to change their tactics. India scored nine 
300-plus totals in 2025, which is the most by any team in an ODI calendar year."""

text = summ(original, max_length=120, min_length=40, truncation=True)

print("Summary:", text[0]['summary_text'])
print("\nLength of original text:", len(original.split()))
print("\nLength of summary text:", len(text[0]['summary_text'].split()))

#Q2
text_gen = pipeline("text-generation", model="gpt2")
output = text_gen("The AI boom has led to a shift", max_length=70, num_return_sequences=2, max_new_tokens=50, truncation=True)
for i, out in enumerate(output):
    print("\nGenerated Text", i+1, ":", out['generated_text'])


#Q3
reviewer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

movie1 = '''This classic French short is a visually inventive and wondrous portrait of a young boy's friendship with a sentient red balloon 
floating through Paris. Its simple, wordless narrative beautifully captures themes of innocence and companionship, making it a poignant and 
heartwarming experience for all ages.'''
movie2 = '''Set in 1957, this Christmas short film masterfully conveys the essence of hope and sacrifice during a time still scarred by World War 
II. With commendable performances by Ben Radcliffe and John Travolta, the heartwarming narrative effectively tugs at the heartstrings and checks 
all the boxes for a captivating, meaningful holiday viewing experience.'''
movie3 = '''This Academy Award-winning animation tells a beautiful and touching story of the enduring love between a girl and her father. The 
stunning visuals and emotional music underscore the narrative as the daughter revisits the last place she saw him throughout her life. It's an 
unforgettable film that can be watched repeatedly.'''
movie4 = '''A slightly whimsical and charming Oscar-winning short about two estranged Northern Irish brothers who must come to terms with each 
other and their mother's death while attempting to complete her bucket list. The film balances lighthearted moments with poignant themes of 
familial duty and loss.'''
movie5 = '''This superhero spin-off takes a bold, non-traditional approach, leaning more into 2000s-style suspense than modern spectacle. Dakota 
Johnson brings a unique, deadpan energy to the role of a paramedic discovering psychic abilities. While its unconventional pacing and script 
choices were polarizing, the film is notable for its gritty, grounded tone and its attempt to build a story around mental powers rather than just 
physical combat. It serves as a curious experiment in genre-bending that has already begun to find a following as an unintentional "cult" piece.'''

list_of_movies = [movie1, movie2, movie3, movie4, movie5]

output = reviewer(list_of_movies)

for i, result in enumerate(output):
    print(f"\nMovie Review {i+1}:")
    print(f"  Predicted Label: {result['label']}")
    print(f"  Confidence Score: {result['score']:.4f}")










