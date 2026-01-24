# Q3 Sentiment Analysis pipeline: given 5 movie reviews (by the user), it gives the movie a star rating, and a confidence score.

from transformers import pipeline 

reviewer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment") # initializing the pipeline

movie1 = input("Enter movie 1's review: ")
movie2 = input("\nEnter movie 2's review: ")
movie3 = input("\nEnter movie 3's review: ")
movie4 = input("\nEnter movie 4's review: ")
movie5 = input("\nEnter movie 5's review: ")

list_of_movies = [movie1, movie2, movie3, movie4, movie5]

output = reviewer(list_of_movies) # passing the list of movie reviews to the pipeline, and storing the results in a variable

for i, result in enumerate(output):
    print(f"\nMovie Review {i+1}:")
    print(f"  Predicted Label: {result['label']}")
    print(f"  Confidence Score: {result['score']:.4f}")
