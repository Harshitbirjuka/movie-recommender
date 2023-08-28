import pandas as pd
import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the dataset
data = pd.read_csv("D:\movie\movies_data.csv")

# Convert text data to lowercase
data['Genre'] = data['Genre'].str.lower()
data['Actor 1'] = data['Actor 1'].str.lower()
data['Actor 2'] = data['Actor 2'].str.lower()
data['Actor 3'] = data['Actor 3'].str.lower()
data['Director'] = data['Director'].str.lower()

# Combine relevant features into a single column
data['Features'] = data['Genre'] + ' ' + data['Actor 1'] + ' ' + data['Actor 2'] + ' ' + data['Actor 3'] + ' ' + data['Director']

# Tokenize the features
data['Tokenized_Features'] = data['Features'].apply(nltk.word_tokenize)

# Train a Doc2Vec model
tagged_data = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(data['Tokenized_Features'])]
doc2vec_model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4, epochs=20)
doc2vec_model.build_vocab(tagged_data)
doc2vec_model.train(tagged_data, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the data
tfidf_matrix = tfidf_vectorizer.fit_transform(data['Features'])

# Calculate the cosine similarity matrix using TF-IDF
cosine_sim_tfidf = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_recommendations(movie_title, doc2vec_model=doc2vec_model, cosine_sim_tfidf=cosine_sim_tfidf, num_recommendations=10):
    idx = data[data['Name'].str.lower() == movie_title.lower()].index[0]
    
    # Adjust weights for features
    weight_genre = 2.0
    weight_actor1 = 1.5
    weight_other = 1.0
    
    # Doc2Vec-based recommendation
    inferred_vector = doc2vec_model.infer_vector(data['Tokenized_Features'][idx])
    similar_documents_doc2vec = doc2vec_model.docvecs.most_similar([inferred_vector], topn=num_recommendations + 1)
    similar_indices_doc2vec = [int(idx) for idx, _ in similar_documents_doc2vec if int(idx) != idx]
    
    # TF-IDF-based recommendation with adjusted weights
    sim_scores_tfidf = list(enumerate(cosine_sim_tfidf[idx]))
    sim_scores_tfidf = sorted(sim_scores_tfidf, key=lambda x: (weight_genre * data['Genre'][x[0]] + 
                                                               weight_actor1 * data['Actor 1'][x[0]] +
                                                               weight_other * x[1]), reverse=True)
    top_similar_indices_tfidf = [i[0] for i in sim_scores_tfidf[:num_recommendations + 1]]
    
    # Combine and return recommendations (excluding the input movie)
    combined_indices = set(similar_indices_doc2vec).union(set(top_similar_indices_tfidf))
    combined_indices.discard(idx)
    return data['Name'].iloc[list(combined_indices)]

# Rest of your code remains the same

def get_top_rated_movies():
    top_rated_movies = data.sort_values(by='Rating', ascending=False).head(10)
    return top_rated_movies


def get_movies_by_director(input_movie):
    director_name = data[data['Name'].str.lower() == input_movie.lower()]['Director'].iloc[0]
    director_movies = data[data['Director'].str.lower() == director_name.lower()]
    director_movies = director_movies[director_movies['Name'].str.lower() != input_movie.lower()]
    return director_movies
user_movie = input("Enter a movie name: ").strip()

if user_movie.lower() in data['Name'].str.lower().values:
    recommended_movies = get_recommendations(user_movie)
    print(f"Recommended movies based on '{user_movie}':")
    print(recommended_movies)
    
    
    director_movies = get_movies_by_director(user_movie)
    if not director_movies.empty:
        print(f"Movies by the same director as '{user_movie}':")
        print(director_movies['Name'])
    else:
        print(f"No other movies by the same director as '{user_movie}' found.")
else:
    print(f"Movie '{user_movie}' not found in the dataset.")

    # Get top 10 rated movies
top_rated_movies = get_top_rated_movies()
print("Top 10 Rated Movies:")
print(top_rated_movies[['Name', 'Rating']])

