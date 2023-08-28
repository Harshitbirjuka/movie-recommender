from flask import Flask, render_template, request
import pandas as pd
import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("D:\movie\movies_data.csv")

# ... (rest of your code)
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
    
    # Doc2Vec-based recommendation
    inferred_vector = doc2vec_model.infer_vector(data['Tokenized_Features'][idx])
    similar_documents_doc2vec = doc2vec_model.docvecs.most_similar([inferred_vector], topn=num_recommendations + 1)
    similar_indices_doc2vec = [int(idx) for idx, _ in similar_documents_doc2vec if int(idx) != idx]
    
    # TF-IDF-based recommendation
    sim_scores_tfidf = list(enumerate(cosine_sim_tfidf[idx]))
    sim_scores_tfidf = sorted(sim_scores_tfidf, key=lambda x: x[1], reverse=True)
    top_similar_indices_tfidf = [i[0] for i in sim_scores_tfidf[:num_recommendations + 1]]
    
    # Combine and return recommendations (excluding the input movie)
    combined_indices = set(similar_indices_doc2vec).union(set(top_similar_indices_tfidf))
    combined_indices.discard(idx)
    return data['Name'].iloc[list(combined_indices)]

def get_movies_by_director(input_movie):
    director_name = data[data['Name'].str.lower() == input_movie.lower()]['Director'].iloc[0]
    director_movies = data[data['Director'].str.lower() == director_name.lower()]
    director_movies = director_movies[director_movies['Name'].str.lower() != input_movie.lower()]
    return director_movies

def get_top_rated_movies():
    top_rated_movies = data.sort_values(by='Rating', ascending=False).head(10)
    return top_rated_movies




# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_movie = request.form['movie_name'].strip()

        if user_movie.lower() in data['Name'].str.lower().values:
           recommended_movies_series = get_recommendations(user_movie)
           recommended_movies = recommended_movies_series.tolist()  # Convert Series to list
    
           director_movies_df = get_movies_by_director(user_movie)
           director_movies = director_movies_df['Name'].tolist()  # Extract and convert Series to list

           top_rated_movies_df = get_top_rated_movies()
           top_movies = top_rated_movies_df['Name'].tolist()

           # Fetch details of the input movie
           # Fetch details of the input movie
           # Fetch details of the input movie
           input_movie_details = data[data['Name'].str.lower() == user_movie.lower()]

           if not input_movie_details.empty:
              input_movie_details_list = input_movie_details.iloc[0].tolist()
              # Assuming you have the movie rating in the input_movie_details_list
              movie_rating = input_movie_details_list[4]  # Assuming the rating is out of 10
              number_of_stars = int(movie_rating * 0.5)  # Calculate the number of stars (each star represents 0.5)
           else:
              input_movie_details_list = []

            



           return render_template('index.html', recommended_movies=recommended_movies,
                       director_movies=director_movies, user_movie=user_movie, input_movie_details_list=input_movie_details_list, number_of_stars=number_of_stars, top_movies=top_movies)

        else:
            return render_template('index.html', error_message=f"Movie '{user_movie}' not found in the dataset.")
    return render_template('index.html', error_message=None)

if __name__ == '__main__':
    app.run(debug=True)
