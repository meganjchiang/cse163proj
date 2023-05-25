import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import nltk
import string
from nltk import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from wordcloud import WordCloud

sns.set()


def merge_and_clean(movies: pd.DataFrame,
                    reviews: pd.DataFrame) -> pd.DataFrame:
    # select necessary columns
    movies = movies[['rotten_tomatoes_link', 'movie_title', 'genres']]
    reviews = reviews[['rotten_tomatoes_link', 'critic_name', 'review_score',
                       'review_date', 'review_content']]

    # join two datasets on movie link
    movie_reviews = movies.merge(reviews, on='rotten_tomatoes_link')
    movie_reviews = movie_reviews.reset_index(drop=True)

    # get only rows with numeric scores (no A+, B-, etc.)
    pattern = r'^[A-F][+-]?$'
    not_alpha = ~movie_reviews['review_score'].str.match(pattern)
    movie_reviews = movie_reviews[not_alpha]

    # remove rows with scores w/o denominator (no "/")
    movie_reviews['review_score']
    movie_reviews = \
        movie_reviews[movie_reviews['review_score'].str.contains("/")]

    # convert scores to floats
    movie_reviews['review_score'] = \
        movie_reviews['review_score'].apply(convert_to_float)

    # add column for updated scores (labels of 1-5)
    movie_reviews['score_category'] = \
        movie_reviews['review_score'].apply(convert_to_label)

    # return merged and cleaned dataset
    return movie_reviews


def convert_to_float(fraction: str) -> float:
    """
    Converts a given fraction (string) to a float.
    """
    numbers = fraction.split("/")

    return float(numbers[0]) / float(numbers[1])


def convert_to_label(score: float) -> int:
    """
    Converts a given score (float) to a label (int).
    """
    if score <= 0.2:
        return 1
    elif score <= 0.4:
        return 2
    elif score <= 0.6:
        return 3
    elif score <= 0.8:
        return 4
    else:
        return 5


# First data visualization
def plot_top_20_movies(movie_reviews: pd.DataFrame) -> None:
    # create new plot (otherwise plots will save on top of it)
    plt.figure()

    # get number of reviews for each movie
    num_reviews = movie_reviews.groupby('movie_title')['movie_title'].count()

    # filter for only movies with at least 25 reviews
    at_least_25_reviews = num_reviews[num_reviews >= 25]  # 471 total

    # get movie titles
    movie_titles = pd.Series(at_least_25_reviews.keys())

    # get rows of movies (from movie_reviews) that arein movie_titles series
    has_at_least_25_reviews = movie_reviews['movie_title'].isin(movie_titles)
    movies = movie_reviews[has_at_least_25_reviews]

    # get top 20 movies with highest average ratings
    avg_scores = movies.groupby('movie_title')['score_category'].mean()
    top_20_movies = avg_scores.nlargest(20)

    # convert series to dataframe
    top_20_movies = top_20_movies.reset_index()

    # make bar chart
    top_20_plot = sns.barplot(data=top_20_movies, x='score_category',
                              y='movie_title')

    # add avg rating next to each bar
    # source:
    # matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar_label.html
    top_20_plot.bar_label(top_20_plot.containers[0], fmt='%.3f',
                          fontsize=10, fontweight='bold', padding=4)

    # title and axes labels
    plt.suptitle('Top 20 Highest-Rated Movies* on Rotten Tomatoes (2015-2020)',
                 fontsize='large', fontweight='bold', x=0.2)
    plt.title('*For Movies With At Least 30 Reviews',
              fontsize='small', x=0.46, y=1.02)
    plt.xlabel("Average Review Score", fontweight='bold')
    plt.ylabel("Movie", fontweight='bold')

    # save plot
    plt.savefig('20_highest_rated_movies.png', bbox_inches='tight')


def plot_bottom_20_movies(movie_reviews: pd.DataFrame) -> None:
    # create new plot (otherwise plots will save on top of it)
    plt.figure()

    # get number of reviews for each movie
    num_reviews = movie_reviews.groupby('movie_title')['movie_title'].count()

    # filter for only movies with at least 25 reviews
    at_least_25_reviews = num_reviews[num_reviews >= 25]  # 471 total

    # get movie titles
    movie_titles = pd.Series(at_least_25_reviews.keys())

    # get rows of movies (from movie_reviews) that arein movie_titles series
    has_at_least_25_reviews = movie_reviews['movie_title'].isin(movie_titles)
    movies = movie_reviews[has_at_least_25_reviews]

    # get top 20 movies with lowest average ratings
    avg_scores = movies.groupby('movie_title')['score_category'].mean()
    bottom_20_movies = avg_scores.nsmallest(20)

    # convert series to dataframe
    bottom_20_movies = bottom_20_movies.reset_index()

    # make bar chart
    bottom_20_plot = sns.barplot(data=bottom_20_movies,
                                 x='score_category', y='movie_title')

    # add avg rating next to each bar
    # source:
    # matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar_label.html
    bottom_20_plot.bar_label(bottom_20_plot.containers[0], fmt='%.3f',
                             fontsize=10, fontweight='bold', padding=4)

    # title and axes labels
    plt.suptitle('Top 20 Lowest-Rated Movies* on Rotten Tomatoes (2015-2020)',
                 fontsize='large', fontweight='bold', x=0.35)
    plt.title('*For Movies With At Least 30 Reviews',
              fontsize='small', x=0.65, y=1.02)
    plt.xlabel("Average Review Score", fontweight='bold')
    plt.ylabel("Movie", fontweight='bold')
    plt.xticks(range(6))
    plt.xlim(0, 5)

    # save plot
    plt.savefig('20_lowest_rated_movies.png', bbox_inches='tight')

# Second data visualization
def wordcloud_positive(movie_reviews: pd.DataFrame) -> None:

    # get number of reviews for each movie
    num_reviews = movie_reviews.groupby('movie_title')['movie_title'].count()

    # filter for only movies with at least 25 reviews
    at_least_25_reviews = num_reviews[num_reviews >= 25]  # 471 total
    print(at_least_25_reviews)

    # get movie titles
    movie_titles = pd.Series(at_least_25_reviews.keys())

    # get rows of movies (from movie_reviews) that arein movie_titles series
    has_at_least_25_reviews = movie_reviews['movie_title'].isin(movie_titles)
    movies = movie_reviews[has_at_least_25_reviews]

    # get top 20 movies with highest average ratings
    avg_scores = movies.groupby('movie_title')['score_category'].mean()
    top_20_movies = avg_scores.nlargest(20)

    # convert series to dataframe
    top_20_movies = top_20_movies.reset_index()

    # getting the top 20 movies with its columns 
    top_20_movie_reviews = movie_reviews[movie_reviews['movie_title'].isin(top_20_movies['movie_title'])]
    top_20_movie_reviews = top_20_movie_reviews.reset_index()
    # top_20_movie_reviews.to_csv('testttt.csv')

    # creating subset of positive and negative reviews
    positive_reviews = movie_reviews[movie_reviews['score_category'] > 3]

    # Get the set of stopwords
    stopwords_set = set(stopwords.words('english'))

    # Get the set of punctuations 
    punctuation_set = set(string.punctuation)

    # Additional punctuations 
    add_punc = ["'s", "'nt", "n't"]

    # Remove stopwords and punctuations
    positive_words = []
    for review in positive_reviews['review_content']:
        words = word_tokenize(review)
        cleaned_words = [word.lower() for word in words if word.lower() not in stopwords_set and word not in punctuation_set and not all(char in string.punctuation for char in word) and word not in add_punc]
        positive_words.extend(cleaned_words)

    # Find the top 10 most common words
    positive_freq = FreqDist(positive_words)

    # Convert the word frequency distribution to a string
    positive_freq_txt = ' '.join(positive_freq.keys())

    # Create word cloud for positive subset
    positive_wordcloud = WordCloud(width=1000, height=1000, background_color='white').generate(positive_freq_txt)
    plt.figure(figsize=(12, 12))
    plt.imshow(positive_wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.savefig('positive_wordcloud.png', dpi=300, bbox_inches='tight')
    plt.show()


def wordcloud_negative(movie_reviews: pd.DataFrame) -> None:

    # get number of reviews for each movie
    num_reviews = movie_reviews.groupby('movie_title')['movie_title'].count()

    # filter for only movies with at least 25 reviews
    at_least_25_reviews = num_reviews[num_reviews >= 25]  # 471 total
    print(at_least_25_reviews)
   
    # get movie titles
    movie_titles = pd.Series(at_least_25_reviews.keys())

    # get rows of movies (from movie_reviews) that are in movie_titles series
    has_at_least_25_reviews = movie_reviews['movie_title'].isin(movie_titles)
    movies = movie_reviews[has_at_least_25_reviews]
    #print(movies)

    # get top 20 movies with highest average ratings
    avg_scores = movies.groupby('movie_title')['score_category'].mean()
    top_20_movies = avg_scores.nlargest(20)

    # convert series to dataframe
    top_20_movies = top_20_movies.reset_index()

    # getting the top 20 movies with its columns 
    top_20_movie_reviews = movie_reviews[movie_reviews['movie_title'].isin(top_20_movies['movie_title'])]
    top_20_movie_reviews = top_20_movie_reviews.reset_index()
    # top_20_movie_reviews.to_csv('testttt.csv')

    # creating subset of negative reviews
    negative_reviews = movie_reviews[movie_reviews['score_category'] < 3]

    # Get the set of stopwords
    stopwords_set = set(stopwords.words('english'))

    # Get the set of punctuations 
    punctuation_set = set(string.punctuation)

    # additional punctuations 
    add_punc = ["'s", "'nt", "n't"]
    
    # remove stopwords and punctuations for negative reviews
    negative_words = []
    for review in negative_reviews['review_content']:
        words = word_tokenize(review)
        cleaned_words = [word.lower() for word in words if word.lower() not in stopwords_set and word not in punctuation_set and not all(char in string.punctuation for char in word) and word not in add_punc]
        negative_words.extend(cleaned_words)

    # Find the top 10 most common words
    negative_freq = FreqDist(negative_words)

    # Convert the word frequency distribution to a string
    negative_freq_txt = ' '.join(negative_freq.keys())

    # Create word cloud for negative subset
    negative_wordcloud = WordCloud(width=1000, height=1000, background_color='white').generate(negative_freq_txt)
    plt.figure(figsize=(12, 12))
    plt.imshow(negative_wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.savefig('negative_wordcloud.png', dpi=300, bbox_inches='tight')
    plt.show()

# Third data visualization


def main():
    movies = pd.read_csv('rotten_tomatoes_movies.csv')
    reviews = pd.read_csv('filtered_reviews.csv')

    # join 2 datasets and clean
    movie_reviews = merge_and_clean(movies, reviews)

    # first data visualization (top and bottom 20 movies)
    plot_top_20_movies(movie_reviews)
    plot_bottom_20_movies(movie_reviews)
    wordcloud_positive(movie_reviews)
    wordcloud_negative(movie_reviews)


    # to do:
    # remove rows with null review scores -> DONE
    # need to filter for top critics and/or certain years -> DONE
    # remove alphabetic scores -> DONE
    # convert scores to fractions/floats -> DONE
    # add new column of sentiment score (NLTK) to merged dataset
    # create visualizations
    # create ML model and calculate accuracy/error


if __name__ == '__main__':
    main()
