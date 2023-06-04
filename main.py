"""
Megan Chiang, Michelle Kim, Jasmine Wong
CSE 163 AD/Group 055
Implements functions that visualize and predict
movie review ratings from Rotten Tomatoes.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
from nltk import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


sns.set()


def merge_and_clean(movies: pd.DataFrame,
                    reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe that combines the two given datasets
    (movies and reviews), excluding rows with letter scores
    or scores without a denominator.
    """
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
        movie_reviews['review_score'].apply(_convert_to_float)

    # add column for updated scores (labels of 1-5)
    movie_reviews['score_category'] = \
        movie_reviews['review_score'].apply(_convert_to_label)

    # return merged and cleaned dataset
    return movie_reviews


def _convert_to_float(fraction: str) -> float:
    """
    Private helper function for merge_and_clean()
    Converts a given fraction (as a string type) to its float equivalent.
    """
    numbers = fraction.split("/")

    return float(numbers[0]) / float(numbers[1])


def _convert_to_label(score: float) -> int:
    """
    Private helper function for merge_and_clean()
    Converts a given score (float between 0 and 1) to
    a label (int from 1 to 5).
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
def plot_top_20_movies(movie_reviews: pd.DataFrame,
                       img_file_name: str) -> None:
    """
    Takes the given dataset and plots a bar chart of the 20 best
    movies based on average critic rating. Only considers movies
    with at least 25 ratings. If multiple movies have the same
    average rating, they will be displayed in the order they
    appear in the dataset.
    """
    # create new plot (otherwise plots will save on top of it)
    plt.figure()

    # get top 20 movies (from movies with at least 25 reviews)
    top_20_movies = _get_top_20_movies(movie_reviews)

    # make bar chart
    top_20_plot = sns.barplot(data=top_20_movies, x='score_category',
                              y='movie_title')

    # add avg rating next to each bar
    # source:
    # matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar_label.html
    top_20_plot.bar_label(top_20_plot.containers[0], fmt='%.3f',
                          fontsize=10, fontweight='bold', padding=4)

    # title and axes labels
    plt.suptitle('Top 20 Highest-Rated Movies* on Rotten Tomatoes ' +
                 '(Reviews from 2015-2020)',
                 fontsize='large', fontweight='bold', x=0.2)
    plt.title('*For Movies With At Least 25 Reviews',
              fontsize='small', x=0.58, y=1.02)
    plt.xlabel("Average Review Score", fontweight='bold')
    plt.ylabel("Movie", fontweight='bold')
    plt.xticks(range(6))

    # save plot
    plt.savefig(img_file_name, bbox_inches='tight')


def plot_bottom_20_movies(movie_reviews: pd.DataFrame,
                          img_file_name: str) -> None:
    """
    Takes the given dataset and plots a bar chart of the 20 worst
    movies based on average critic rating. Only considers movies
    with at least 25 ratings. If multiple movies have the same
    average rating, they will be displayed in the order they
    appear in the dataset.
    """
    # create new plot (otherwise plots will save on top of it)
    plt.figure()

    # get bottom 20 movies (from movies with at least 25 reviews)
    bottom_20_movies = _get_bottom_20_movies(movie_reviews)

    # make bar chart
    bottom_20_plot = sns.barplot(data=bottom_20_movies,
                                 x='score_category', y='movie_title')

    # add avg rating next to each bar
    # source:
    # matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar_label.html
    bottom_20_plot.bar_label(bottom_20_plot.containers[0], fmt='%.3f',
                             fontsize=10, fontweight='bold', padding=4)

    # title and axes labels
    plt.suptitle('Top 20 Lowest-Rated Movies* on Rotten Tomatoes ' +
                 '(Reviews from 2015-2020)',
                 fontsize='large', fontweight='bold', x=0.35)
    plt.title('*For Movies With At Least 25 Reviews',
              fontsize='small', x=0.78, y=1.02)
    plt.xlabel("Average Review Score", fontweight='bold')
    plt.ylabel("Movie", fontweight='bold')
    plt.xticks(range(6))
    plt.xlim(0, 5)

    # save plot
    plt.savefig(img_file_name, bbox_inches='tight')


# Second data visualization
def wordcloud_positive(movie_reviews: pd.DataFrame,
                       img_file_name: str) -> None:
    '''
    Takes the given dataset and generates a word cloud
    containing the most frequent words in positive reviews.
    '''
    # Get top 20 movies
    top_20_movies = _get_top_20_movies(movie_reviews)

    # Get the top 20 movies with its columns
    top_20_movie_reviews = movie_reviews[
            movie_reviews['movie_title'].isin(top_20_movies['movie_title'])]
    top_20_movie_reviews = top_20_movie_reviews.reset_index()

    # Get the set of stopwords
    stopwords_set = set(stopwords.words('english'))

    # Get the set of punctuations
    punctuation_set = set(string.punctuation)

    # Get the dditional punctuations
    add_punc = ["'s", "'nt", "n't"]

    # Initialize an empty list to store cleaned words from positive reviews
    positive_words = []

    # Iterate through each review and remove stopwords and punctuations
    for review in top_20_movie_reviews['review_content']:
        words = word_tokenize(review)
        cleaned_words = [word.lower() for word in words if (
                         word.lower() not in stopwords_set and
                         word not in punctuation_set and
                         not all(char in string.punctuation for char in word)
                         and word not in add_punc)]
        positive_words.extend(cleaned_words)

    # Find the frequency of positve words
    positive_freq = FreqDist(positive_words)

    # Find the top 100 most common words in positive reviews
    top_100_positive_words = positive_freq.most_common(100)

    # Extract the words and their frequencies
    positive_words, frequency = zip(*top_100_positive_words)

    # Convert the word frequency distribution to a string
    positive_freq_txt = ' '.join(positive_words)

    # Create word cloud for positive subset
    positive_wordcloud = (WordCloud(width=1000,
                                    height=1000,
                                    background_color='white')
                          .generate(positive_freq_txt))
    plt.figure(figsize=(12, 12))
    plt.imshow(positive_wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.savefig(img_file_name, dpi=300, bbox_inches='tight')


def wordcloud_negative(movie_reviews: pd.DataFrame,
                       img_file_name: str) -> None:
    '''
    Takes the given dataset and generates a word cloud
    containing the most frequent words in negative reviews.
    '''
    # Get bottom 20 movies
    bottom_20_movies = _get_bottom_20_movies(movie_reviews)

    # Get the bottom 20 movies with its columns
    bottom_20_movie_reviews = movie_reviews[
            movie_reviews['movie_title'].isin(bottom_20_movies['movie_title'])]
    bottom_20_movie_reviews = bottom_20_movie_reviews.reset_index()

    # Get the set of stopwords
    stopwords_set = set(stopwords.words('english'))

    # Get the set of punctuations
    punctuation_set = set(string.punctuation)

    # Get additional punctuations
    add_punc = ["'s", "'nt", "n't"]

    # Initialize an empty list to store cleaned words from negative reviews
    negative_words = []

    # Iterate through each review and remove stopwords and punctuations
    for review in bottom_20_movie_reviews['review_content']:
        words = word_tokenize(review)
        cleaned_words = [word.lower() for word in words if (
                         word.lower() not in stopwords_set and
                         word not in punctuation_set and
                         not all(char in string.punctuation for char in word)
                         and word not in add_punc)]
        negative_words.extend(cleaned_words)

    # Find the frequency of negative words
    negative_freq = FreqDist(negative_words)

    # Find the top 100 most common words in negative reviews
    top_100_negative_words = negative_freq.most_common(100)

    # Extract the words and their frequencies
    negative_words, frequency = zip(*top_100_negative_words)

    # Convert the word frequency distribution to a string
    negative_freq_txt = ' '.join(negative_words)

    # Create word cloud for negative subset
    negative_wordcloud = (WordCloud(width=1000,
                                    height=1000,
                                    background_color='white')
                          .generate(negative_freq_txt))
    plt.figure(figsize=(12, 12))
    plt.imshow(negative_wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.savefig(img_file_name, dpi=300, bbox_inches='tight')


def _get_movies_at_least_25_reviews(movie_reviews:
                                    pd.DataFrame) -> pd.DataFrame:
    """
    Private helper function for _get_top_20_movies() and
    _get_bottom_20_movies(). Returns a dataframe containing only
    movies that have at least 25 reviews.
    """
    # get number of reviews for each movie
    num_reviews = movie_reviews.groupby('movie_title')['movie_title'].count()

    # filter for only movies with at least 25 reviews
    at_least_25_reviews = num_reviews[num_reviews >= 25]  # 471 total

    # get movie titles
    movie_titles = pd.Series(at_least_25_reviews.keys())

    # get rows of movies (from movie_reviews) that are in movie_titles series
    has_at_least_25_reviews = movie_reviews['movie_title'].isin(movie_titles)

    return movie_reviews[has_at_least_25_reviews]


def _get_top_20_movies(movie_reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Private helper function for plot_top_20_movies() and wordcloud_positive().
    Returns a dataframe containing the 20 best movies (which have at least
    25 reviews) and their average critic ratings.
    """
    movies = _get_movies_at_least_25_reviews(movie_reviews)

    # get top 20 movies with highest average ratings
    avg_scores = movies.groupby('movie_title')['score_category'].mean()
    top_20_movies = avg_scores.nlargest(20)

    # return top_20_movies as dataframe
    return top_20_movies.reset_index()


def _get_bottom_20_movies(movie_reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Private helper function for plot_bottom_20_movies() and
    wordcloud_negative(). Returns a dataframe containing the
    20 worst movies (which have at least 25 reviews) and their
    average critic ratings.
    """
    movies = _get_movies_at_least_25_reviews(movie_reviews)

    # get top 20 movies with highest average ratings
    avg_scores = movies.groupby('movie_title')['score_category'].mean()
    top_20_movies = avg_scores.nsmallest(20)

    # return top_20_movies as dataframe
    return top_20_movies.reset_index()


# Third data visualization
def word_count_vs_review_score(movie_reviews: pd.DataFrame,
                               img1_file_name: str,
                               img2_file_name: str) -> None:
    """
    Takes in the given dataset and generates a data visualization
    in the form of a scatterplot that compares the average review
    score to average review word count per movie. Also generates
    a distribution subplot to show the distributions of each
    variable, word count and review score.
    """
    # create new plot (otherwise plots will save on top of it)
    plt.figure()

    # get average review scores
    avg_scores = movie_reviews.groupby('movie_title')['review_score'].mean()

    # get average review word counts
    # source:
    # https://stackoverflow.com/questions/65677906/average-word-length-of-a-column-using-python
    avg_word_counts = (movie_reviews.groupby('movie_title')['review_content']
                       .apply(
                        lambda x: sum(len(str(review).split()) for review in x)
                        / float(len(x))
                      ))

    # imputting the points into a dataframe to be used to plot
    data = pd.DataFrame({'Average Review Score': avg_scores,
                         'Average Word Count': avg_word_counts})

    # create scatter plot
    sns.scatterplot(x='Average Word Count', y='Average Review Score',
                    data=data)

    # plot labels
    plt.xlabel('Average Word Count')
    plt.ylabel('Average Review Score')
    plt.title('Average Word Count vs. Average Review Score per Movie')

    # save scatter plot to image
    plt.savefig(img1_file_name, bbox_inches='tight')

    # create a figure size for distribution visualization
    plt.figure(figsize=[10, 5])

    # distribution of word counts
    plt.subplot(1, 2, 1)
    plt.hist(avg_word_counts, bins=10)
    plt.xlabel('Average Word Count')
    plt.ylabel('Frequency')
    plt.title('Distribution of Average Review Word Counts')

    # distribution of review scores
    plt.subplot(1, 2, 2)
    plt.hist(avg_scores, bins=10)
    plt.xlabel('Average Review Score')
    plt.ylabel('Frequency')
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.title('Distribution of Average Review Scores')

    # save distribution plots to image
    plt.savefig(img2_file_name)


# sources:
# https://www.mygreatlearning.com/blog/bag-of-words/
# https://sahanidharmendra19.medium.com/understanding-countvectorizer-tfidftransformer-tfidfvectorizer-with-calculation-7d509efd470f
# https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html
# https://seaborn.pydata.org/generated/seaborn.heatmap.html
def fit_and_predict(movie_reviews: pd.DataFrame, img_file_name: str) -> str:
    """
    Fits a logistic regression model to predict the review score
    category based on the review content, using a bag of words approach,
    from the given movie_reviews data set and also generates a heat map
    visualization of the model to show accuracy vs predicted. Returns
    the f-1 scores and accuracy score of the model in string format.
    """
    # Extract the columns
    X = movie_reviews['review_content']
    y = movie_reviews['score_category']

    # Split the data into training and testing sets by 80/20
    # and the random state by 42
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)

    # Create an instance of the CountVectorizer to
    # create bag of words
    count_vectorizer = CountVectorizer()

    # Fit and transform the training data with the count_vectorizer
    X_train_counts = count_vectorizer.fit_transform(X_train)

    # Create an instance of the TfidfTransformer to rank the words
    tfidf_transformer = TfidfTransformer()

    # Fit and transform the training data with TF-IDF instance
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # Transform the testing data using the same vectorizers
    X_test_counts = count_vectorizer.transform(X_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    # Create an instance of the logistic regression model
    model = LogisticRegression(max_iter=1000,
                               multi_class='multinomial',
                               solver='lbfgs')

    # Train the model using the vectorized training data and corresponding
    # labels
    model.fit(X_train_tfidf, y_train)

    # Predict the score categories with the testing data
    y_pred = model.predict(X_test_tfidf)

    # calculate f1 scores & accuracy score
    f1_scores = f1_score(y_test, y_pred, average=None)
    accuracy = accuracy_score(y_test, y_pred)

    # get the unique labels from the 'score_category' column
    labels = movie_reviews['score_category'].unique()

    # create a dictionary to map labels to F-1 scores
    f1_scores_dict = dict(zip(labels, f1_scores))

    # get a string representation of the F-1 scores
    f1_scores_str = ', '.join([f'{label}: {score}' for label,
                               score in f1_scores_dict.items()])

    # generate a confusion matrix using the actual and predicted values
    cm = pd.crosstab(y_test,
                     y_pred,
                     rownames=['Actual'],
                     colnames=['Predicted'])

    # generate a heatmap using the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title("Confusion Matrix - Actual vs. Predicted")
    plt.savefig(img_file_name)

    # return the f-1 scores and accuracy in string format
    return f"F-1 Scores: {f1_scores_str}, Accuracy: {accuracy}"


def main():
    movies = pd.read_csv('rotten_tomatoes_movies.csv')
    reviews = pd.read_csv('filtered_reviews.csv')

    # join 2 datasets and clean
    movie_reviews = merge_and_clean(movies, reviews)

    plot_top_20_movies(movie_reviews, '20_highest_rated_movies.png')
    plot_bottom_20_movies(movie_reviews, '20_lowest_rated_movies.png')
    wordcloud_positive(movie_reviews, 'positive_wordcloud.png')
    wordcloud_negative(movie_reviews, 'negative_wordcloud.png')
    word_count_vs_review_score(movie_reviews, 'ave_word_count_vs_score.png',
                               'distributions_count_score.png')
    print(fit_and_predict(movie_reviews, 'machine_learning.png'))


if __name__ == '__main__':
    main()
