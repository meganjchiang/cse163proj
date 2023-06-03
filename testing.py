"""
Megan Chiang, Michelle Kim, Jasmine Wong
CSE 163 AD/Group 055
Tests the functions that visualize and predict
movie review ratings from Rotten Tomatoes.
"""

import main as m

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
from sklearn.metrics import accuracy_score

sns.set()


def test_plot_top_20_movies(movie_reviews: pd.DataFrame) -> None:
    """
    Tests the plot_top_20_movies() function.
    """
    # create new plot (otherwise plots will save on top of it)
    plt.figure()

    # get top 20 movies (from movies with at least 25 reviews)
    top_20_movies = m._get_top_20_movies(movie_reviews)

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
    plt.savefig('test_20_highest_rated_movies.png', bbox_inches='tight')


def test_plot_bottom_20_movies(movie_reviews: pd.DataFrame) -> None:
    """
    Tests the plot_bottom_20_movies() function.
    """
    # create new plot (otherwise plots will save on top of it)
    plt.figure()

    # get bottom 20 movies (from movies with at least 25 reviews)
    bottom_20_movies = m._get_bottom_20_movies(movie_reviews)

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
    plt.savefig('test_20_lowest_rated_movies.png', bbox_inches='tight')


def test_word_count_vs_review_score(movie_reviews: pd.DataFrame) -> None:
    """
    Tests the word_count_vs_review_score() function.
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

    # save plot to image
    plt.savefig('test_ave_word_count_vs_score.png', bbox_inches='tight')


def test_fit_and_predict(movie_reviews: pd.DataFrame) -> float:
    """
    Tests the fit_and_predict() function.
    """
    X = movie_reviews['review_content']
    y = movie_reviews['score_category']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    # Create an instance of the CountVectorizer
    count_vectorizer = CountVectorizer()

    # Fit and transform the training data
    X_train_counts = count_vectorizer.fit_transform(X_train)

    # Create an instance of the TfidfTransformer
    tfidf_transformer = TfidfTransformer()

    # Fit and transform the training data with TF-IDF
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

    # Predict the score categories for the testing data
    y_pred = model.predict(X_test_tfidf)

    # Create a confusion matrix
    cm = pd.crosstab(y_test,
                     y_pred,
                     rownames=['Actual'],
                     colnames=['Predicted'])

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title("Confusion Matrix - Actual vs. Predicted")
    plt.savefig("test_machine_learning.png")

    # Calculate the accuracy of the model
    return accuracy_score(y_test, y_pred)


def main():
    test_movie_reviews = pd.read_csv('40_movies.csv')

    test_plot_top_20_movies(test_movie_reviews)
    test_plot_bottom_20_movies(test_movie_reviews)
    # # call function to test positive word cloud
    # # call function to test negative word cloud
    # test_word_count_vs_review_score(test_movie_reviews)
    print('Testing accuracy:')
    print(test_fit_and_predict(test_movie_reviews))


if __name__ == "__main__":
    main()
