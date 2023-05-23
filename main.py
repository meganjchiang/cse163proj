import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import nltk

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
# Third data visualization


def main():
    movies = pd.read_csv('rotten_tomatoes_movies.csv')
    reviews = pd.read_csv('filtered_reviews.csv')

    # join 2 datasets and clean
    movie_reviews = merge_and_clean(movies, reviews)

    # first data visualization (top and bottom 20 movies)
    plot_top_20_movies(movie_reviews)
    plot_bottom_20_movies(movie_reviews)

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
