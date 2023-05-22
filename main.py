import pandas as pd
# import nltk


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
        movie_reviews['review_score'].apply(convert_to_category)

    # return merged and cleaned dataset
    return movie_reviews


def convert_to_float(fraction: str) -> float:
    """
    Converts a given fraction (string) to a float.
    """
    numbers = fraction.split("/")

    return float(numbers[0]) / float(numbers[1])


def convert_to_category(score: float) -> str:
    if score <= 0.2:
        return "1"
    elif score <= 0.4:
        return "2"
    elif score <= 0.6:
        return "3"
    elif score <= 0.8:
        return "4"
    else:
        return "5"

# First data visualization
# Second data visualization
# Third data visualization


def main():
    movies = pd.read_csv('rotten_tomatoes_movies.csv')
    reviews = pd.read_csv('filtered_reviews.csv')

    # join 2 datasets and clean
    movie_reviews = merge_and_clean(movies, reviews)

    # to do:
    # remove rows with null review scores -> DONE
    # need to filter for top critics and/or certain years -> DONE
    # remove alphabetic scores -> DONE
    # convert scores to fractions/floats -> DONE
    # add new column of sentiment score (NLTK) to merged dataset
    # create visualizations
    # create ML model and calculate accuracy/error

    movie_reviews.to_csv('movie_reviews.csv')


if __name__ == '__main__':
    main()
