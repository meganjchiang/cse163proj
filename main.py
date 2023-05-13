import pandas as pd
# from datetime import datetime
# import numpy as np
# import nltk


def main():
    # movies = pd.read_csv('rotten_tomatoes_movies.csv')
    # movies = movies[['rotten_tomatoes_link', 'movie_title', 'genres']]

    reviews = pd.read_csv('rotten_tomatoes_critic_reviews.csv')
    reviews = reviews[['rotten_tomatoes_link', 'critic_name', 'top_critic',
                       'review_score', 'review_date', 'review_content']]

    # movie_reviews = movies.merge(reviews, on='rotten_tomatoes_link')
    # print(movie_reviews.head())

    # filter review dataset (< 100MB) and push to github (needs
    # to be in same folder)
    # dropping missing values in 'review_score'
    # reviews.dropna()
    # filtered_reviews = reviews[reviews['review_score'].notnull()]
    # reviews['review_score'].notnull()
    is_top_critic = (reviews['top_critic'] == True)

    filtered_reviews = reviews[is_top_critic]
    filtered_reviews = filtered_reviews.dropna()

    # filter for 2015-2020
    # convert dates (str) to dates
    filtered_reviews['review_date'] = pd.to_datetime(filtered_reviews[
                                            'review_date'], format='%Y-%m-%d')
    # filter for between 2015-01-01 to 2020-12-31
    # filtered_reviews = filtered_reviews[filtered_reviews['review_date'] >
    filtered_reviews = filtered_reviews[(filtered_reviews["review_date"] >=
                                        "2015-01-01") &
                                        (filtered_reviews["review_date"]
                                         <= "2020-12-31")]

    # print(filtered_reviews.head())
    filtered_reviews.to_csv('filtered_reviews.csv')

    # remove rows with null review scores
    # need to filter for top critics and/or certain years
    # convert scores to fractions/floats
    # normalize text


if __name__ == '__main__':
    main()
