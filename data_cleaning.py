"""
Megan Chiang, Michelle Kim, Jasmine Wong
CSE 163 AD/Group 055
Implements the data cleaning for the original dataset
to make it usable for the scope of our project.
"""
import pandas as pd


def main():
    """
    Filters the original data, keeps top critis, drops
    non values, and keeps reviews from 2015-2020.
    """
    # reading in ORIGINAL review data for cleaning
    reviews = pd.read_csv('rotten_tomatoes_critic_reviews.csv')
    reviews = reviews[['rotten_tomatoes_link', 'critic_name', 'top_critic',
                       'review_score', 'review_date', 'review_content']]

    # filtering for top critic
    is_top_critic = reviews['top_critic'] == 'True'
    filtered_reviews = reviews[is_top_critic]

    # dropping reviews without review scores
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

    # create new csv of filtered reviews
    filtered_reviews.to_csv('filtered_reviews.csv')


if __name__ == '__main__':
    main()
