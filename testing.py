"""
Megan Chiang, Michelle Kim, Jasmine Wong
CSE 163 AD/Group 055
Tests the functions that visualize and predict
movie review ratings from Rotten Tomatoes.
"""

import main as m
import pandas as pd


def main():
    test_movie_reviews = pd.read_csv('40_movies.csv')

    m.plot_top_20_movies(test_movie_reviews,
                         'test_20_highest_rated_movies.png')
    m.plot_bottom_20_movies(test_movie_reviews,
                            'test_20_lowest_rated_movies.png')
    m.wordcloud_positive(test_movie_reviews, 'test_positive_wordcloud.png')
    m.wordcloud_negative(test_movie_reviews, 'test_negative_wordcloud.png')
    m.word_count_vs_review_score(test_movie_reviews,
                                 'test_ave_word_count_vs_score.png',
                                 'test_distributions_count_score.png')
    print('Testing accuracy:')
    print(m.fit_and_predict(test_movie_reviews, 'test_machine_learning.png'))


if __name__ == "__main__":
    main()
