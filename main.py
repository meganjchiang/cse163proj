def main():
    reviews = pd.read_csv('filtered_reviews.csv')
    reviews = reviews[['rotten_tomatoes_link', 'critic_name', 'top_critic',
                       'review_score', 'review_date', 'review_content']]

    movie_reviews = movies.merge(reviews, on='rotten_tomatoes_link')

    # TODO
    # remove rows with null review scores
    # need to filter for top critics and/or certain years
    # convert scores to fractions/floats
    # normalize text



if __name__ == '__main__':
    main()