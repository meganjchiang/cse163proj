import pandas as pd
# import nltk


def main():
    movies = pd.read_csv('rotten_tomatoes_movies.csv')
    movies = movies[['rotten_tomatoes_link', 'movie_title', 'genres']]
    
    reviews = pd.read_csv('filtered_reviews.csv')
    reviews = reviews[['rotten_tomatoes_link', 'critic_name', 'review_score',
                       'review_date', 'review_content']]

    # join two datasets on movie link
    movie_reviews = movies.merge(reviews, on='rotten_tomatoes_link')
    movie_reviews = movie_reviews.reset_index(drop=True)

    # get only rows with numeric scores (no A+, B-, etc.)
    pattern = r'^[A-F][+-]?$'
    movie_reviews = movie_reviews[~movie_reviews['review_score'].str.match(pattern)]

    # remove rows with scores w/o denominator (no "/")
    movie_reviews['review_score'] 
    movie_reviews = movie_reviews[movie_reviews['review_score'].str.contains("/")]
    
    # convert scores to floats
    movie_reviews['review_score'] = movie_reviews['review_score'].apply(convert_to_number)


    # TODO
    # remove rows with null review scores -> DONE
    # need to filter for top critics and/or certain years -> DONE
    # remove alphabetic scores -> DONE
    # convert scores to fractions/floats -> DONE
    # add new column of sentiment score (NLTK) to merged dataset

def convert_to_number(fraction: str) -> float:
        numbers = fraction.split("/")
        return float(numbers[0]) / float(numbers[1])


if __name__ == '__main__':
    main()