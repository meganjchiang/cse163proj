# Analysis and Prediction of Movie Review Ratings on Rotten Tomatoes

The project cannot be ran directly in the Ed Workspace. The project can be ran locally on an IDE of your choice in the CSE 163 Anaconda environment.

Download the required data sets (rotten_tomatoes_critic_reviews.csv and rotten_tomatoes_movies.csv) at https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset.

Note: These files are 243.17 MB total.

To install the required NLTK and word cloud libraries:
* Install NLTK by running `pip install - user - U nltk` in the IDE's terminal
* Install wordclound by running `pip install wordcloud` in the IDE's terminal

To run the project's Python code files:
* data_cleaning.py - reads in the "rotten_tomatoes_critic_reviews.csv" file, and outputs a new csv called "filtered_reviews.csv)
* main.py - reads in the "filtered_reviews.csv" and "rotten_tomatoes_movies.csv", then runs the main project code.
* testing.py - reads in a smaller subset of data "40_movies.csv" (found in the Ed Workspace) to run the tests.

Link to our project's GitHub Repository: https://github.com/meganjchiang/cse163proj
