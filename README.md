# datamining-project

Comparing Critic and User Movie Reviews
Course

CSCE 474 – Introduction to Data Mining

University of Nebraska Lincoln

# Team Members
Brady Lauritsen
Sagun Karki
Harrison Johs
Max Buettenback

Project Overview

This project analyzes movie reviews from Rotten Tomatoes (critics) and Letterboxd (users) to compare how professional critics and general audiences evaluate films.

We aim to identify patterns in rating differences, perform sentiment analysis on review text, and develop predictive models to determine whether critic ratings can predict user ratings.

- Objectives:
1. Data Preprocessing: Integration of Rotten Tomatoes and Letterboxd datasets, including rating normalization and text tokenization.

2. Correlation Analysis: Measuring statistical relationships using Pearson and Spearman coefficients.

3. Sentiment Analysis: Analyzing the emotional tone of reviews using NLP techniques.

4. Predictive Modeling: Developing Linear and Random Forest Regression models to predict user ratings.

# Datasets

- Rotten Tomatoes Movies Dataset

- Rotten Tomatoes Critic Reviews Dataset

- Letterboxd Movie Dataset

- Letterboxd User Reviews Dataset

# Getting started

Since the datasets exceed GitHub's file limit the datasets are tracked in .gitignore and must be downloaded manually.
1. Rotten Tomatoes Datasett: Download from https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset
    - Place rotten_tomatoes_movies.csv and rotten_tomatoes_critic_reviews.csv into data/raw/
2. Letterboxd Dataset: Download from https://www.kaggle.com/datasets/samlearner/letterboxd-movie-ratings-data
    - Place movie_data.csv and ratings_export.csv into data/raw/

- Make sure to run pip install -r requirements.txt
so that you have all of the libraries installed


