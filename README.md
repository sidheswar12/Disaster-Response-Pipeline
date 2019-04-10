# Disaster-Response-Pipeline

## Project Overview

Main purpose of this project is to analyze disaster data set from Figure Eight to build a model for an API that classifies disaster messages. It also includes web app where an emergency worker can input new message and get classification results in several categories.
## Project Components

Distaster Response project contains three parts: 
1. ETL Pipeline
2. Machine Learning Pipeline
3. Flask App

    ### 1. ETL Pipeline
        ETL Pipeline Preparation.ipynb jupyter notebook shows the code and development of ETL pipeline.
        process_data.py Python script loads the messages & categories datasets, and merges the clean data then store the data into a SQLite database.
    ### 2. Machine Line Pipeline
        ML Pipeline Preparation.ipynb Jupyther notebook shows the code and develoment of Machine Learning Pipeline.
        train_classifier.py Python script loads the data from a SQLite database. And it uses the data to train and tune a Machine Learning model using GridSearchCV. Finally the model will output as a .pkl file.
    ### 3. Flask App
        The web app can receive an input of new message and returns classification results in several categories.

## Instructions

  1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

  2. Run the following command in the app's directory to run your web app.
    `python run.py`

  3. Go to http://0.0.0.0:3001/
