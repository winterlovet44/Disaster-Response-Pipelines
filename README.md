# Disaster Response Pipeline Project


### Table of Contents

1. [Dependencies](#dependencies)
2. [Installation](#installation)
3. [Project Description](#motivation)
4. [File Descriptions](#files)
5. [Instructions](#results)


## Dependencies <a name="dependencies"></a>
To run code in this project, your enviroments need:
1. [Pandas](https://pandas.pydata.org/)
2. [Numpy](https://numpy.org/)
3. [Scikit-learn](https://scikit-learn.org/stable/)
4. [NLTK](https://www.nltk.org/)
5. [SQLAlchemy](https://sqlalchemy.org/)
6. [Plotly](https://plotly.com/)
7. [Flask](https://flask.palletsprojects.com/)

## Installation <a name="installation"></a>

The code was implemented in Python 3.9. All necessary package was contained in `requirements.txt` file.

For quick installation:
```sh
pip install -r requirements.txt
```


## Project Description<a name="motivation"></a>

In this project, I will perform my data engineering skill with data from [Appen](https://www.figure-eight.com/).

This project contains three steps:

1. ETL: data will be cleaned and stored in SQLite database
2. Model pipeline: Build a Machine learning pipeline to feature engineering and train ML model.
3. Deployment: Build Flask web app to visualize data and predict user's input query


## File Descriptions <a name="files"></a>

```bash
├── README.md
├── app
│   ├── run.py # Flask app
│   └── templates # Folder contains html file to render web app
│       ├── go.html
│       └── master.html
├── data
│   ├── DisasterResponse.db # SQLite database file
│   ├── YourDatabaseName.db
│   ├── disaster_categories.csv # Raw data
│   ├── disaster_messages.csv # Raw data
│   └── process_data.py # ETL pipeline
├── models
│   ├── classifier.pkl # Saved model
│   └── train_classifier.py # Model pipeline
└── requirements.txt # Package requirement
```

## Instructions<a name="results"></a>

1. To run ETL pipeline

```bash
cd data
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
```
2. To build and train model

```bash
cd models
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
```

3. Run web app
Run the code below to start the web app at localhost
```bash
cd app
python run.py
```
And go to [http://localhost:3000](http://localhost:3000) to see the web app



