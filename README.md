# Disaster Response Pipeline Project

### Summary:
The following proyect contains the steps to train a model that classifies disaster messages data from [Figure Eight] (https://appen.com/) into 36 different categories. The proyect is divided into 3 main steps. 

1. data cleaning and preparation
2. training and tunning a model
3. depployment of a webpage to classify new entries

#### Technologies
The proyect uses  
1. **SQLite** database 
2. pickle to store the trained model
3. a web application to retrieve data and show classification results

#### Model Training
The model steps are the following
1. applies text transformation to the messages, to tokenize and find tfidf
    1.1. Applies a Count of the terms in the message by CountVectorizer
    1.2. Transform a count matrix to a tf-idf representation 
2. perform a grid search on Random Forest and a MultiOutputClassifier to train model

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Files:
Three folders 
* **app**
    * **templates**
        * go.html
        * master.html
    * run.py:  script to run the web app
* **data**
    * disaster_messages.csv: File with all the messages to train the model
    * disaster_categories.csv: Fiñe with the categories of the messages
    * process_data.py: script to read the csv files pre process, join them and save the resulting info in a Sqlite database
* **model**
    * train_calssifier.py: script to create, train the model and save as pickle file
    *
