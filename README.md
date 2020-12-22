# Disaster Response Pipeline Project

### Introduction:
This proyect analyzes disaster data from [Figure Eight] (https://appen.com/) in order to classify them in different categories, that are already stated in a file.

The proyect uses  
1. **SQLite** database for it implementation
2. pickle to store the trained model
3. a web application to retrieve data and show classification results

The implementation does not take into account model calibration or finding hiperparameters

### Model:
The model steps are the following
1. applies text transformation to the messages, to tokenize and find tfidf
    1.1. Applies a Count of the terms in the message by CountVectorizer
    1.2. Transform a count matrix to a tf-idf representation 
2. use a Random Forest and a MultiOutputClassifier to train model

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Project File structure:
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
    * train_calssifier.py: script to create, train tje model and save as pickle file
    *
