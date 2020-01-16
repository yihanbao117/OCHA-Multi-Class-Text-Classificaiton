# OCHA-Multi-Class-Text-Classificaiton

## Description
This reporsitory contains the Flask Api to make multi-class text classification prediction for OCHA (United Nations Office for the Coordination of Humanitarian Affairs). This project attempts to improve the efficiency of classifying job documents on RelifWeb website for OCHA. This project's main objective is to build a Web Prevention Tool for OCHA to improve the efficiency for editors.

Currently the main focus is to allow:

* The endpoint to uploading binay joson file for career category label;
* the endpoint to get the predicted labels and probabilities for career category label.

The following are the main features of the prediction endpoint:

* Loading data, and models;
* Cleaning and transforming text data into numerical data;
* Geting the prediction output and probabilities for each job document.

## Dependencies

There are a number of project dependencies required to develop and operate the OCHA Web Prevention Tool.

The following list details project dependencies:

* (IDE) Visual Studio Code Version 1.31;
* (PACKAGE) pyenchant 2.0.0;
* (PACKAGE) nltk 3.4;
* (CORPUS) nltk stopwords corpus;
* (PACKAGE) pandas 0.24.1;
* (PACKAGE) numpy 1.16.1;
* (PACKAGE) scikit-learn 0.20.1;
* (PACKAGE) textblob 0.15.2;
* (CORPUS) textblob corpus;
* (PACKAGE) re build-in;
* (PACKAGE) pickle build-in;
* (PACKAGE) Flask 1.0.3;
* (PACKAGE) Flask-Cors 3.0.8;
* (PACKAGE) Flask-RESTful 0.3.7;
* (PACKAHE) pyxlsb==1.0.6

## Getting Started

* Clone the repository into your machine;
* Install all the above dependencies with correct Python version;
* In the terminal, change the path to api and run "export FLASK_APP=run.py" and then "flask run" to start the local server;
* Upload a binay json file with POST method for "http://127.0.0.1:5000/upload/category";
* Use GET method for "http://127.0.0.1:5000/predict/category" to get the predicted results.

## Author
* OICT/PSGD/ETT | yihan.bao@un.org