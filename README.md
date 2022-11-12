# NLP Module Project

This project was carried out to solve three specific tasks related to NLP.

## Description

The project has three classes that are used throughout the program, each of them has the purpose of solving the following tasks:
* Sentiment Analysis for movie reviews:
    For this task I selected the "distilbert-base-uncased-finetuned-sst-2-english" for running a sentiment analysis on a movie reviews dataset, then the output printed     to the terminal.
* Training a NER Model for a medical purpose:
    For this task a model from scratch using Spacy and the Medical NER dataset (https://www.kaggle.com/datasets/finalepoch/medical-ner), the training loss graph can be     viewed when running the program, also some metrics are used to evaluate the model.
    
    <p align="center"><img style="width: 500px; height: 400px;" src="loss_graph.jpg" alt="loss_graph"></p>
    
* Obtaining the Bleu Score of translations (english to spanish) made by AWS and GCP Translation APIs.
    For this task we will use the AWS and GCP Translation APIs, to set up the env file which contains the AWS API keys you will have to create a new "secrets.env" and     write the following variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN, REGION_NAME.
    To make this process much easier there is a function on the Translate class which takes like paramas the secrets.env and the json file from GCP.
    
### Installing
Installation of the project can be done by following these steps:
* Make a new virtual environment using Anaconda or venv.
* Activate your virtual environment.
* Execute the following command in the terminal: pip install -r requirements.txt (you can use the requirments.txt file from this repository).


### Executing program

* Activate a virtual environment or create a new one.
* Move to the directory where you cloned this repo.
* Execute the following command in the terminal: py run.py

## Authors

Carlos de Jesús Ávila González
A01750220@itesm.mx
