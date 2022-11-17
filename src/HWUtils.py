import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import json
import spacy
from spacy.training import Example
from spacy.scorer import Scorer
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import warnings
from nltk.translate.bleu_score import sentence_bleu
import boto3
warnings.filterwarnings('ignore') # Add a comment here about why you did this 
import os
from dotenv import load_dotenv
from google.cloud import translate_v2 as translate
# Nit: sort imports according to python style: https://peps.python.org/pep-0008/#imports
# pycharm will do this for you automatically if you just use the "optimize imports" function! 

# Overall, really clean code, great job!!! You should add tests for these classes in a tests file.
# In general, writing tests as you code will really help. every time I finish writing a class, I 
# immediately write some tests to make sure that it functions as expected and I often catch bugs that 
# way! 

# Also, were this my code, I would probably split into three files, one for each class. usually, the
# python standard is one class per file, and the file name imitates the class name but with snake case rather than camel case. 
NUM_LINES_TO_PREPROCESS=100
class Sentiment:
    """
    A class used to give solution to the first task of the homework.
    
    ...
    
    Attributes
    ----------
    rawdata : str
        string that contains the name of the file where the data is.
    
    Methods
    -------
    read_sentiment_data():
        Reads the data from the file.
    
    predict_sentiment():
        Executes the model to predict the sentiment with the previosuly processed
        data.
    
    check():
        Returns a string containing the class name.
    
    """
    
    def __init__(self, rawdata=None):
        
        """
        Constructs all the necessary attributes for the sentiment object.

        Parameters
        ----------
        rawdata : str
            string that contains the name of the file where the data is.
        """
        
        self.rawdata = rawdata
        self.data = None
        
    def read_sentiment_data(self):
        
        """
        Reads the data that contains the file which path is contained in rawdata.
        
        Returns
        -------
        None
        """
        
        with open(self.rawdata, 'r') as f:
            self.data = f.readlines()
        
    def predict_sentiment(self):
        
        """
        Executes the selected model to predict the sentiment on the text data.
        
        Returns
        -------
        None
        """
        
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

        for d in self.data:
            inputs = tokenizer(d, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                logits = model(**inputs).logits

            predicted_class_id = logits.argmax().item()
            if predicted_class_id == 1:
                print("POSITIVE")
            else:
                print("NEGATIVE")
    
    def check(self): # I don't think this method is necessary! but if you need to get the name of the class, you should use  __class__.__name__ rather than hardcoding it. 
        
        """
        Checks the name of the class.
        
        Returns
        -------
        A string that contains the name of the class.
        """
        
        return "Sentiment"

class NER:
    
    """
    A class used to give solution to the second task of the homework.
    
    ...
    
    Attributes
    ----------
    rawdata : str
        string that contains the name of the file where the data is.
    
    Methods
    -------
    read_ner_data():
        Reads the data from the file.
    
    preprocess_data():
        Puts the data in the Spacy training dataset way, which is:
        "{content: str, annotations: {(start, end, tag)}}"
    
    model_trainer():
        Trains the model using a blank spacy model and the training dataset,
        plots the training loss and saves the model to the same path.
        
    model_evaluation():
        Evaluates the model using the F1-Score, Recall and Precision for the
        entities (how well did the model tagged the content).
        
    check():
        Returns a string containing the class name.
    
    """
    
    def __init__(self, rawdata=None):
        
        """
        Constructs all the necessary attributes for the ner object.

        Parameters
        ----------
        rawdata : str
            string that contains the name of the file where the data is.
        """
        
        self.rawdata = rawdata
        self.data = None
        self.train = None
        self.test = None
        
    def read_ner_data(self):
        
        """
        Reads the data that contains the file which path is contained in rawdata.
        
        Returns
        -------
        None
        """
        
        with open(self.rawdata, 'r' ) as f:
            self.data = json.load(f)['examples']
    
    def preprocess_data(self):
        
        """
        Puts the data in the Spacy training dataset way, which is:
        "{content: str, annotations: {(start, end, tag)}}"
        
        Returns
        -------
        None
        """
        
        full_data = []

        for d in self.data: # in general, one-letter var names are discouraged! for datapoint in self.data or something 
            ent = [] # same comment, make this a more descriptive/readable variable name 
            for a in d['annotations']:
                if len(a['value']) == len(a['value'].strip()):
                    if len(a['human_annotations']) == 0:
                        continue
                    ent.append((a['start'], a['end'], a['tag_name']))
            
            if len(ent) > 0:
                full_data.append((d['content'], {'entities': ent}))
        
        self.train = full_data[:20]
        self.test = full_data[20:30]
        
    def model_trainer(self):
        
        """
        Trains the model using a blank spacy model and the training dataset,
        plots the training loss and saves the model to the same path.
        
        Returns
        -------
        None
        """
        
        iter = 30
        train_loss= []
        nlp = spacy.blank('en')

        if 'ner' not in nlp.pipe_names: # shouldnt need this conditional, can just always add the pipe
            ner = nlp.add_pipe('ner', last=True)

        for _, annotations in self.train:
            for ent in annotations.get('entities'): # for entity in 
                ner.add_label(ent[2])
                
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes):
            optimizer = nlp.begin_training()
            print('Training the model')
            for it in tqdm(range(iter), desc='Training'):
                random.shuffle(self.train)
                losses = {}
                for text, annotations in self.train:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    nlp.update([example], drop= 0.2, sgd=optimizer, losses=losses)
                train_loss.append(losses['ner'])
                
        fig, ax = plt.subplots()
        ax.plot(train_loss, color='r')
        ax.set_title('Training loss')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Steps (epochs)')
        plt.show()

        nlp.to_disk('./') # make this a named constant at the top. see https://peps.python.org/pep-0008/
        
    def model_evaluation(self):
        
        """
        Evaluates the model using the F1-Score, Recall and Precision for the
        entities (how well did the model tagged the content).
        
        Returns
        -------
        None
        """
        
        model = spacy.load('./')
        scorer = Scorer()

        examples = []

        for input, annotations in self.test:
            doc_gold_text = model.make_doc(input)
            example = Example.from_dict(doc_gold_text, annotations)
            example.predicted = model(str(example.predicted))
            examples.append(example)

        scores = scorer.score(examples)

        print('\nEvaluation Metrics\n')
        print('Entity precision Score: {}%\nEntity recall Score: {}\nEntity F1-Score: {}'.format(round(scores['ents_p']*100, 2),
                                                                                                 round(scores['ents_r'], 3),
                                                                                                 round(scores['ents_f'], 3)
                                                                                                 ))
        
    def check(self):
        
        """
        Returns a string containing the class name.
        
        Returns
        -------
        A string containing the class name.
        """
        
        return "NER"

class Translate:
    
    """
    A class used to give solution to the third task of the homework.
    
    ...
    
    Attributes
    ----------
    lang1_data : str
        string that contains the name of the file where the data from the first
        language is.
        
    lang2_data : str
        string that contains the name of the file where the data from the second
        language is.
    
    Methods
    -------
    env_variables(env_file, env_json):
        Reads the data from the files and sets up the env variables.
    
    preprocess_data():
        Reads and preprocesss the data contained in the two files and assigns the 
        first 100 lines of the file to a list.
    
    translate_bleu():
        Translate the texts (from english to spanish) using AWS Translate API 
        and Cloud Translation API and calculates the bleu scores of the translation 
        and the original text (spanish).
        
    check():
        Returns a string containing the class name.
    
    """
    
    def __init__(self, lang1_data=None, lang2_data=None):
        
        """
        Constructs all the necessary attributes for the translate object.

        Parameters
        ----------
        lang1_data : str
            string that contains the name of the file where the data from the first
            language is.
        
        lang2_data : str
            string that contains the name of the file where the data from the first
            language is.
        """
        
        self.lang1_data = lang1_data
        self.lang2_data = lang2_data
        self.__aws_auth = {}
        self.lang1 = None
        self.lang2 = None
        
    def env_variables(self, env_file, env_json):
        
        """
        Reads the data from the files and sets up the env variables.
        
        Parameters
        ----------
        env_file : str
            string that contains the file where the AWS API keys are.
            
        env_json : str
            string that contains the file where the GCP API keys are.
            
        Returns
        -------
        None
        """
        
        load_dotenv(env_file)

        self.__aws_auth['access_key'] = os.getenv("AWS_ACCESS_KEY_ID")
        self.__aws_auth['secret_access_key'] = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.__aws_auth['session_token'] = os.getenv("AWS_SESSION_TOKEN")
        self.__aws_auth['region'] = os.getenv("REGION_NAME")

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = env_json
        
    def preprocess_data(self):
        
        """
        Reads and preprocesses the data contained in the two files and assigns the 
        first NUM_LINES_TO_PREPROCESS lines of the file to a list.
        
        Returns
        -------
        None
        """
        
        spanish_texts = []
        english_texts = []

        with open(self.lang1_data, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[:NUM_LINES_TO_PREPROCESS]: 
                spanish_texts.append(line)
            
        with open(self.lang2_data, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[:NUM_LINES_TO_PREPROCESS]:
                english_texts.append(line)

        spanish_texts = [x.replace("\n", "") for x in spanish_texts]
        english_texts = [x.replace("\n", "") for x in english_texts]
        
        self.lang1 = spanish_texts
        self.lang2 = english_texts
        
    def translate_bleu(self):
        
        """
        Translate the texts (from english to spanish) using AWS Translate API 
        and Cloud Translation API and calculates the bleu scores of the translation 
        and the original text (spanish).
        
        Returns
        -------
        None
        """
        
        aws_translate = boto3.client('translate',
                                aws_access_key_id= self.__aws_auth['access_key'],
                                aws_secret_access_key= self.__aws_auth['secret_access_key'],
                                aws_session_token= self.__aws_auth['session_token'],
                                region_name=self.__aws_auth['region'])

        google_translate = translate.Client()

        aws_bleu = []
        google_bleu = []

        for i, input_to_translate in enumerate(self.lang2)

            aws_result = aws_translate.translate_text(Text=input_to_translate, 
                                            SourceLanguageCode='en', 
                                            TargetLanguageCode='es')


            google_result = google_translate.translate(sinput_to_translate, 'es')


            Ableu = sentence_bleu(input_to_translate.split(), aws_result['TranslatedText'].split())
            Gbleu = sentence_bleu(sinput_to_translate.split(), google_result['translatedText'].split())
            
            aws_bleu.append(Ableu)
            google_bleu.append(Gbleu)
            
        print("AWS Score: %s" % (sum(aws_bleu)/100))
        print("Google Score: %s" % (sum(google_bleu)/100))
        
    def check(self):
        
        """
        Returns a string containing the class name.
        
        Returns
        -------
        A string containing the class name.
        """
        
        return "Translate"
