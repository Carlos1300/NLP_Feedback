import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, pipeline

import json
import spacy
from spacy import displacy
from spacy.training import Example
from spacy.util import filter_spans
from spacy.tokens import Doc
from spacy.scorer import Scorer
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import warnings
import requests
from nltk.translate.bleu_score import sentence_bleu
import boto3
warnings.filterwarnings('ignore')
import os
from dotenv import load_dotenv
from google.cloud import translate_v2 as translate


class Sentiment:
    
    def __init__(self, rawdata):
        self.rawdata = rawdata
        self.data = None
        
    def read_sentiment_data(self):
        with open(self.rawdata, 'r') as f:
            self.data = f.readlines()
        
    def predict_sentiment(self):
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
        


        

# print('-' *25 + 'EXERCISE I' + '-'*25)

# def read_tiny_data():

#     with open('tiny_movie_reviews_dataset.txt', 'r') as f:
#         review = f.readlines()
        
#     return review

# def predict_sentiment(data):
#     tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

#     model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

#     for d in data:
#         inputs = tokenizer(d, return_tensors="pt", max_length=512, truncation=True)
#         with torch.no_grad():
#             logits = model(**inputs).logits

#         predicted_class_id = logits.argmax().item()
#         if predicted_class_id == 1:
#             print("POSITIVE")
#         else:
#             print("NEGATIVE")
        

# reviews = read_tiny_data()
# predict_sentiment(reviews)


class NER:
    
    def __init__(self, rawdata):
        self.rawdata = rawdata
        self.data = None
        self.train = None
        self.test = None
        
    def read_ner_data(self):
        with open(self.rawdata, 'r' ) as f:
            self.data = json.load(f)['examples']
    
    def preprocess_data(self):
        full_data = []

        for d in self.data:
            ent = []
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
        iter = 30
        train_loss= []
        nlp = spacy.blank('en')

        if 'ner' not in nlp.pipe_names:
            ner = nlp.add_pipe('ner', last=True)

        for _, annotations in self.train:
            for ent in annotations.get('entities'):
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

        nlp.to_disk('./')
        
    def model_evaluation(self):
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


# print('-' *25 + 'EXERCISE II' + '-'*25)

# #### READING DATA ####

# with open('Corona2.json', 'r' ) as f:
#     data = json.load(f)['examples']

# #### MAKING THE TRAIN SET ####

# full_data = []

# for d in data:
#     ent = []
#     for a in d['annotations']:
#         if len(a['value']) == len(a['value'].strip()):
#             if len(a['human_annotations']) == 0:
#                 continue
#             ent.append((a['start'], a['end'], a['tag_name']))
    
#     if len(ent) > 0:
#         full_data.append((d['content'], {'entities': ent}))
                
# #### TRAINING FUNCTION ####

# train = full_data[:20]
# test = full_data[20:30]

# iter = 30
# train_loss= []
# nlp = spacy.blank('en')

# if 'ner' not in nlp.pipe_names:
#     ner = nlp.add_pipe('ner', last=True)

# for _, annotations in train:
#     for ent in annotations.get('entities'):
#         ner.add_label(ent[2])
        
# other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
# with nlp.disable_pipes(*other_pipes):
#     optimizer = nlp.begin_training()
#     print('Training the model')
#     for it in tqdm(range(iter), desc='Training'):
#         random.shuffle(train)
#         losses = {}
#         for text, annotations in train:
#             doc = nlp.make_doc(text)
#             example = Example.from_dict(doc, annotations)
#             nlp.update([example], drop= 0.2, sgd=optimizer, losses=losses)
#         train_loss.append(losses['ner'])
        
# fig, ax = plt.subplots()
# ax.plot(train_loss, color='r')
# ax.set_title('Training loss')
# ax.set_ylabel('Loss')
# ax.set_xlabel('Steps (epochs)')
# plt.show()

# nlp.to_disk('./')

# ##### EVALUATE THE MODEL #####
# model = spacy.load('./')
# scorer = Scorer()

# examples = []

# for input, annotations in test:
#     doc_gold_text = model.make_doc(input)
#     example = Example.from_dict(doc_gold_text, annotations)
#     example.predicted = model(str(example.predicted))
#     examples.append(example)

# scores = scorer.score(examples)

# print('Evaluation Metrics')
# print('Entity precision Score: {}%\nEntity recall Score: {}\nEntity F1-Score: {}'.format(scores['ents_p']*100,
#                                                                                          scores['ents_r'],
#                                                                                          scores['ents_f']
#                                                                                          ))

class Translate:
    
    def __init__(self, lang1_data, lang2_data):
        self.lang1_data = lang1_data
        self.lang2_data = lang2_data
        self.__aws_auth = {}
        self.lang1 = None
        self.lang2 = None
        
    def env_variables(self, env_file, env_json):
        load_dotenv(env_file)

        self.__aws_auth['access_key'] = os.getenv("AWS_ACCESS_KEY_ID")
        self.__aws_auth['secret_access_key'] = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.__aws_auth['session_token'] = os.getenv("AWS_SESSION_TOKEN")
        self.__aws_auth['region'] = os.getenv("REGION_NAME")

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = env_json
        
    def preprocess_data(self):
        spanish_texts = []
        english_texts = []

        with open(self.lang1_data, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[:100]:
                spanish_texts.append(line)
            
        with open(self.lang2_data, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[:100]:
                english_texts.append(line)

        spanish_texts = [x.replace("\n", "") for x in spanish_texts]
        english_texts = [x.replace("\n", "") for x in english_texts]
        
        self.lang1 = spanish_texts
        self.lang2 = english_texts
        
    def translate_bleu(self):
        aws_translate = boto3.client('translate',
                                aws_access_key_id= self.__aws_auth['access_key'],
                                aws_secret_access_key= self.__aws_auth['secret_access_key'],
                                aws_session_token= self.__aws_auth['session_token'],
                                region_name=self.__aws_auth['region'])

        google_translate = translate.Client()

        aws_bleu = []
        google_bleu = []

        for i in range(len(self.lang2)):

            aws_result = aws_translate.translate_text(Text= self.lang2[i], 
                                            SourceLanguageCode='en', 
                                            TargetLanguageCode='es')


            google_result = google_translate.translate(self.lang2[i], 'es')


            Ableu = sentence_bleu(self.lang1[i].split(), aws_result['TranslatedText'].split())
            Gbleu = sentence_bleu(self.lang1[i].split(), google_result['translatedText'].split())
            
            aws_bleu.append(Ableu)
            google_bleu.append(Gbleu)
            
        print("AWS Score: %s" % (sum(aws_bleu)/100))
        print("Google Score: %s" % (sum(google_bleu)/100))
        

# print('-' *25 + 'EXERCISE III' + '-'*25)


# ### ENV VARIABLES ###

# load_dotenv('secrets.env')

# acces_key = os.getenv("AWS_ACCESS_KEY_ID")
# secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
# session_token = os.getenv("AWS_SESSION_TOKEN")
# region = os.getenv("REGION_NAME")

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'nlpretro-06b1ef850554.json'


# spanish_texts = []
# english_texts = []

# with open('europarl-v7.es-en.es', 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#     for line in lines[:100]:
#         spanish_texts.append(line)
    
# with open('europarl-v7.es-en.en', 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#     for line in lines[:100]:
#         english_texts.append(line)

# spanish_texts = [x.replace("\n", "") for x in spanish_texts]
# english_texts = [x.replace("\n", "") for x in english_texts]


# aws_translate = boto3.client('translate',
#                         aws_access_key_id=acces_key,
#                         aws_secret_access_key=secret_access_key,
#                         aws_session_token=session_token,
#                         region_name=region)

# google_translate = translate.Client()

# aws_bleu = []
# google_bleu = []

# for i in range(len(english_texts)):

#     aws_result = aws_translate.translate_text(Text= english_texts[i], 
#                                     SourceLanguageCode='en', 
#                                     TargetLanguageCode='es')


#     google_result = google_translate.translate(english_texts[i], 'es')


#     Ableu = sentence_bleu(spanish_texts[i].split(), aws_result['TranslatedText'].split())
#     Gbleu = sentence_bleu(spanish_texts[i].split(), google_result['translatedText'].split())
    
#     aws_bleu.append(Ableu)
#     google_bleu.append(Gbleu)
    
# print("AWS Score: %s" % (sum(aws_bleu)/100))
# print("Google Score: %s" % (sum(google_bleu)/100))

if __name__ == '__main__':
    print('-' *25 + 'EXERCISE I' + '-'*25 + '\n')
    task1 = Sentiment('tiny_movie_reviews_dataset.txt')
    task1.read_sentiment_data()
    task1.predict_sentiment()
    print('\n' + '-' *25 + 'EXERCISE II' + '-'*25 + '\n')
    task2 = NER('Corona2.json')
    task2.read_ner_data()
    task2.preprocess_data()
    task2.model_trainer()
    task2.model_evaluation()
    print('\n' + '-' *25 + 'EXERCISE III' + '-'*25 + '\n')
    task3 = Translate('europarl-v7.es-en.es', 'europarl-v7.es-en.en')
    task3.preprocess_data()
    task3.env_variables('secrets.env', 'nlpretro-06b1ef850554.json')
    task3.translate_bleu()