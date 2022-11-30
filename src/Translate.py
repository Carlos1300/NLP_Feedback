import boto3
import os

from nltk.translate.bleu_score import sentence_bleu
from dotenv import load_dotenv
from google.cloud import translate_v2 as translate


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
        Reads and preprocesses the data contained in the two files and assigns the 
        first NUM_LINES_TO_PROCESS of the file to a list.
    
    translate_bleu():
        Translate the texts (from english to spanish) using AWS Translate API 
        and Cloud Translation API and calculates the bleu scores of the translation 
        and the original text (spanish).
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
        first NUM_LINES_TO_PROCESS of the file to a list.
        
        Returns
        -------
        A message indicating the data has been collected
        """
        
        NUM_LINES_TO_PROCESS = 100
        
        spanish_texts = []
        english_texts = []

        with open(self.lang1_data, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[:NUM_LINES_TO_PROCESS]:
                spanish_texts.append(line)
            
        with open(self.lang2_data, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[:NUM_LINES_TO_PROCESS]:
                english_texts.append(line)

        spanish_texts = [x.replace("\n", "") for x in spanish_texts]
        english_texts = [x.replace("\n", "") for x in english_texts]
        
        self.lang1 = spanish_texts
        self.lang2 = english_texts
        
        return 'Data collected'
        
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

        for i, input_to_translate in enumerate(self.lang2):

            aws_result = aws_translate.translate_text(Text= input_to_translate, 
                                            SourceLanguageCode='en', 
                                            TargetLanguageCode='es')


            google_result = google_translate.translate(input_to_translate, 'es')


            Ableu = sentence_bleu(self.lang1[i].split(), aws_result['TranslatedText'].split()) # Here I'm referencing the other language data, so I have to use the index to extract the texts.
            Gbleu = sentence_bleu(self.lang1[i].split(), google_result['translatedText'].split())
            
            aws_bleu.append(Ableu)
            google_bleu.append(Gbleu)
            
        print("AWS Score: %s" % (sum(aws_bleu)/100))
        print("Google Score: %s" % (sum(google_bleu)/100))