import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
        A message indicating the data has been collected.
        """
        
        with open(self.rawdata, 'r') as f:
            self.data = f.readlines()
            
        return 'Data collected'
        
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