import os
import sys


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(os.path.join(ROOT, 'src')) #Need to run this code first in order to have the test and src file on the same path.

from Sentiment import Sentiment
from NER import NER
from Translate import Translate

def test_sentiment():
    
    sentiment = Sentiment('tiny_movie_reviews_dataset.txt')
    assert sentiment.read_sentiment_data() == 'Data collected'

def test_ner():

    ner = NER('Corona2.json')
    assert ner.read_ner_data() == 'Data collected'
    
def test_trans():
    
    translate = Translate('europarl-v7.es-en.es', 'europarl-v7.es-en.en')
    assert translate.preprocess_data() == 'Data collected'
    
test_sentiment()
test_ner()
test_trans()
    