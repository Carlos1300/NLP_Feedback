import json
import spacy
import random
import matplotlib.pyplot as plt
import warnings

from spacy.training import Example
from spacy.scorer import Scorer
from tqdm import tqdm

warnings.filterwarnings('ignore') #Used to avoid spacy's warnings that make the code difficult to read.

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
        A message indicating the data has been collected.
        """
        
        with open(self.rawdata, 'r' ) as f:
            self.data = json.load(f)['examples']
        
        return 'Data collected'
    
    def preprocess_data(self):
        
        """
        Puts the data in the Spacy training dataset way, which is:
        "{content: str, annotations: {(start, end, tag)}}"
        
        Returns
        -------
        None
        """
        
        full_data = []

        for datapoints in self.data:
            entities = []
            for annotations in datapoints['annotations']:
                if len(annotations['value']) == len(annotations['value'].strip()):
                    if len(annotations['human_annotations']) == 0:
                        continue
                    entities.append((annotations['start'], annotations['end'], annotations['tag_name']))
            
            if len(entities) > 0:
                full_data.append((datapoints['content'], {'entities': entities}))
        
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
        FINAL_MODEL = None
        iter = 30
        train_loss= []
        nlp = spacy.blank('en')

        ner = nlp.add_pipe('ner', last=True)

        for _, annotations in self.train:
            for entity in annotations.get('entities'):
                ner.add_label(entity[2])
                
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

        FINAL_MODEL = nlp.to_disk('./')
        
    def model_evaluation(self):
        
        """
        Evaluates the model using the F1-Score, Recall and Precision for the
        entities (how well did the model tagged the content).
        
        Returns
        -------
        None.
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