from src.HWUtils import Sentiment, NER, Translate

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