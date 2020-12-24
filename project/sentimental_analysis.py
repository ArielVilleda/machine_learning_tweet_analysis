# coding=utf-8

import sys
import numpy as np
import pandas as pd  # procesado y filtrado del csv
import nltk
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
# para la calificación de nuestros resultados (medida f1 y recall)
from sklearn import metrics
# para separar los datos aleatoriamente en test y train
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

__author__ = 'Ariel Villeda'

# importing english stopwords
STOPWORD_LIST = nltk.corpus.stopwords.words('english')
DEBUG = 0
MAX_ITERATIONS = 128


def debug_print(data):
    global DEBUG
    if DEBUG:
        str_data = str(data)
        print(str_data)
        print('------------------------------------------------\n\n')
        return True
    return False


def tokenize(text):
    # convertimos a minúsculas (preserve_case=False)
    # a excepción de los emoticones
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True,
                           preserve_case=False)
    return tknzr.tokenize(text)


def remove_stopwords(text):
    tokens = tokenize(text)
    # removemos las stop words del arreglo de tokens
    filtered_tokens = [token for token in tokens
                       if token not in STOPWORD_LIST]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def normalize_corpus(corpus):
    normalized_corpus = []
    for text in corpus:
        text = text.lower()
        text = remove_stopwords(text)
        normalized_corpus.append(text)
    return normalized_corpus


def main(argv=None):
    try:
        global DEBUG
        path_csv = argv[1]
        DEBUG = 1 if (len(argv) > 2) else 0
    except IndexError:
        err = 'Uso: {} <archivo_tweets.csv> (--debug)'.format(argv[0])
        raise SystemExit(err)

    # leyendo datos del csv
    data = pd.read_csv(path_csv)

    # imprimos los primeros datos leídos
    debug_print(data.loc[range(15), 'text'])

    # filter data based on training sentiment confidence
    data_clean = data.copy()
    # para entrenar seleccionamos solo los datos
    # con una confianza de sentimiento alta
    data_clean = data_clean[
        data_clean['airline_sentiment_confidence'] > 0.65
    ]

    # separamos nuestros datos aleatoriamente
    train, test = train_test_split(data_clean, test_size=0.2,
                                   random_state=1)
    # escojemos solamente la características del texto del tweet
    # y como se clasifica en el ámbito de sentimiento
    train_tweets = train['text'].values
    test_tweets = test['text'].values
    train_sentiments = train['airline_sentiment']
    test_sentiments = test['airline_sentiment']

    # normalización de tweets para el training
    norm_train = normalize_corpus(train_tweets)
    debug_print(norm_train)  # tokenizacion y n-gramas

    # preparamos el objeto de "matriz de tokenizacion"
    # en rango de 1 a 2 n-gramas, sobrecargamos el método
    # de tokenizacion con nuestro método creado
    vectorizer = CountVectorizer(ngram_range=(1, 2),
                                 tokenizer=tokenize)

    # creando la matriz término-documento
    train_features = vectorizer.fit_transform(norm_train).astype(float)
    debug_print(train_features)

    # construyendo el modelo de support vector machines
    print('Construyendo Modelo (TRAINING)...')
    svm = SGDClassifier(max_iter=MAX_ITERATIONS)
    svm.fit(train_features, train_sentiments)

    # normalizando los tweets de Testing
    norm_test = normalize_corpus(test_tweets)
    # matriz término-documento
    test_features = vectorizer.transform(norm_test)
    # obteniendo la media precisión en la predicción
    # svm.score(test_features, test_sentiments)

    # prediciendo sentimiento
    print('Probando Modelo (TESTING)...')
    predicted_sentiments = svm.predict(test_features)

    # imprimiento reporte de inducadores (medidas de precisión del modelo)
    report = metrics.classification_report(
        y_true=test_sentiments,
        y_pred=predicted_sentiments,
        labels=['positive', 'neutral', 'negative']
    )
    print(report)

    print('Ejecución TERMINADA')
    return 1


if __name__ == "__main__":
    main(sys.argv)
