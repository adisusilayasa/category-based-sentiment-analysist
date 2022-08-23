from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
from nltk.tokenize import word_tokenize
import numpy as np
import pickle
import tensorflow as tf


# load tokenizer weight
with open('./static/model/tokenizer.pickle', 'rb') as handle:
    mamam = pickle.load(handle)


# making prediction of the department
def get_prediction_department(seq_text):

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(
        './static/model/klasifikasi_terbaik.tflite')
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    #input_shape = input_details[0]['shape']
    input_data = np.array(seq_text, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
    #print(output_data[0][0] + output_data[0][1] + output_data[0][2])

    department_code = np.argmax(output_data[0])

    if department_code == 0:
        output_department = 'Bottoms'
    elif department_code == 1:
        output_department = 'Dresses'
    else:
        output_department = 'Tops'

    print(output_department)

    return output_department, output_data


# get sentiment model by predicted categories
def get_model_sentiment(kategori, seq_text):

    if kategori == 'Bottoms':
        interpreter = tf.lite.Interpreter(
            './static/model/sentimen_bottom.tflite')

    elif kategori == 'Dresses':
        interpreter = tf.lite.Interpreter(
            './static/model/sentimen_dresses.tflite')

    else:
        interpreter = tf.lite.Interpreter(
            './static/model/sentimen_tops.tflite')

    # Load TFLite model and allocate tensors.
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    #input_shape = input_details[0]['shape']
    input_data = np.array(seq_text, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
    #print(output_data[0][0] + output_data[0][1] + output_data[0][2])

    sentiment_code = np.argmax(output_data[0])

    if sentiment_code == 0:
        output_sentiment = 'Negatif'
    else:
        output_sentiment = 'Positif'

    return output_sentiment, output_data

# making prediction of sentiment


def get_prediction_cat_sentiment(seq_text):

    # get review categories
    kategori, percentage_cat = get_prediction_department(seq_text)

    # get sentiment model by predicted categories
    sentiment, percentage_sent = get_model_sentiment(kategori, seq_text)

    final_output = [kategori, sentiment]

    return final_output, percentage_cat, percentage_sent


def identify_tokens(df):
    stopwords_list = requests.get(
        "https://raw.githubusercontent.com/stopwords-iso/stopwords-en/master/stopwords-en.txt").content
    stopwords = set(stopwords_list.decode().splitlines())
    review = df['review'].lower()
    tokens = word_tokenize(review)
    token_words = ""
    # taken only words (not punctuation) and remove stopwords
    token_words = [w for w in tokens if w.isalpha()]
    token_words = [w for w in token_words if not w in stopwords]

    # menggunakan perulangan dan metode isalpha() untuk mengembalikan hanya huruf a-z
    return token_words


# encode text
def get_encode(text):
    seq = mamam.texts_to_sequences(text)
    seq = pad_sequences(seq, maxlen=1000, padding="post")
    return seq
