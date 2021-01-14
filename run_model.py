from os import listdir
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from tqdm import tqdm
import numpy as np
import h5py as h5py
from compiler.classes.Compiler import *
from keras.models import Model,load_model

vocab_file="./resources/my_words.vocab"
test_dir_name = './resources/test_data/'
model_h5file="./resources/my_model.h5"
dsl_path = "compiler/assets/web-dsl-mapping.json"
output_html_path="index.html"
inp_shape=(512,7,7)
# Read a file and return a string
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def load_data(data_dir):
    text = []
    images = []
    # Load all the files and order them
    all_filenames = listdir(data_dir)
    all_filenames.sort()
    for filename in (all_filenames):
        if filename[-3:] == "npz":
            # Load the images already prepared in arrays
            image = np.load(data_dir+filename)
            images.append(image['features'].reshape(inp_shape))
        else:
            # Load the boostrap tokens and rap them in a start and end tag
            syntax = '<START> ' + load_doc(data_dir+filename) + ' <END>'
            # Seperate all the words with a single space
            syntax = ' '.join(syntax.split())
            # Add a space after each comma
            syntax = syntax.replace(',', ' ,')
            text.append(syntax)
    images = np.array(images, dtype=float)
    return images, text

train_features, texts = load_data(test_dir_name)

# Initialize the function to create the vocabulary 
tokenizer = Tokenizer(filters='', split=" ", lower=False)
# Create the vocabulary 
tokenizer.fit_on_texts([load_doc(vocab_file)])
# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None
print(word_for_id(17, tokenizer))


model=load_model(model_h5file)


# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    photo = np.array([photo])
    # seed the generation process
    in_text = '<START> '
    # iterate over the whole length of the sequence
    print('\nPrediction---->\n\n<START> ', end='')
    for i in range(150):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo, sequence], verbose=0)
        # convert probability to integer
        yhat = np.argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += word + ' '
        # stop if we predict the end of the sequence
        print(word + ' ', end='')
        if word == '<END>':
            break
    return in_text

max_length = 48+1 
# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    # step over the whole set
    for i in range(len(texts)):
        yhat = generate_desc(model, tokenizer, photos[i], max_length)
        # store actual and predicted
        print('\n\nReal---->\n\n' + texts[i])
        actual.append([texts[i].split()])
        predicted.append(yhat.split())
    # calculate BLEU score
    #bleu = corpus_bleu(actual, predicted)
    bleu=0.1
    return bleu, actual, predicted

bleu, actual, predicted = evaluate_model(model, texts, train_features, tokenizer, max_length)

#Compile the tokens into HTML and css

compiler = Compiler(dsl_path)
compiled_website = compiler.compile(predicted[0], output_html_path)

print(compiled_website )
print(bleu)