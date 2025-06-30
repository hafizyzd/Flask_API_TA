import re
import numpy as np
import pickle 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def preprocess_text(text, tokenizer, max_len=100):

    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s\(\)\*\=\-\/\+\%\>\<\!\;\,\.\']", '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([=><\+\-\*/\%\(\),;])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    sequence = tokenizer.texts_to_sequences([text])
    
    padded = pad_sequences(sequence, maxlen=max_len, padding='post')
    
    return padded