import tensorflow as tf
import numpy as np
import random
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ========== GPT-2 Model ==========

def generate_text_gpt2(prompt, max_length=100):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ========== LSTM-Based Model ==========

def train_lstm_model(text_data, max_words=5000, seq_length=20):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(text_data)
    total_words = len(tokenizer.word_index) + 1
    
    sequences = []
    for line in text_data:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            sequences.append(n_gram_sequence)
    
    sequences = pad_sequences(sequences, padding='pre')
    X, y = sequences[:, :-1], sequences[:, -1]
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)
    
    model = Sequential([
        Embedding(total_words, 50, input_length=seq_length-1),
        LSTM(100, return_sequences=True),
        LSTM(100),
        Dense(100, activation='relu'),
        Dropout(0.2),
        Dense(total_words, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=50, verbose=1)
    return model, tokenizer

# ========== Generating Text Using LSTM ==========

def generate_text_lstm(model, tokenizer, seed_text, next_words=50):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=model.input_shape[1], padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                seed_text += " " + word
                break
    return seed_text

# Example Usage
gpt2_output = generate_text_gpt2("Artificial Intelligence is")
print("GPT-2 Generated Text:", gpt2_output)

# Example LSTM Training Data
text_corpus = [
    "Artificial Intelligence is revolutionizing the world.",
    "Deep learning is a subset of machine learning.",
    "Natural language processing enables AI to understand text."
]

lstm_model, lstm_tokenizer = train_lstm_model(text_corpus)
lstm_output = generate_text_lstm(lstm_model, lstm_tokenizer, "Artificial Intelligence")
print("LSTM Generated Text:", lstm_output)

