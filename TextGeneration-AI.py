#!/usr/bin/env python
# coding: utf-8

# In[13]:


import tkinter as tk
from tkinter import ttk, scrolledtext

import numpy as np
import pickle
import random

from nltk.tokenize import RegexpTokenizer
from tensorflow.keras.models import load_model

# Load tokenizer and model artifacts
tokenizer = RegexpTokenizer(r"\w+")

# Load the model and history
model = load_model("text_gen_model2.keras")
history = pickle.load(open("history2.p", "rb"))

# Sample partial text for recreating tokenizer (needed for index mapping)
import pandas as pd
text_df = pd.read_csv(r"C:\Users\ASUS\Downloads\fake_or_real_news.csv")
joined_text = " ".join(list(text_df.text.values))
partial_text = joined_text[:10000]
tokens = tokenizer.tokenize(partial_text.lower())
unique_tokens = np.unique(tokens)
unique_token_index = {token: index for index, token in enumerate(unique_tokens)}

n_words = 10

def predict_next_word(input_text, n_best):
    input_text = input_text.lower()
    X = np.zeros((1, n_words, len(unique_tokens)))
    split_input = input_text.split()
    
    if len(split_input) < n_words:
        return [random.randint(0, len(unique_tokens)-1)]
    
    for i, word in enumerate(split_input[-n_words:]):
        if word in unique_token_index:
            X[0, i, unique_token_index[word]] = 1
    predictions = model.predict(X, verbose=0)[0]
    return np.argpartition(predictions, -n_best)[-n_best:]

def generate_text(input_text, num_words, creativity=3):
    word_sequence = input_text.split()
    current = 0
    for _ in range(num_words):
        sub_sequence = " ".join(tokenizer.tokenize(" ".join(word_sequence).lower())[current:current+n_words])
        try:
            choice = unique_tokens[random.choice(predict_next_word(sub_sequence, creativity))]
        except:
            choice = random.choice(unique_tokens)
        word_sequence.append(choice)
        current += 1
    return " ".join(word_sequence)


def on_generate():
    prompt = prompt_entry.get()
    try:
        num_words = int(num_words_entry.get())
        creativity = int(creativity_entry.get())
    except ValueError:
        result_text.delete('1.0', tk.END)
        result_text.insert(tk.END, "Please enter valid integers for number of words and creativity.")
        return

    result = generate_text(prompt, num_words, creativity)
    result_text.delete('1.0', tk.END)
    result_text.insert(tk.END, result)

# Set up the window
window = tk.Tk()
window.title("AI Text Generator")
window.geometry("700x500")
window.resizable(False, False)

# Input frame
input_frame = ttk.Frame(window, padding=10)
input_frame.pack(fill='x')

ttk.Label(input_frame, text="Enter Prompt:").grid(row=0, column=0, sticky='w')
prompt_entry = ttk.Entry(input_frame, width=80)
prompt_entry.grid(row=0, column=1, padx=5, pady=5)

ttk.Label(input_frame, text="Words to Generate:").grid(row=1, column=0, sticky='w')
num_words_entry = ttk.Entry(input_frame, width=10)
num_words_entry.insert(0, "50")
num_words_entry.grid(row=1, column=1, sticky='w')

ttk.Label(input_frame, text="Creativity Level:").grid(row=2, column=0, sticky='w')
creativity_entry = ttk.Entry(input_frame, width=10)
creativity_entry.insert(0, "5")
creativity_entry.grid(row=2, column=1, sticky='w')

generate_button = ttk.Button(input_frame, text="Generate Text", command=on_generate)
generate_button.grid(row=3, column=1, sticky='w', pady=10)

# Output frame
result_frame = ttk.Frame(window, padding=10)
result_frame.pack(fill='both', expand=True)

result_text = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, width=80, height=20)
result_text.pack(fill='both', expand=True)

window.mainloop()


# In[ ]:




