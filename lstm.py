import pandas as pd
import numpy as np
import nltk
import gensim
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional

#DATA

df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

#DATA ANALYSIS

# Adding a column to indicate whether the news is true or fake
df_true['isfake'] = 0
df_fake['isfake'] = 1

# Concatenating True and Fake news
df = pd.concat([df_true, df_fake]).reset_index(drop=True)
df.drop(columns=['date'], inplace=True)

# Combining 'title' and 'text' colmuns together
df['original'] = df['title'] + ' ' + df['text']
#print(df['original'])


#DATA CLEANING

# Obtaining additional stopwords from nltk
stop_words = stopwords.words('english')

# Removing stopwords and words with 2 or less characters
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        # Taking words that don't belong to stopwords and have more than 2 characters
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
            result.append(token)

    return result

df['clean'] = df['original'].apply(preprocess)
#print(df)

# All words present in dataset
bag_of_words = []
for i in df.clean:
    for j in i:
        bag_of_words.append(j)

# All unique words present in dataset in one string
df['clean_joined'] = df['clean'].apply(lambda x: " ".join(x))
print(df['original'][0])
print(df['clean'][0])
print(df['clean_joined'][0])

#print(df['isfake'])

df_shuffled = df.sample(frac=1).reset_index(drop=True)
print(df_shuffled['isfake'])
'''
#VISUALIZING CLEANED UP DATA

# Number of samples in 'subject' column
plt.figure(figsize=(8, 8))
sns.countplot(y = "subject", data = df)
plt.show()

plt.figure(figsize = (8, 8))
sns.countplot(y = "isfake", data = df)
plt.show()

# The Word Cloud for real news
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = stop_words).generate(" ".join(df[df.isfake == 1].clean_joined))
plt.imshow(wc, interpolation = 'bilinear')

# The Word Cloud for fake news
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = stop_words).generate(" ".join(df[df.isfake == 0].clean_joined))
plt.imshow(wc, interpolation = 'bilinear')
'''
# Looking for sample with maximum length in a dataframe (needed to create word embeddings)
maxlen = -1
for doc in df_shuffled.clean_joined:
    tokens = nltk.word_tokenize(doc)
    if(maxlen<len(tokens)):
        maxlen = len(tokens)
print("The maximum number of words in any document is =", maxlen)

#TOKENIZATION AND PADDING

# Target class is 'isfake'
x_train, x_test, y_train, y_test = train_test_split(df_shuffled.clean_joined, df_shuffled.isfake, test_size=0.2)

# Number of unique words
total_words = len(list(set(bag_of_words)))

# Tokenizing words and creating sequneces of tokenized words
tokenizer = Tokenizer(num_words = total_words)
tokenizer.fit_on_texts(x_train)
train_sequences = tokenizer.texts_to_sequences(x_train)
test_sequences = tokenizer.texts_to_sequences(x_test)

print("The encoding for document\n", df_shuffled.clean_joined[0],"\n is : ",train_sequences[0])

# Adding padding
padding_train = pad_sequences(train_sequences, maxlen=40, padding='post', truncating='post')
padding_test = pad_sequences(test_sequences, maxlen=40, truncating='post')

#BUILDING AND TRAINING MODEL

model = Sequential()
model.add(Embedding(total_words, output_dim=128))
model.add(Bidirectional(LSTM(128)))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # Cause it's either 1 or 0
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.summary()

y_train = np.asarray(y_train)

#Training the model
model.fit(padding_train, y_train, batch_size=64, validation_split=0.1, epochs=2)
model.summary()

#ASSESING TRAINED MODEL PERFORMANCE
pred = model.predict(padding_test)

prediction = []
# If hte prediction is > 0.5 then the news is real otherwise it is fake
for i in range(len(pred)):
    if pred[i].item() > 0.5:
        prediction.append(1)
    else:
        prediction.append(0)

# Getting accuracy
accuracy = accuracy_score(list(y_test), prediction)
print("Model accuracy is : ", accuracy)

# Confusion matrix
c_matrix = confusion_matrix(list(y_test), prediction)
plt.figure(figsize=(25,25))
sns.heatmap(c_matrix, annot=True)
plt.show()

