import tensorflow as ts 
from keras import layers, models, preprocessing
import tensorflow_datasets as tfds

# Loading IMDB dataset

(train_data, test_data), info = tfds.load('imdb_reviews', split=('train', 'test'), with_info=True, as_supervised=True)

# Preprocessing data

vocab_size = 10000
max_len = 100
embedding_dim = 16

tokenizer = preprocessing.text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_data)
train_seq = tokenizer.texts_to_sequences(train_data)
test_seq = tokenizer.texts_to_sequences(test_data)

train = preprocessing.sequence.pad_sequences(train_seq, maxlen=max_len)
test = preprocessing.sequence.pad_sequences(test_seq, maxlen=max_len)

# Defining model

model = models.Sequential([
    layers.Embedding(vocab_size, embedding_dim, input_len = max_len),
    layers.GlobalAveragePolling1D(),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compiling model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training model

history = model.fit(train, epochs=10, validation_data=test)

test_loss, test_acc = model.evaluate(test, test_data)

print(f"Model accuracy: {test_acc}")