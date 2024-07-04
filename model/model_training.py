
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense
import numpy as np
from sklearn.model_selection import train_test_split
from data import all_queries,labels
from tensorflow.keras.optimizers import Adam  # Example optimizer
from tensorflow.keras.losses import \
    CategoricalCrossentropy  # Example loss function



# Split data for training and testing (implement logic here)
train_queries,test_queries,train_labels,test_labels = train_test_split(
    all_queries,labels,test_size=0.2
    )

# Tokenize text
tokenizer = Tokenizer(num_words=5000)  # Adjust vocabulary size
tokenizer.fit_on_texts(train_queries)
train_sequences = pad_sequences(
    tokenizer.texts_to_sequences(train_queries),maxlen=20
    )  # Adjust max sequence length
test_sequences = pad_sequences(
    tokenizer.texts_to_sequences(test_queries),maxlen=20
    )

# One-hot encode labels
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)
train_labels_encoded = encoder.fit_transform(
    np.array(train_labels).reshape(-1,1)
    )
test_labels_encoded = encoder.transform(
    np.array(test_labels).reshape(-1,1)
    )

# Define model
model = Sequential()
model.add(Embedding(5000,150,input_length=20))  # Adjust embedding size
model.add(LSTM(25))  # Adjust number of LSTM units
model.add(
    Dense(6,activation="softmax")
    )  # 3 for general, exceptions, refund

# Compile model

model.compile(
    optimizer=Adam(learning_rate=0.001),  # Adjust learning rate as needed
    loss=CategoricalCrossentropy(from_logits=False),
    # Adjust loss function if needed
    metrics=['accuracy']
    )  # Add metrics to track (optional)

# Train model
model.fit(
    train_sequences,train_labels_encoded,epochs=15,
    validation_data=(test_sequences,test_labels_encoded)
    )

# Save model (optional)
model.save("model/model.h5")
import pickle

with open('model/tokenizer.pickle','wb') as handle:
    pickle.dump(tokenizer,handle,protocol=pickle.HIGHEST_PROTOCOL)






