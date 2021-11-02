# NLP with Disaster Tweets
A challenge to build a machine learning model that predicts which Tweets are about real disasters and which one’s aren’t.


### Model Architectures
#### - Bidirectional LSTM with 100D GloVe Embedding (Accuracy 82.3%)
```python
model = Sequential()
model.add(Embedding(
    input_dim = embedding_matrix.shape[0],
    output_dim = embedding_matrix.shape[1],
    weights = [embedding_matrix],
    input_length = input_length
))
model.add(Bidirectional(LSTM(units = 64, return_sequences = True, recurrent_dropout = 0.2)))
model.add(GlobalMaxPool1D())
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units = 128, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(units = 128, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(units = 1, activation = 'sigmoid'))
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
```
#### - Bert Model (Accuracy 83.8%)
```python
input_ids = Input(shape = (60,), dtype = 'int32')
attention_masks = Input(shape = (60,), dtype = 'int32')    

output = model_layer([input_ids, attention_masks])[1]
output = Dropout(0.2)(output)    
output = Dense(units = 1, activation = 'sigmoid')(output)

model = Model(inputs = [input_ids,attention_masks],outputs = output)
model.compile(Adam(lr = learning_rate), loss = 'binary_crossentropy', metrics = ['accuracy'])
```


### links
- [Kaggle page](https://www.kaggle.com/c/nlp-getting-started)
