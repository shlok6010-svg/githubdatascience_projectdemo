import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences 
import pickle
import numpy as np 

#load the saved model 
model = tf.keras.models.load_model('sentiment_analysis_model.h5')
with open('tokenizer.pickle','rb') as f:
    tokenizer = pickle.load(f)


    def predict_sentiment(tweet):
        #Preprocess the input tweet
        sequences = tokenizer.texts_to_sequences([tweet])
        padded = pad_sequences(sequences,maxlen=100, padding='post')

        prediction = model.predict(padded)
        predicted_class = np.argmax(prediction, axis=1)[0]
        return predicted_class



    st.title("Twitter Sentiment Analysis")
    st.write("Analyze the sentiment of tweets using a pre-trained LSTM model.")

    user_input = st.text_area("Enter a tweet :", placeholder= "Type your tweet here...")

    if st.button("Predict Sentiment"):
        if user_input:

            predict_class = predict_sentiment(user_input)

            sentiment_labels = {0: 'Negative', 1:'Neutral', 2:'Posoitive'}
            sentiment = sentiment_labels[predict_class]


            if sentiment == 'Positive' :
                st.success(f"The sentiment of the tweet is :{sentiment}")
            elif sentiment == 'Negative':
                st.error(f"The sentiment of the tweet is:{sentiment}")
            else:
                st.info(f"The sentiment of the tweet is:{sentiment}")

        else:
            st.warning("Please Enter a Tweet to Analyze.")