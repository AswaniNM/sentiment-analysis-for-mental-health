import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#st.write('Hello')
model = joblib.load('assets/Models/best_model.joblib')
vectorizer = joblib.load('assets/Models/vectorizer.joblib')
#st.write(model)

from plot import Tweet_Rates_per_Sentiment_Type,pie_chart,distribution_of_statement_lengths,word_cloud_for_Anxiety,word_cloud_for_Depression,word_cloud_for_Suicidal,word_cloud_for_Normal

current_dir = os.path.dirname(__file__)
csv_path = os.path.join(current_dir, "assets\Datasets\sentimentalds.csv")
df = pd.read_csv(csv_path)


def app():
    st.markdown("""<div style="text-align: justify;">
    Tweet Sentiment Analysis of Mental Health is aims to extract and analyse the sentiments expressed in tweets about mental health. The  goal is to categorise tweets into suicidal,normal,anxiety and depression sentiments using machine learning techniques.</div>""", unsafe_allow_html=True)
    st.markdown("""<div style="text-align: justify;">
    It entails collecting tweets with mental health-related keywords or hashtags, preprocessing the text data, and categorising the sentiments using algorithms. This analysis can reveal people's attitudes towards mental health issues, the difficulties they face, and the help they seek. Furthermore, the findings can help mental health professionals and researchers better address the needs of their communities.</div>""", unsafe_allow_html=True)
    st.markdown("#### Objective")
    st.markdown("""<div style="text-align: justify;">
    The primary goal is to use machine learning, specifically Decision Tree algorithms, to analyse the sentiment in mental health-related content. This can aid in identifying trends, interpreting public sentiment, and detecting early signs of mental health issues in online communities.</div>""", unsafe_allow_html=True)
def input_tab():
    
    # Get user input
    #statement = st.text_area("Enter the Statement", placeholder="Type your statement here...")
    statement = st.text_area("Enter the Statement", value=statement if 'statement' in locals() else "", placeholder="Type your statement here...")

    # Function to preprocess input and make predictions
    def make_prediction(input_text):
        input_tfidf = vectorizer.transform([input_text])
        prediction = model.predict(input_tfidf)
        return prediction
    
    # When the user clicks the predict button
    if st.button('Check Mental Health'):
        if statement:
            try:
                result = make_prediction(statement)
                st.write(f"Predicted Mental Health Status: {result[0]}")
            except Exception as e:
                st.error(f"Error making prediction: {e}")
        else:
            st.error("Please enter a statement.")
    


def info_graphics():
    st.write("#### About Dataset")
    st.markdown("""<div style="text-align: justify;">
    The dataset consists of textual statements tagged with one of four mental health statuses. 
    This dataset aims to facilitate the development and evaluation of machine learning models 
    for detecting and categorizing mental health conditions based on textual input.
    </div>""", unsafe_allow_html=True)
    
    st.write("#### Attributes")
    st.markdown("###### Statements")
    st.write("The primary feature of the dataset is a collection of textual statements.")
    st.markdown("###### Mental Health Status")
    st.write("Each statement is tagged with one of the following four mental health statuses:")  
    st.markdown("""
    - Normal: Indicates that the statement does not exhibit signs of mental health issues.
    - Depression: Statements that reflect symptoms of depression, such as feelings of sadness, hopelessness, and lack of interest in daily activities.
    - Suicidal: Statements that express suicidal thoughts or intentions.
    - Anxiety: Statements that show signs of anxiety, such as excessive worry, restlessness, and fear.
    """)

    Tweet_Rates_per_Sentiment_Type(df)
    pie_chart(df)
    distribution_of_statement_lengths(df)
    word_cloud_for_Anxiety(df)
    word_cloud_for_Depression(df)
    word_cloud_for_Suicidal(df)
    word_cloud_for_Normal(df)
    

      
def main():
    st.markdown("## Tweet Sentiment Analysis of Mental Health")
    tabs = ["App Overview","Info Graphics", "Prediction"]
    selected_tab = st.sidebar.radio("Choose a tab", tabs)
    if selected_tab == "Prediction":
        input_tab()
    elif selected_tab == "App Overview":
        app()
    elif selected_tab == "Info Graphics":
        info_graphics()

if __name__ == "__main__":
    main()