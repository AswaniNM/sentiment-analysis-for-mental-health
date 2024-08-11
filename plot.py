import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st 
import plotly.express as px
from wordcloud import WordCloud

def Tweet_Rates_per_Sentiment_Type(df):
    st.markdown("###### 1. Number of tweets per sentiment type:")
    # Creating the countplot
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='status', hue='status')
    plt.title('Distribution of Mental Health Status')
    plt.xlabel('Mental Health Status')
    plt.ylabel('Count')

    # Display the plot in Streamlit
    st.pyplot(plt)

def pie_chart(df):
    # Creating the pie chart
    fig = px.pie(df, names='status', title='Proportion of Each Status Category')

    # Display the pie chart in Streamlit
    st.plotly_chart(fig)

def distribution_of_statement_lengths(df):
    # Calculate the length of each statement
    df['statement'] = df['statement'].fillna('')
    df['statement_length'] = df['statement'].apply(len)
    st.markdown("###### Basic Statistics of Statement Lengths")
    
    # Plot the distribution of statement lengths
    plt.figure(figsize=(10, 6))
    df['statement_length'].hist(bins=100)
    plt.title('Distribution of Statement Lengths')
    plt.xlabel('Length of Statements')
    plt.ylabel('Frequency')

    # Display the plot in Streamlit
    st.pyplot(plt)

def word_cloud_for_Anxiety(df):
    with st.spinner("Generating Word Cloud for Anxiety..."):
        st.markdown("###### Word Cloud for Anxiety")
        #st.subheader("Word Cloud for Anxiety")
        status_text = ' '.join(df[df['status'] == 'Anxiety']['statement'])
        plt.figure(figsize=(15, 10))
        wordcloud_anxiety = WordCloud(max_words=500, height=800, width=1500, background_color="black", colormap='viridis').generate(status_text)
        plt.imshow(wordcloud_anxiety, interpolation="bilinear")
        plt.axis('off')
        st.pyplot(plt)

def word_cloud_for_Depression(df):
    with st.spinner("Generating Word Cloud for Depression..."):
        st.markdown("###### Word Cloud for Depression")
        status_text_depression = ' '.join(df[df['status'] == 'Depression']['statement'])
        plt.figure(figsize=(15, 10))
        wordcloud_depression = WordCloud(max_words=500, height=800, width=1500, background_color="black", colormap='viridis').generate(status_text_depression)
        plt.imshow(wordcloud_depression, interpolation="bilinear")
        plt.axis('off')
        st.pyplot(plt)

def word_cloud_for_Suicidal(df):
    with st.spinner("Generating Word Cloud for Suicidal..."):
        st.markdown("###### Word Cloud for Suicidal")
        status_text_Suicidal = ' '.join(df[df['status'] == 'Suicidal']['statement'])
        plt.figure(figsize=(15, 10))
        wordcloud_Suicidal = WordCloud(max_words=500, height=800, width=1500, background_color="black", colormap='viridis').generate(status_text_Suicidal)
        plt.imshow(wordcloud_Suicidal, interpolation="bilinear")
        plt.axis('off')
        st.pyplot(plt)

def word_cloud_for_Normal(df):
    with st.spinner("Generating Word Cloud for Normal..."):
        st.markdown("###### Word Cloud for Normal")
        status_text_Normal = ' '.join(df[df['status'] == 'Normal']['statement'])
        plt.figure(figsize=(15, 10))
        wordcloud_Normal = WordCloud(max_words=500, height=800, width=1500, background_color="black", colormap='viridis').generate(status_text_Normal)
        plt.imshow(wordcloud_Normal, interpolation="bilinear")
        plt.axis('off')
        st.pyplot(plt)



# # Generate word clouds for each status using a loop
# for status in statuses:
#     generate_word_cloud(df, status)
# def word_cloud(df):
#     def generate_word_cloud(text, title):
#         wordcloud = WordCloud(width=800, height=400).generate(text)
#         plt.figure(figsize=(10, 5))
#         plt.imshow(wordcloud, interpolation='bilinear')
#         plt.title(title)
#         plt.axis('off')
#         st.pyplot(plt)  # Use st.pyplot to display the plot in Streamlit
#         plt.clf()  # Clear the plot to avoid overlap with next plots

#     # Generate word clouds for each status
#     statuses = df['status'].unique()

#     for status in statuses:
#         with st.spinner(f"Generating Word Cloud for {status}..."):
#             status_text = ' '.join(df[df['status'] == status]['statement'])
#             generate_word_cloud(status_text, title=f'Word Cloud for {status}')
