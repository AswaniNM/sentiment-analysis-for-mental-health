import streamlit as st
import joblib
#st.write('Hello')
model = joblib.load('assets/Models/best_model.joblib')
vectorizer = joblib.load('assets/Models/vectorizer.joblib')
#st.write(model)


def app():
    st.write("Sentiment analysis, also known as opinion mining, involves the use of computational techniques to identify and extract subjective information from text. It aims to determine the emotional tone behind words, thereby categorizing sentiments as Normal, Depression,Suicidal or Anxiety. ")
    st.write("The dataset consists of textual statements tagged with one of four mental health status. This dataset aims to facilitate the development and evaluation of machine learning models for detecting and categorizing mental health conditions based on textual input.They are:")
    st.markdown("""
    - Statements
    - Mental Health Status
    """)
    st.markdown("###### Statements")
    st.write("The primary feature of the dataset is a collection of textual statements.")
    st.markdown("###### Mental Health Status")
    st.write("Each statement is tagged with one of the following seven mental health statuses:")  
    st.markdown("""
    - Normal: Indicates that the statement does not exhibit signs of mental health issues.
    - Depression: Statements that reflect symptoms of depression, such as feelings of sadness, hopelessness, and lack of interest in daily activities.
    - Suicidal: Statements that express suicidal thoughts or intentions.
    - Anxiety: Statements that show signs of anxiety, such as excessive worry, restlessness, and fear.
    """)
    
    st.markdown("#### Objective")
    st.write("The objective of this Mental Health Prediction App is to leverage advanced natural language processing (NLP) techniques and machine learning models to analyze textual statements and predict the associated mental health status.")

def input_tab():
    
    # Get user input
    statement = st.text_input("Enter the Statement", placeholder="Type your statement here...")

    # Function to preprocess input and make predictions
    def make_prediction(input_text):
        input_tfidf = vectorizer.transform([input_text])
        prediction = model.predict(input_tfidf)
        return prediction
    
    # When the user clicks the predict button
    if st.button('Predict Mental Health'):
        if statement:
            try:
                result = make_prediction(statement)
                st.write(f"Predicted Mental Health Status: {result[0]}")
            except Exception as e:
                st.error(f"Error making prediction: {e}")
        else:
            st.error("Please enter a statement.")
    


def info_graphics():
    pass

def main():
    st.markdown("# Mental Health Prediction App")
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