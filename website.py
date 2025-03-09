import streamlit as st
from nltk.stem.porter import PorterStemmer
import pickle
import re

# Load trained models
load_vector = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))

# Initialize Porter Stemmer
port_stem = PorterStemmer()

# Define stopwords
custom_stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
                    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", 
                    "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", 
                    "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", 
                    "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
                    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", 
                    "for", "with", "about", "against", "between", "into", "through", "during", "before", 
                    "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", 
                    "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", 
                    "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", 
                    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", 
                    "will", "just", "don", "should", "now"]

# Text preprocessing function
def stemming(content):
    new_content = re.sub('[^a-zA-Z]', ' ', content)
    new_content = new_content.lower()
    new_content = new_content.split()
    new_content = [port_stem.stem(word) for word in new_content if word not in custom_stopwords]
    return ' '.join(new_content)

# Prediction function
def detector(news):
    news = stemming(news)
    input_text = [news]
    new_load_vector = load_vector.transform(input_text)
    result = load_model.predict(new_load_vector)
    return result

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: red;'>📰 Fake News Detector! 📰</h1>", unsafe_allow_html=True)
st.subheader("Enter content below and wait for a result")
sentence = st.text_area("📝 **Enter your news content here:**", "", height=200)
predict_btt = st.button("🔍 Detect")

# Output results
if predict_btt:
    prediction_class = detector(sentence)
    
    if prediction_class == [0]:
        st.success("✅ **This information is likely based on FACTS**")
    elif prediction_class == [1]:
        st.error("❌ **This information is likely based on FALSE information**")