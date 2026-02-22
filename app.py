import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('punkt_tab')

ps = PorterStemmer()

vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

def preProcessing(text):
    text=text.lower()
    text=nltk.word_tokenize(text)

    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text=y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y); 

st.set_page_config(page_title="AI Spam Detector", page_icon="🤖", layout="centered")

st.markdown("""
<style>
/* Main App Background */
.stApp {
    background-color: #0f172a;
    color: white;
}

/* Neon Gradient Header */
.futuristic-header {
    padding: 40px;
    text-align: center;
    border-radius: 15px;
    background: linear-gradient(90deg, #00f5ff, #8a2be2);
    color: black;
    font-size: 40px;
    font-weight: 800;
    letter-spacing: 2px;
    box-shadow: 0 0 20px #00f5ff, 0 0 40px #8a2be2;
    margin-bottom: 10px;
}

/* Subtitle */
.futuristic-sub {
    text-align: center;
    color: #94a3b8;
    font-size: 18px;
    margin-bottom: 30px;
}

/* Glowing Button */
div.stButton > button {
    background: linear-gradient(90deg, #00f5ff, #8a2be2);
    color: white;
    font-weight: bold;
    border-radius: 10px;
    border: none;
    box-shadow: 0 0 10px #00f5ff;
    transition: 0.3s;
}

div.stButton > button:hover {
    box-shadow: 0 0 20px #8a2be2;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="futuristic-header">🤖 SPAMIFY</div>', unsafe_allow_html=True)
st.markdown('<div class="futuristic-sub">Powered by Machine Learning & Natural Language Processing</div>', unsafe_allow_html=True)

with st.container():
    input_message = st.text_area("✍️ Enter your message here:", height=150)

col1, col2 = st.columns([1,1])

with col1:
    predict_btn = st.button("🔍 Analyze Message", use_container_width=True)

with col2:
    st.empty()

if predict_btn:
    if input_message.strip() != "":
        # preprocessing
        transformed_message = preProcessing(input_message)

        # vectorization
        vector_input = vectorizer.transform([transformed_message])

        # prediction
        result = model.predict(vector_input)[0]
        prob = model.predict_proba(vector_input)[0]

        spam_prob = prob[1]
        st.subheader("🧠 Spam Confidence Score")
        st.progress(int(spam_prob * 100))
        st.write(f"Probability of Spam: {spam_prob*100:.2f}%")

        # display result
        if result == 1:
            st.markdown("""
                <div style='padding:20px; border-radius:10px; 
                background-color:#1e293b; 
                box-shadow:0 0 15px red; 
                text-align:center; font-size:24px;'>
                🚨 <span style='color:#ff4b4b;'>SPAM DETECTED</span>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='padding:20px; border-radius:10px; 
                background-color:#1e293b; 
                box-shadow:0 0 15px #00ffcc; 
                text-align:center; font-size:24px;'>
                ✅ <span style='color:#00ffcc;'>Wooho! Message is Not Spam</span>
                </div>
            """, unsafe_allow_html=True)

    else:
        st.warning("⚠️ Please enter a message first.")

with st.sidebar:
    st.title("📌 About")
    st.write("This app uses Machine Learning to classify SMS/Email messages.")
    st.write("Model: Multinomial Naive Bayes")
    st.write("Vectorizer: TF-IDF")
    st.write("Created by Krish Kumar 🚀")

