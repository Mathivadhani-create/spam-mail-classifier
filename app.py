import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📧 Spam Mail Classifier")
st.write("Enter a message below and I will predict whether it's spam or not.")

message = st.text_area("Message")

if st.button("Predict"):
    message_vec = vectorizer.transform([message])
    result = model.predict(message_vec)[0]
    if result == "spam":
        st.error("🚨 This looks like SPAM!")
    else:
        st.success("✔️ This message seems legitimate.")
