import streamlit as st
import pickle
import sklearn


tfidf= pickle.load(open('vectorizer_pkl','rb'))
text_transform= pickle.load(open('model_pkl','rb'))

st.title('SMS/EMAIL SPAM DETECTOR')

st.write('This app will help us detect spam messages')
st.write('Built with streamlit and Python ')
input_sms=st.text_input('Enter your text here')
if st.button("predict"):
    st.text("predict  {}\n".format(input_sms))
    cv_text= tfidf.transform([input_sms]).toarray()
    prediction= text_transform.predict(cv_text)
    if prediction== 0:
        st.write("Not spam")
    else:
        st.write("spam")