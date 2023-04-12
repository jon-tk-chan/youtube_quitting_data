import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

stopwords = set(STOPWORDS)
extra_stopwords = set(['job', 'reason', 'poster', 'pursue',
                       'list', 'wanted', 'leaving', 'previous'])
stopwords.update(extra_stopwords)

st.title('Why YouTubers Quit - Text Analysis')
st. markdown("""
This app examines transcripts from YouTube videos about quitting their jobs and performs NLP Tasks such as:

""")

st.set_option('deprecation.showPyplotGlobalUse', False)
st.sidebar.header("Select Words to Display")
words = st.sidebar.slider("No. of words",
                          min_value=10,
                          max_value=1000,
                             value=500)
prev_keywords = []

in_df = pd.read_csv("chatgpt_output.csv")


split_list = [full_string.split() for full_string in in_df['quitting_reasons']]
for sent_list in split_list:
    for word in sent_list:
        cleaned_word = re.sub(r'[^\w\s]', '', word)
        cleaned_word = lemmatizer.lemmatize(cleaned_word.lower())
        if cleaned_word not in stopwords:
            prev_keywords.append(cleaned_word.strip())
prev_keywords_str = " ".join(prev_keywords)

st.markdown("### Top keywords mentioned in previous job (QA BY CHATGPT)")
wordcloud = WordCloud(background_color = "white", max_words =
              words,stopwords = stopwords).generate(prev_keywords_str)
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis("off")
plt.show()
st.pyplot()
# st.sidebar.header("Select Link")
# links = ["https://seaportai.com/blog-predictive-maintenance/",
#          "https://seaportai.com/healthcare-analytics/",
#          "https://seaportai.com/blog-rpameetsai/",
#          "https://seaportai.com/covid-19/"]

# URL = st.sidebar.selectbox('Link', links)
# st.sidebar.header("Select No. of words you want to display")

# words = st.sidebar.selectbox("No. of words", range(10, 1000, 10))

# if URL is not None:
#     r = requests.get(URL)
#     #using the web scraping library that is Beautiful Soup
#     soup = BeautifulSoup(r.content, 'html.parser')
#     #extracting the data that is in 'div' content of HTML page
#     table = soup.find('div', attrs = {'id':'main-content'})
#     text = table.text
#     #cleaning the data with regular expression library
#     cleaned_text = re.sub('\t', "", text)
#     cleaned_texts = re.split('\n', cleaned_text)
#     cleaned_textss = "".join(cleaned_texts)
#     st.write("Word Cloud Plot")
#     #using stopwords to remove extra words
#     stopwords = set(STOPWORDS)
#     wordcloud = WordCloud(background_color = "white", max_words =
#               words,stopwords = stopwords).generate(cleaned_textss)
    
# plt.imshow(wordcloud, interpolation = 'bilinear')
# plt.axis("off")
# plt.show()
# st.pyplot()

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)