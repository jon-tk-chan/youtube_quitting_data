import streamlit as st
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

extra_stopwords={"work", "career", "time", "working", "poster", "job", "wa", "something", "pursue",
                "video", "new", "full"}
stopwords = set(STOPWORDS)

stopwords.update(extra_stopwords)
lemmatizer = WordNetLemmatizer()


st.set_option('deprecation.showPyplotGlobalUse', False)

df = pd.read_csv("chatgpt_output.csv")
df= df.fillna("NA")


def clean_reasons(in_list):
    """"""
    cleaned_reasons = []

    for response in in_list:
        reasons = response.split("\n")[-3:]
        reasons = [x[3:] for x in reasons] #remove the number from numbered list
        for rsn in reasons:
            #split into tokens
            tokens = rsn.split()
            lowered = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    #         lowered = []
    #         for word in tokens:
    #             lowered.append(lemmatizer.lemmatize(word.lower()))
            cleaned_rsn = " ".join(lowered).replace(".", "")
#             print(cleaned_rsn)
            cleaned_reasons.append(cleaned_rsn)
#         print("---")
    return cleaned_reasons
def create_ngram_df(sents_list, n_gram=3):
    """Returns a df frequency counts for each n_gram in the list of sentences in sent list"""
    n_gram_dict = { 
    }
    ngram_list = []
    for reason in sents_list:
        tokens = reason.split()
        end_i = len(tokens) - (n_gram - 1)
        start_i = 0
    #     print(tokens)
        for i in range(start_i, end_i):
            n_gram_end = i +(n_gram)
            n_gram_tuple = tuple(tokens[i:n_gram_end])
            ngram_list.append(n_gram_tuple)
#             if n_gram_tuple not in n_gram_dict.keys():
#                 n_gram_dict[n_gram_tuple] = 1
#             else:
#                 n_gram_dict[n_gram_tuple] += 1

#     to_df = {
#         "n_grams": [],
#         'count': []  
#     }
#     for key, count in n_gram_dict.items():
#         to_df['n_grams'].append(key)
#         to_df['count'].append(count)
    common_ngrams_df = pd.DataFrame(Counter(ngram_list).most_common(25))
    common_ngrams_df.columns = ('ngram', 'count')
    common_ngrams_df['ngram_str'] = common_ngrams_df['ngram'].apply(lambda x: "_".join(x))
    return common_ngrams_df


st.title('Why YouTubers Quit - Text Analysis')
st. markdown("""
This app examines transcripts from YouTube videos about quitting their jobs and performs NLP Tasks such as:

""")

st.sidebar.header("Select Words to Display")
words = st.sidebar.slider("No. of words",
                          min_value=10,
                          max_value=1000,
                             value=500)


cleaned_reasons = clean_reasons(df.quitting_reasons.tolist())
ngram_df = create_ngram_df(cleaned_reasons,4)


# text = "".join(df['quitting_reasons'].tolist())
text = "".join(cleaned_reasons)

wordcloud = WordCloud(background_color = "white", max_words =
              words,stopwords = stopwords).generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
st.pyplot()



col1, col2 = st.columns(2)
with col1:
    st.markdown("Most Common Ngrams (N=4)")
    sns.set(font_scale = 1.5)
    sns.barplot(x= ngram_df['count'], y=ngram_df['ngram_str'])
    plt.xticks(rotation='horizontal')
    plt.title("Key Word Count", fontsize = 15)
    plt.show()
    st.pyplot()
with col2:
    st.markdown("TF-IDF Terms from Quitting Reasons")
    # Instantiate
    vectorizer = TfidfVectorizer()
    # Fit the data
    tfidf = vectorizer.fit_transform(cleaned_reasons)
    # Create a dataframe of TFIDF
    tfidf_df = pd.DataFrame(tfidf[0].T.todense(), 
                        index=vectorizer.get_feature_names(), 
                        columns=["TF-IDF"])
    # Sort
    tfidf_df = tfidf_df.sort_values('TF-IDF', ascending=False).reset_index()
    # Bar Plot
    num_tf_terms = 15
    st.bar_chart(tfidf_df[:num_tf_terms])

# prev_keywords = []

# in_df = pd.read_csv("chatgpt_output.csv")


# split_list = [full_string.split() for full_string in in_df['quitting_reasons']]
# for sent_list in split_list:
#     for word in sent_list:
#         cleaned_word = re.sub(r'[^\w\s]', '', word)
#         cleaned_word = lemmatizer.lemmatize(cleaned_word.lower())
#         if cleaned_word not in stopwords:
#             prev_keywords.append(cleaned_word.strip())
# prev_keywords_str = " ".join(prev_keywords)

# st.markdown("### Top keywords mentioned in previous job (QA BY CHATGPT)")
# wordcloud = WordCloud(background_color = "white", max_words =
#               words,stopwords = stopwords).generate(prev_keywords_str)
# plt.imshow(wordcloud, interpolation = 'bilinear')
# plt.axis("off")
# plt.show()
# st.pyplot()
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