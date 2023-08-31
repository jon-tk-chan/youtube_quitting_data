import streamlit as st
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from nltk import word_tokenize
# from nltk.util import ngrams


import collections
import spacy
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import json
import plotly.express as px
import re

in_filepath = "data/03_finalStatsAndQA.csv"
full_df = pd.read_csv(in_filepath)
lemmatizer = WordNetLemmatizer()
extra_stopwords={"job", "occupation", "work", "career", "not", "specified", "N/A",
                "pursue", "wanted", "to", "as", "full time", "currently", "work", "career", "time", "working", "poster", "job", "wa", "something", "pursue",
                "video", "new", "full", "N", "feeling", "take", "formerly",
                "um", 'uh', 'u', 'oh', 'thing','people', 'thing'}
stopwords = set(STOPWORDS)
prefix1 = "The current job/occupation of the speaker is"
prefix2 = "The former job/occupation of the speaker was"
prefix3 = "The main reason that the speaker had for quitting their job/occupation was"
full_stopwords = set(prefix1.split() + prefix2.split() + prefix3.split()) | stopwords | extra_stopwords
full_stopwords = set([lemmatizer.lemmatize(word.lower()) for word in list(full_stopwords)])
def make_barchart(in_column, sep_char = ".",word_count=50, title='BARCHART_TITLE'):
    """Return wordcloud matplotlib figure using a list of df column be counted and made into a frequency dictionary.
    
    Assume that in_column is a pd Series of list of strings where units to be counted are separated by 
    sep_char (can count words or phrases)
    
    """
    joined_string = " ".join(in_column)
    string_list = joined_string.split(sep_char)
    counter = collections.Counter(string_list)
    counts_dict = {}
    for word, count in counter.items():
        counts_dict[word] = count
    
    df = pd.DataFrame({'words': counts_dict.keys(),
                      "counts": counts_dict.values()}).sort_values(by='counts',ascending=False).iloc[:word_count]
    
    fig = px.bar(df, y='words', x='counts',title=title)
    return fig
def make_wordcloud(in_column, sep_char = ".",word_count=50, title='WORDCLOUD TITLE'):
    """Return wordcloud matplotlib figure using a list of df column be counted and made into a frequency dictionary.
    
    Assume that in_column is a pd Series of list of strings where units to be counted are separated by 
    sep_char (can count words or phrases)
    
    """
    joined_string = " ".join(in_column)
    string_list = joined_string.split(sep_char)
    counter = collections.Counter(string_list)
    counts_dict = {}
    for word, count in counter.items():
        counts_dict[word] = count

    wordcloud = WordCloud(background_color='black',max_words=word_count,
                         stopwords=full_stopwords)
    wordcloud.generate_from_frequencies(frequencies=counts_dict)
    fig = plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout(pad=0)

    return fig

word_count = st.sidebar.slider(
    "Select number of words to display:",
    1,
    150,
    15
)
cat_options = st.sidebar.multiselect(
    'Select Video Categories: ',
    full_df['categoryName'].unique(),
    ['Education', 'People & Blogs','Entertainment'])


# st.write('You selected:', cat_options)


curr_df = full_df[full_df['categoryName'].isin(cat_options)]
st.sidebar.write(f"NUMBER OF VIDEOS INCLUDED: {curr_df.shape[0]}")
# st.write(curr_df.columns)
# curr_transcripts = " ".join(curr_df['transcript_strings'].tolist())
# curr_transcripts_nouns = " ".join([str(x) for x in curr_df['transcript_nouns'].dropna().tolist()])
# curr_transcripts_adj = " ".join([str(x) for x in curr_df['transcript_adj'].dropna().tolist()])
# st.table(curr_df.isna())
# st.write(curr_adjectives)

job_wc = make_wordcloud(curr_df['former_job_nc'].dropna(), word_count=word_count,
                        title="Wordcloud - Former Jobs from Selected YouTube Transcripts")
reason_wc = make_wordcloud(curr_df['main_reason_nc'].dropna(), word_count=word_count,
                        title="Wordcloud - Quitting Reasons from YouTube Transcripts")

st.title('Why YouTubers Quit - Text Analysis')
st. markdown("""
This app examines transcripts from YouTube videos about quitting their jobs and performs NLP Tasks such as:

""")
          
fig1 = px.scatter(curr_df, x="commentCount", y="viewCount", color="categoryName", 
        title="",
        hover_data=['title'], 
        log_x = True, log_y= True)
fig1.update_layout(legend=dict(
    yanchor="top",
    y=0.35,
    xanchor="left",
    x=0.45
))
fig2 = make_barchart(curr_df['transcript_adj'].dropna(),word_count=word_count, 
                     title="")
col1, col2 , col3= st.columns([0.33,0.33, 0.33])
col1.subheader("'Why I Quit My Job' YouTube Video views vs comments - by category")
col1.plotly_chart(fig1, use_container_width=True)
col2.subheader("What is the former jobs mentioned by Youtubers who quit?")
col2.pyplot(job_wc) 
col2.subheader("What was the main reason for quitting their job?")
col2.pyplot(reason_wc) 
col3.subheader(f"Count of top {word_count} most frequent adjectives in Selected Transcripts")
col3.plotly_chart(fig2,use_container_width=True) 
# st.plotly_chart(fig1)