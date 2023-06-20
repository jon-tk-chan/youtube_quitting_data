import streamlit as st
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import json
import plotly.express as px
import re

# extra_stopwords={"work", "career", "time", "working", "poster", "job", "wa", "something", "pursue",
#                 "video", "new", "full"}
# stopwords = set(STOPWORDS)
# stopwords.update(extra_stopwords)
lemmatizer = WordNetLemmatizer()

extra_stopwords={"job", "occupation", "work", "career", "not", "specified", "N/A",
                "pursue", "wanted", "to", "as", "full time", "currently", "work", "career", "time", "working", "poster", "job", "wa", "something", "pursue",
                "video", "new", "full", "N", "feeling", "take", "formerly"}
stopwords = set(STOPWORDS)
prefix1 = "The current job/occupation of the speaker is"
prefix2 = "The former job/occupation of the speaker was"
prefix3 = "The main reason that the speaker had for quitting their job/occupation was"
full_stopwords = set(prefix1.split() + prefix2.split() + prefix3.split()) | stopwords | extra_stopwords

category_dict = {
    1:"Film & Animation",
    2:"Autos & Vehicles",
    10:"Music",
    15:"Pets & Animals",
    17: "Sports",
    18:"Short Movies",
    19:"Travel & Events",
    20:"Gaming",
    21:"Videoblogging",
    22:"People & Blogs",
    23:"Comedy",
    24:"Entertainment",
    25:"News & Politics",
    26:"Howto & Style",
    27:"Education",
    28:"Science & Technology",
    29:"Nonprofits & Activism",
    30:"Movies",
    31:"Anime/Animation",
    32:"Action/Adventure",
    33:"Classics",
    34:"Comedy",
    35:"Documentary",
    36:"Drama",
    37:"Family",
    38:"Foreign",
    39:"Horror",
    40:"Sci-Fi/Fantasy",
    41:"Thriller",
    42:"Shorts",
    43:"Shows",
    44:"Trailers"
}

with open("QA_output/final_QA_259.json", 'r') as f:
    qa_json = json.load(f)

with open("youtube_stats_and_transcripts.json", 'r') as f:
    youtube_json = json.load(f)

#process data
youtube_df = pd.DataFrame(youtube_json)
youtube_df['viewCount'] = pd.to_numeric(youtube_df['viewCount'], errors='coerce')
youtube_df['likeCount'] = pd.to_numeric(youtube_df['likeCount'], errors='coerce')
youtube_df['commentCount'] = pd.to_numeric(youtube_df['commentCount'], errors='coerce')
youtube_df['categoryId'] = pd.to_numeric(youtube_df['categoryId'], errors='coerce')
# youtube_df['categoryId'] = youtube_df['categoryId'].astype(str)
youtube_df['categoryId'] = youtube_df['categoryId'].apply(lambda x: category_dict[x])
youtube_df = youtube_df.fillna(0.0)

def create_qa_df(in_json):
    """Return a dataframe with processed text for curr job/former job/quitting reason"""
    
    #process_qa
    video_ids = []
    curr_jobs = []
    former_jobs = []
    main_reasons = []
    for video_id, text in in_json.items():
        video_ids.append(video_id)
        curr_job = None
        former_job = None
        main_reason = None
        text = re.sub('\d\.', '', text)
        text = text.replace("\n", "")
        text_list = text.split(".")
        if '' in text_list:
            text_list.remove('')
        if len(text_list) == 3:
            curr_job = text_list[0]
            former_job = text_list[1]
            main_reason = text_list[2]
        else:
            for i, sent in enumerate(text_list):
                if 'current' in sent:
                    curr_job = sent
                if 'former' in sent:
                    former_job = sent
                if "main reason" in sent:
                    main_reason = " ".join(text_list[i:])
                else:
                    main_reason = text_list[-1]
        curr_jobs.append(curr_job)
        former_jobs.append(former_job)    
        main_reasons.append(main_reason)
    #     print("---")
    qa_dict = {
        "videoId": video_ids,
        "curr_job": curr_jobs,
        'former_job': former_jobs,
        'main_reason':main_reasons
    }
    qa_df = pd.DataFrame(qa_dict)
    
    return qa_df

qa_df = create_qa_df(qa_json)
full_df = youtube_df.merge(qa_df, on='videoId', how='left')
full_df.update(full_df[['curr_job','former_job','main_reason']].fillna("N/A"))

st.title('Why YouTubers Quit - Text Analysis')
st. markdown("""
This app examines transcripts from YouTube videos about quitting their jobs and performs NLP Tasks such as:

""")


max_views = st.sidebar.slider(
    'Select a range of view counts to display:',
    int(youtube_df['viewCount'].min()), 
    int(youtube_df['viewCount'].max()),
    int(youtube_df['viewCount'].median()))

max_wordcloud_words = st.sidebar.slider(
    'Select number of words to display in wordclouds:',
    1, 
    100,
    50)

curr_df = full_df[full_df['viewCount'] < max_views]
curr_df.shape
# st.dataframe(curr_df)
fig = px.scatter(curr_df, x="commentCount", y="viewCount", color="categoryId", 
                    hover_data=['title'], 
                    log_x = True, log_y= True)
st.plotly_chart(fig)
curr_ids = set(curr_df['channelId'].unique())

curr_reason_str = " ".join(curr_df.main_reason.tolist())
former_jobs_str = " ".join(curr_df.former_job.tolist())
curr_jobs_str = " ".join(curr_df.curr_job.tolist())
column1, column2, column3 = st.columns(3)
with column1:
    st.subheader("Quitting reasons - keywords")
    wordcloud = WordCloud(background_color = "black", max_words =
                max_wordcloud_words,stopwords = full_stopwords).generate(curr_reason_str)
    fig = plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    st.pyplot(fig)
with column2:
    st.subheader("Former job - keywords")
    wordcloud = WordCloud(background_color = "black", max_words =
            max_wordcloud_words,stopwords = full_stopwords).generate(former_jobs_str)
    fig = plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    st.pyplot(fig)
with column3:
    st.subheader("Current job - keywords")
    wordcloud = WordCloud(background_color = "black", max_words =
            max_wordcloud_words,stopwords = full_stopwords).generate(curr_jobs_str)
    fig = plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    st.pyplot(fig)





