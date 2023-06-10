# youtube_quitting_data


## Summary

Purpose is to collect and analyze transcripts from YouTube videos explaining why they quit their jobs. Techniques will include text natural language processing tasks such as question answering, text summarization, and visualizing text data. First draft is to focus on the pipeline running, with further experimentation to address runtime and code clarity. 

### Methodology

- YouTube API v3 - collect video ids when searching for a text query
- YouTubeTranscript API - returns transcript strings when given URL
- Text summarization and question Answering using ChatGPT: text-davinci-003


### Steps

01.
    a. scrape youtube data using YouTube API - iterate through pages of video IDs and supporting information, collect video ids and supplementary information
    b. Iterate through relevant video IDs, and collect transcript text from each video ID using YoutubeTranscriptAPI
    c. combine into JSON: youtube_stats_and_transcripts.json where each item type has it's own list of items containing that information
02 
    a. Run ChatGPT to summarize chunks of transcripts to reduce token count (ChatGPT has max count of 2048)
    b. Iterate through summaries to answer 3 questions (all in 1 query, separated by newline char)
        - What was the speaker's former job/occupation?
        - What is the speaker's current job/occupation?
        - What is the primary reason for leaving their previous job?
    c. Output into JSON: final_100_QA.json where each key is a video id and the item is the text of the response for that video
    
3. 



#### Refactor: 
- rerun create emotions_df with processed_ALL.json
- rerun qa_working_script.ipynb (long run)

#### Limitations
- Videos returned by YouTube API are not all about quitting work (ie: quitting smoking, quitting and moving to another platform)
- Limitations of certain prompts 
    - search query used in step 1
    - 3 questions used in QA model 

