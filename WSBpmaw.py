# -*- coding: utf-8 -*-
# Require packate:
# pip install pmaw

import datetime as dt
import pandas as pd
from tqdm import tqdm
from pmaw import PushshiftAPI

# Create time filter
before = int(dt.datetime(2021,1,12,0,0).timestamp()) # timestamp in seconds


# Create subreddit and parameters
subreddit="wallstreetbets"
size = 500 # maximum submission size = 500
min_score = '>100' # > here is >=
api = PushshiftAPI()
limit = 1000000

# # Download by smaller step to avoid imcomplete result (too man result due to PushShift shards are down???)
# for i in tqdm(range(73)):
#     after = before - (i+1)*60*60*24*1
#     submissions = api.search_submissions(score=min_score, sort='desc', sort_type='score', subreddit=subreddit, size=size, before=before, after=after, limit=limit)
#     submissions_df = pd.DataFrame(submissions)
#     text_date = dt.datetime.utcfromtimestamp(after).strftime("%d-%m-%Y")
#     print(text_date)
#     submissions_df.to_csv('submissions-'+text_date+'.csv', header=True, index=False, columns=list(submissions_df.axes[1]), encoding='utf-8-sig')
    
#------------------------------------COMMENTS------------------------------------#
# Create subreddit and parameters
subreddit="wallstreetbets"
size = 500 # maximum submission size = 500
min_score = '>50' # > here is >=
api = PushshiftAPI()
limit = 1000000    
for i in tqdm(range(365)):
    after = before - (i+1)*60*60*24*1
    comments = api.search_comments(score=min_score, sort='desc', sort_type='score', subreddit=subreddit, size=size, before=before, after=after, limit=limit)
    comments_df = pd.DataFrame(comments)
    text_date = dt.datetime.utcfromtimestamp(after).strftime("%d-%m-%Y")
    print(text_date)
    comments_df.to_csv('comments-'+text_date+'.csv', header=True, index=False, columns=list(comments_df.axes[1]), encoding='utf-8-sig')    