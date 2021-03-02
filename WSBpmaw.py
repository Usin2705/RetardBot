# -*- coding: utf-8 -*-
# Require packate:
# pip install pmaw

import datetime as dt
import pandas as pd
import numpy as np
from tqdm import tqdm
from pmaw import PushshiftAPI

# Create time filter
before = int(dt.datetime(2021,1,12,0,0).timestamp()) # timestamp in seconds
# Create subreddit and parameters
subreddit="wallstreetbets"
size = 500 # maximum submission size = 500
api = PushshiftAPI()
limit = 1000000
test_list_data = []
total_results_df = []

#------------------------------------PARAMETERS EDIT------------------------------------# 
min_score = '>40' # > here is >=  #Sumission score 40, comment score 10
time_range = 365*2
download_per_loop = 10 # days
IS_SUBMISSION = True


#Download by smaller step to avoid imcomplete result (too many resuls coulde be lost due to PushShift shards are down???)
for i in tqdm(range(int(time_range/download_per_loop)+2)):    
    after = before - 60*60*24*download_per_loop  # download per day
    
    if IS_SUBMISSION:
        file_name = 'submissions'
        results = api.search_submissions(score=min_score, sort='desc', 
                                         sort_type='score', 
                                         subreddit=subreddit, 
                                         size=size, 
                                         before=before, 
                                         after=after, 
                                         limit=limit)
        results_df = pd.DataFrame(results)
        
        if not results_df.empty:        
            # Remove anypost that was removed by any (post that was not removed has this value as nan)
            if hasattr(results_df, 'removed_by_category'):
                results_df = results_df[results_df.removed_by_category.isnull()] 
            
            text_list_title = results_df.title.tolist()     
            text_list_content = results_df.selftext.replace('', np.nan) #Replace empty text with nan
            text_list_content = text_list_content.dropna().tolist()  #Drop empty row (a lot of submission don't have text content)        
            test_list_data = test_list_data + text_list_title + text_list_content        
    
    else:
        file_name = 'comments'
        results = api.search_comments(score=min_score, sort='desc', 
                                      sort_type='score', 
                                      subreddit=subreddit, 
                                      size=size, 
                                      before=before, 
                                      after=after, 
                                      limit=limit)
        results_df = pd.DataFrame(results)
        
        # Extract text to list if result is not empty
        if not results_df.empty:
            text_list = results_df.body.tolist()        
            test_list_data = test_list_data + text_list

            
    if not results_df.empty:
        if len(total_results_df) == 0:
            total_results_df = results_df.copy(deep=True)
        else:
            total_results_df = total_results_df.append(results_df)
        
    text_date_before = dt.datetime.utcfromtimestamp(before).strftime("%Y-%m-%d")    
    text_date_after = dt.datetime.utcfromtimestamp(after).strftime("%Y-%m-%d")        
    file_number = str(i).zfill(3)     
    before = after
    
    print("\n---Number: {} before: {}, after: {}, results length: {}".format(file_number,
                                                                           text_date_before,
                                                                           text_date_after,
                                                                           len(test_list_data)))
    
    # results_df.to_csv(file_name+'-'+text_date_after+'.csv', header=True, index=False,
    #                   columns=list(results_df.axes[1]), encoding='utf-8-sig')    
    
df  = pd.DataFrame(test_list_data, columns=['text'])
df.to_csv(file_name+ '.csv', encoding='utf-8-sig')

total_results_df.to_csv(file_name+'-total.csv', header=True, index=False,
                        columns=list(total_results_df.axes[1]), 
                        encoding='utf-8-sig') 