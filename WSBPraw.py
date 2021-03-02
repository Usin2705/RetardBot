import praw
from pandas import DataFrame
import io
import password



reddit = praw.Reddit(client_id = password.REDDIT_CLIENT_ID,
                     client_secret = password.REDDIT_CLIENT_SECRET,
                     password = password.REDDIT_PASSWORD,
                     user_agent = password.REDDIT_AGENT,
                     username= password.REDDIT_USERNAME)




#submission = reddit.get_subreddit('wallstreetbets').get_new(limit=0)
subreddit = reddit.subreddit('wallstreetbets')


posts = subreddit.top("all", limit=2)

data_comments = []
#for comment in subreddit.comments(limit=2):
for comment in subreddit.top():
    #print(comment.body)
    data_comments.append(comment.body)

# https://stackoverflow.com/a/43684587/14207179
#df  = DataFrame(data_comments, columns=['Comment'])
#df.to_csv('05-02-2021-2.csv', encoding='utf-8-sig')


#with io.open('04-02-2021.txt', "w", encoding="utf-8") as f:
#    for item in data_comments:
#         f.write("%s\n" % item)    