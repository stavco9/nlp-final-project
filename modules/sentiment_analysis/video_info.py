import json
import pickle
import random

COMMENT_LIMIT = 200

class Video_info:
    def __init__(self, id, views, likes, dislikes, transcript, commentsCount, comments):
        # if they are strings, convert them to integers
        if isinstance(views, str):
            views = int(views)
        if isinstance(likes, str):
            likes = int(likes)
        if isinstance(dislikes, str):
            dislikes = int(dislikes)
        if isinstance(commentsCount, str):
            commentsCount = int(commentsCount)
        self.id = id
        self.views = views
        self.likes = likes
        self.dislikes = dislikes
        self.transcript = transcript
        self.commentsCount = commentsCount
        self.comments = comments
        self.comments_scores_classical = []
        self.comments_controversy_classical = 0
        self.comments_controversy_GPT = 0

    def add_comments_scores(self, sentiment_model, model_name, client=None):
        if model_name == "GPT":
            self.comments_controversy_GPT = sentiment_model(client, self.comments)
        else:
            self.comments_scores_classical = sentiment_model(self.comments)

def json_to_video_info(json_file):
    videos = []
    json_dict = json.load(json_file)
    for video_key in json_dict:
        video_obj = json_dict[video_key]
        id = video_obj['id']
        views = video_obj['viewCount']
        likeCount = video_obj['likeCount']
        dislikeCount = video_obj['dislikeCount']
        transcript = ""
        for t in video_obj['transcript']:
            transcript += " " + t['text']
        commentsCount = video_obj['commentCount']
        comments = []
        for comment in video_obj['comments']:
            comments.append(comment['text'])
        if len(comments) > COMMENT_LIMIT:
            comments = random.sample(comments, COMMENT_LIMIT)
        vid_inf = Video_info(id, views, likeCount, dislikeCount, transcript, commentsCount, comments)
        videos.append(vid_inf)
    return videos

def datumToObj(datasets_json_files, pickle_file):
    videos = []
    # if pickle file exists, load from pickle
    try:
        with open(pickle_file, 'rb') as f:
            videos = pickle.load(f)
    except FileNotFoundError:
        pass
    # load from json files
    for file in datasets_json_files:
        with open(file) as json_file:
            videos.extend(json_to_video_info(json_file))
    # save to pickle
    if len(datasets_json_files) > 0:
        with open(pickle_file, 'wb') as f:
            pickle.dump(videos, f)
    return videos

def save_to_pickle(videos, filename):
    with open(filename, 'wb') as f:
        pickle.dump(videos, f)

def save_to_json(videos, filename):
    vids = []
    for vid in videos:
        vid_dict = {
            'id': vid.id,
            'views': vid.views,
            'likes': vid.likes,
            'dislikes': vid.dislikes,
            'transcript': vid.transcript,
            'commentsCount': vid.commentsCount,
            'comments_controversy_classical': vid.comments_controversy_classical,
            'comments_controversy_GPT': vid.comments_controversy_GPT
        }
        vids.append(vid_dict)
    with open(filename, 'w') as f:
        json.dump(vids, f)