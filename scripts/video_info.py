import json
class Video_info:
    def __init__(self, id, watches, likes, dislikes, comments):
        self.id = id
        self.watches = watches
        self.likes = likes
        self.dislikes = dislikes
        self.comments = comments
        self.comment_scores = [comment['sentiment'] for comment in comments]

def json_to_video_info(json_file):
    videos = []
    json_data = json.load(json_file)
    for video in json_data:
        videos.append(Video_info(video['id'], video['watches'], video['likes'], video['dislikes'], video['comments']))
    return videos