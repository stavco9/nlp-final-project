import numpy as np
from scripts.video_info import Video_info

class Video_score:
    def __init__(self, id, score):
        self.video = id
        self.score = score

def compute_mean_variation(videos : Video_info):
    watches = np.array([vid.watches for vid in videos])
    likes = np.array([vid.likes for vid in videos]) / watches
    dislikes = np.array([vid.dislikes for vid in videos]) / watches
    num_comments = np.array([len(vid.comments) for vid in videos]) / watches
    avg_comment_scores = np.array([np.mean(vid.comment_scores) if vid.comment_scores else 0 for vid in videos]) / watches

    means = {
        'likes': np.mean(likes),
        'dislikes': np.mean(dislikes),
        'num_comments': np.mean(num_comments),
        'avg_comment_scores': np.mean(avg_comment_scores)
    }

    variations = {
        'likes': np.std(likes),
        'dislikes': np.std(dislikes),
        'num_comments': np.std(num_comments),
        'avg_comment_scores': np.std(avg_comment_scores)
    }

    return means, variations

def score_objects(videos, means, variations):
    video_scores = []
    
    for vid in videos:
        watch = vid.watches
        rel_likes = vid.likes / watch
        rel_dislikes = vid.dislikes / watch
        rel_num_comments = len(vid.comments) / watch
        rel_avg_comment_scores = np.mean(vid.comment_scores) / watch if vid.comment_scores else 0

        z_scores = {
            'likes': (rel_likes - means['likes']) / variations['likes'] if variations['likes'] != 0 else 0,
            'dislikes': (rel_dislikes - means['dislikes']) / variations['dislikes'] if variations['dislikes'] != 0 else 0,
            'num_comments': (rel_num_comments - means['num_comments']) / variations['num_comments'] if variations['num_comments'] != 0 else 0,
            'avg_comment_scores': (rel_avg_comment_scores - means['avg_comment_scores']) / variations['avg_comment_scores'] if variations['avg_comment_scores'] != 0 else 0
        }

        mod_z_scores = {k: abs(v) + 1 for k, v in z_scores.items()}

        score = mod_z_scores['likes'] * mod_z_scores['dislikes'] * mod_z_scores['num_comments'] * mod_z_scores['avg_comment_scores']
        video_scores.append(Video_score(vid.id, score))

    # min max normalization
    min_score = min([score.score for score in video_scores])
    max_score = max([score.score for score in video_scores])
    for score in video_scores:
        video_scores.score = (score.score - min_score) / (max_score - min_score) * 10

    return video_scores
