import numpy as np
from video_info import Video_info

def compute_mean_variation(videos : list):
    views = np.array([vid.views for vid in videos], dtype=int)
    likes = np.array([vid.likes for vid in videos], dtype=int) / views
    dislikes = np.array([vid.dislikes for vid in videos], dtype=int) / views
    num_comments = np.array([len(vid.comments) for vid in videos], dtype=int) / views
    avg_comment_scores = np.array([np.mean(vid.comments_scores_classical) if vid.comments_scores_classical else 0 for vid in videos], dtype=float) / views

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
    all_scores = []
    
    for vid in videos:
        views = vid.views
        rel_likes = vid.likes / views
        rel_dislikes = vid.dislikes / views
        rel_num_comments = len(vid.comments) / views
        rel_avg_comment_scores = np.mean(vid.comments_scores_classical) / views if vid.comments_scores_classical else 0

        z_scores = {
            'likes': (rel_likes - means['likes']) / variations['likes'] if variations['likes'] != 0 else 0,
            'dislikes': (rel_dislikes - means['dislikes']) / variations['dislikes'] if variations['dislikes'] != 0 else 0,
            'num_comments': (rel_num_comments - means['num_comments']) / variations['num_comments'] if variations['num_comments'] != 0 else 0,
            'avg_comment_scores': (rel_avg_comment_scores - means['avg_comment_scores']) / variations['avg_comment_scores'] if variations['avg_comment_scores'] != 0 else 0
        }

        modified_z_scores = {k: abs(v) + 1 for k, v in z_scores.items()}

        score = modified_z_scores['likes'] * modified_z_scores['dislikes'] + modified_z_scores['num_comments'] * modified_z_scores['avg_comment_scores']
        all_scores.append(score)

    # min max normalization
    min_score = min(all_scores)
    max_score = max(all_scores)
    for i, score in enumerate(all_scores):
        all_scores[i] = (score - min_score) / (max_score - min_score) * 10

    return all_scores

def adjust_disagreement_rating(videos):
    means, variations = compute_mean_variation(videos)
    all_scores = score_objects(videos, means, variations)
    for video in videos:
        video.comments_controversy_classical = all_scores.pop(0)
    return videos
