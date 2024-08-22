import numpy as np
from modules.sentiment_analysis.video_info import Video_info

def score_inplace(scores):
    # Number of bins (10% segments)
    num_bins = 10

    # Get the percentile boundaries
    percentiles = np.percentile(scores, np.arange(0, 100 + 100/num_bins, 100/num_bins))

    # Apply the transformation
    for i in range(num_bins):
        if i < num_bins - 1:
            # Create a mask for the current segment
            mask = (scores >= percentiles[i]) & (scores < percentiles[i + 1])
        else:
            # Last bin includes the maximum score
            mask = (scores >= percentiles[i]) & (scores <= percentiles[i + 1])
        
        if percentiles[i + 1] > percentiles[i]:
            # Normalize the scores within this segment
            scores[mask] = (scores[mask] - percentiles[i]) / (percentiles[i + 1] - percentiles[i])
        
        # Shift the normalized scores to the correct bin range
        scores[mask] = scores[mask] + i
    
    return scores/5 - 1


def score_objects(videos):
    all_scores_likes = []
    all_scores_comments = []
    
    for vid in videos:
        views = vid.views
        # views must be at least 10000, otherwise raise an error
        if views < 10000:
            raise ValueError("Views must be at least 10000")
        rel_likes = vid.likes / views
        rel_dislikes = vid.dislikes / views
        rel_num_comments = len(vid.comments) / views
        rel_comments_scores_classical_var = np.var(vid.comments_scores_classical) if vid.comments_scores_classical else 0

        all_scores_likes.append(rel_likes * rel_dislikes)
        all_scores_comments.append(rel_num_comments * rel_comments_scores_classical_var)

    # min max normalization
    min_score = min(all_scores_likes)
    max_score = max(all_scores_likes)
    for i, score in enumerate(all_scores_likes):
        all_scores_likes[i] = ((score - min_score) / (max_score - min_score))

    min_score = min(all_scores_comments)
    max_score = max(all_scores_comments)
    for i, score in enumerate(all_scores_comments):
        all_scores_comments[i] = ((score - min_score) / (max_score - min_score))
    
    all_scores = [likes_score + comments_score for likes_score, comments_score in zip(all_scores_likes, all_scores_comments)]
    all_scores = score_inplace(np.array(all_scores)).tolist()

    return all_scores

def adjust_disagreement_rating(videos):
    all_scores = score_objects(videos)
    for i, video in enumerate(videos):
        video.comments_controversy_classical = all_scores[i]
    return videos
