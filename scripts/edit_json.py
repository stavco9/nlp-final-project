# take the json file and edit it to the format that we want
# file vids_sentiment.json
# turn comments_controversy_GPT from 0 - 10 to -1 to 1
# turn comments_controversy_classical from 0 - 1 to -1 to 1

import json
import numpy as np
import os

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

def edit_json(json_file):
    transcript_and_scores = []
    with open(json_file, 'r') as f:
        data = json.load(f)
    for i in range(len(data)):
        ccg = data[i]['comments_controversy_GPT']
        ccc = data[i]['comments_controversy_classical']
        # if ccg is not int the delete the data at that index
        if type(ccg) == int:
            ccg = (ccg - 5) / 5
            ccc = (ccc - 0.5) / 0.5
            transcript_and_scores.append((data[i]['transcript'], ccg, ccc))
    with open(json_file, 'w') as f:
        json.dump(transcript_and_scores, f)
    
def edit_json2(json_file1, json_file2):
    transcript_and_scores = []
    with open(json_file1, 'r') as f:
        old_data = json.load(f)
    with open(json_file2, 'r') as f:
        new_data = json.load(f)
    for i in range(len(old_data)):
        ccg = old_data[i][1]
        ccc = new_data[i]['comments_controversy_classical']
        # if ccg is not float the delete the data at that index
        if type(ccg) == float and type(ccc) == float:
            transcript_and_scores.append((old_data[i][0], ccg, ccc))
        else:
            print(ccg)
            print(ccc)
            pause = input("Press enter to continue")
            print(f"Deleted index {i}")
    cccs = [x[2] for x in transcript_and_scores]
    median_score_ccc = np.median(cccs)
    max_score_ccc = max(cccs) - median_score_ccc
    min_score_ccc = min(cccs) - median_score_ccc
    # adjust the score so the median is the average
    for i in range(len(transcript_and_scores)):
        ccc = transcript_and_scores[i][2] - median_score_ccc
        if ccc > 0:
            ccc = ccc / max_score_ccc
        else:
            ccc = ccc / min_score_ccc
        transcript_and_scores[i] = (transcript_and_scores[i][0], transcript_and_scores[i][1], ccc)
    with open('data/data/vids_sentiment_fixed.json', 'w') as f:
        json.dump(transcript_and_scores, f)
    print("Done")

def edit_json3(json_file1, json_file2):

    with open(json_file1, 'r') as f:
        videos = json.load(f)

    with open(json_file2, 'r') as f:
        old = json.load(f)

    all_scores_likes = []
    all_scores_comments = []
    
    for vid in videos:
        views = vid["views"]
        # views must be at least 10000, otherwise raise an error
        if views < 10000:
            raise ValueError("Views must be at least 10000")
        rel_likes = vid["likes"] / views
        rel_dislikes = vid["dislikes"] / views
        rel_num_comments = vid["commentsCount"] / views
        rel_comments_scores_classical_var = np.var(vid["comments_scores_classical"])

        all_scores_likes.append(rel_likes * rel_dislikes)
        all_scores_comments.append(rel_num_comments * rel_comments_scores_classical_var)

    all_scores_likes = score_inplace(np.array(all_scores_likes)).tolist()

    all_scores_comments = score_inplace(np.array(all_scores_comments)).tolist()

    all_scores = [(likes_score + comments_score)/2 for likes_score, comments_score in zip(all_scores_likes, all_scores_comments)]

    data = []
    
    diff = 0
    # saves the scores and sequences to the json file
    for i in range(len(old)):
        while old[i][0] != videos[i+diff]["transcript"]:
            diff += 1
        text = old[i][0]
        ccg = old[i][1]
        ccc = all_scores[i+diff]
        data.append((text, ccg, ccc))

    with open('data/data/vids_sentiment_fixed_new.json', 'w') as f:
        json.dump(data, f)

def edit_json4(json_file1):
    with open(json_file1, 'r') as f:
        vids_sentiment = json.load(f)

    ccg = [x[1] for x in vids_sentiment]
    ccc = [x[2] for x in vids_sentiment]

    ccg = score_inplace(np.array(ccg))
    ccc = score_inplace(np.array(ccc))

    data = []
    for i in range(len(vids_sentiment)):
        data.append((vids_sentiment[i][0], ccg[i], ccc[i]))

    with open('data/sentiment/synthetic_data.json', 'w') as f:
        json.dump(data, f)

def add_syntheticGPT(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    for i in range(len(data)):
        data[i] = (data[i][0], int(10 * data[i][1])/10, data[i][1])

    with open(json_file, 'w') as f:
        json.dump(data, f)

add_syntheticGPT('data/sentiment/synthetic_data_short.json')

#edit_json4('data/sentiment/synthetic_data.json')

#edit_json3('data/data/vids_classical_sentiment.json', 'data/data/vids_sentiment.json')

#edit_json('data/data/vids_sentiment.json')