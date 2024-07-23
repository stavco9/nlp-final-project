# take the json file and edit it to the format that we want
# file vids_sentiment.json
# turn comments_controversy_GPT from 0 - 10 to -1 to 1
# turn comments_controversy_classical from 0 - 1 to -1 to 1

import json
import numpy as np
import os

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


edit_json2('data/data/vids_sentiment.json', 'data/data/videos_classical_sentiment_adjusted.json')

#edit_json('data/data/vids_sentiment.json')