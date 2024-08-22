# read json fil, a list of (seq, rating) and create a json list of (seq, rating_gpt, rating_bert)

import json

file = 'data/sentiment/synthetic_data.json'
with open(file, 'r') as f:
    data = json.load(f)

new_data = []
for d in data:
    new_data.append([d[0], int(5 * d[1]) / 5, d[1]])

new_file = 'data/sentiment/synthetic_data.json'
with open(new_file, 'w') as f:
    json.dump(new_data, f)
    