from comments_sentiment_analysis import multi_sentiment_score
from video_info import Video_info
from video_info import datumToObj
from video_info import save_to_json
from video_info import save_to_pickle
from GPT_comment_controversy import init_openai
from GPT_comment_controversy import get_disagreement_rating
from scoring_video import adjust_disagreement_rating
import json

def main():
    # put all the json into pickle
    datasets = [f'data/data/dataset_{i}.json' for i in range(10)]
    #datasets = ['data/data/dataset_0_sample.json']
    to_print = input("Do you want to show prints? (y/n): ") != 'n'
    basic_inf_vids = datumToObj(datasets, 'data/data/videos_sample.pkl', to_print)
    number_of_vids = len(basic_inf_vids)

    save_to_json(basic_inf_vids, 'data/data/vids_no_sentiment.json')
    sentiment_model = multi_sentiment_score
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    count = 0
    for vid in basic_inf_vids:
        print_text = "\r"
        if to_print:
            count += 1
            print_text += f"Processing video {count}/{number_of_vids}"
        vid.add_comments_scores(sentiment_model, model_name)
        if to_print:
            print_text += f" - Disagreement rating: {vid.comments_controversy_classical}"
        if to_print:
            print(print_text)

    save_to_json(basic_inf_vids, 'data/data/vids_classical_sentiment.json')
    basic_inf_vids = adjust_disagreement_rating(basic_inf_vids)
    save_to_json(basic_inf_vids, 'data/data/videos_classical_sentiment_adjusted.json')
    '''

    #load vids from videos_classical_sentiment_adjusted.json and adjust basic_inf_vids with its values
    with open('data/data/videos_classical_sentiment_adjusted.json') as json_file:
        json_obj = json.load(json_file)
    for i, video_obj in enumerate(json_obj):
        basic_inf_vids[i].comments_controversy_classical = float(video_obj['comments_controversy_classical'])
    '''

    
    to_continue = input("Do you want to continue with GPT? (y/n): ")
    if to_continue == 'n':
        print("Done")
        return

    count = 0
    client = init_openai()
    sentiment_model=get_disagreement_rating
    model_name="GPT"
    for vid in basic_inf_vids:
        print_text = "\r"
        if to_print:
            count += 1
            print_text += f"Processing video {count}/{number_of_vids}"
        vid.add_comments_scores(sentiment_model, model_name, client)
        if to_print:
            print_text += f" - Disagreement rating: {vid.comments_controversy_GPT}"
        if to_print:
            print(print_text)
    save_to_json(basic_inf_vids, 'data/data/vids_GPT_sentiment.json')

    print("Done")

main()