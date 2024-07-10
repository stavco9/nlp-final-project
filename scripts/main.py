from comments_sentiment_analysis import multi_sentiment_score
from video_info import Video_info
from video_info import datumToObj
from video_info import save_to_json
from video_info import save_to_pickle
from GPT_comment_controversy import init_openai
from GPT_comment_controversy import get_disagreement_rating
from scoring_video import adjust_disagreement_rating

def main():
    # put all the json into pickle
    datasets = [f'data/data/dataset_{i}.json' for i in range(10)]
    #datasets = ['data/data/dataset_0_sample.json']
    basic_inf_vids = datumToObj(datasets, 'data/data/videos_sample.pkl')
    sentiment_model = multi_sentiment_score
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    for vid in basic_inf_vids:
        vid.add_comments_scores(sentiment_model, model_name)
    client = init_openai()
    sentiment_model=get_disagreement_rating
    model_name="GPT"
    for vid in basic_inf_vids:
        vid.add_comments_scores(sentiment_model, model_name, client)
    vids = adjust_disagreement_rating(basic_inf_vids)

    save_to_json(vids, 'data/data/videos_sample_classical.json')

    print("Done")

main()