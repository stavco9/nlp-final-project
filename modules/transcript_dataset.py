from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled,NoTranscriptFound
from youtube_transcript_api.formatters import JSONFormatter
from youtubesearchpython import VideosSearch
from pythonryd import *
import requests
import isodate
import json
import os

MAX_API_COUNT=500
YOUTUBE_VIDEOS_CHUNK=50
YOUTUBE_COMMENTS_CHUNK=100

class Video:
    def __init__(self, video_id, video_obj):
        base_video_url = 'https://www.youtube.com/watch?v='
        dur = isodate.parse_duration(video_obj['contentDetails']['duration'])
        statistics = video_obj['statistics']
        try:
            video_with_dislikes = ryd_getvideoinfo(video_id)
        except Exception as e:
            video_with_dislikes = {}
        self.video_formatted = {
            'id': video_id,
            'videoLink': base_video_url + video_id,
            'title': video_obj['snippet']['title'],
            'description': video_obj['snippet']['description'],
            'channelId': video_obj['snippet']['channelId'],
            'channelTitle': video_obj['snippet']['channelTitle'],
            'durationSeconds': int(dur.total_seconds()),
            'publishedDate': video_obj['snippet']['publishedAt'],
            'likeCount': int(statistics['likeCount']) if statistics.get('likeCount') else 0,
            'dislikeCount': video_with_dislikes.get('dislikes') or 0,
            'commentCount': int(statistics['commentCount']) if statistics.get('commentCount') else 0,
            'favoriteCount': int(statistics['favoriteCount']) if statistics.get('favoriteCount') else 0,
            'viewCount': int(statistics['viewCount']) if statistics.get('viewCount') else 0,
            'comments': []
        }

class Comments:
    def __init__(self, comment_id, comment_obj):
        comment_obj_snippet = comment_obj['snippet']['topLevelComment']['snippet']
        self.comment_formatted = {
            'id': comment_id,
            'text': comment_obj_snippet['textOriginal'],
            'publishedDate': comment_obj_snippet['publishedAt'],
            'likeCount': int(comment_obj_snippet['likeCount']) if comment_obj_snippet.get('likeCount') else 0
        }

class TranscriptDataset:
    def __init__(self, api_key) -> None:
        self.videos = {}
        self.all_video_ids = []
        self.api_key = api_key

    @staticmethod
    def get_chunks_list(total_count, chunk):
        chunks = []
        for i in range(0, total_count, chunk):
            chunks.append(min(i+chunk, total_count) - i)
        return chunks

    def build_dataset(self, dataset_folder, videos_count=100, language='en', country_code='us'):
        self.language = language
        self.country_code = country_code

        counter = 0
        file_counter, self.all_video_ids = self.load_ids(dataset_folder)
        print(f"There are currnet {len(self.all_video_ids)} videos in {file_counter} files")
        if videos_count > len(self.all_video_ids):
            print(f"Fetching now {videos_count - len(self.all_video_ids)}")
            while file_counter*min(MAX_API_COUNT, videos_count) < videos_count or len(self.videos) > 0:
                print(f"{counter+1} time - Fecthing videos")
                self.get_videos_in_channel(counter, videos_count)
                counter+=1
                if len(self.videos) >= min(MAX_API_COUNT, videos_count):
                    print(f"Reached {len(self.videos)} videos. Saving...")
                    self.all_video_ids.extend(list(self.videos.keys()))
                    if not os.path.exists(dataset_folder):
                        os.makedirs(dataset_folder)
                    self.save(os.path.join(dataset_folder, f"dataset_{file_counter}.json"))
                    self.videos = {}
                    file_counter +=1
        else:
            print(f"There are already at least {len(self.all_video_ids)}. No need to do anything")

    def get_videos_in_channel(self, counter, videos_count):
        base_search_url = 'https://www.googleapis.com/youtube/v3/search?'

        chunks = self.get_chunks_list(min(MAX_API_COUNT, videos_count), YOUTUBE_VIDEOS_CHUNK)

        first_url = base_search_url+ f'key={self.api_key}&type=video&part=snippet,id&order=relevance&relevanceLanguage={self.language}&regionCode={self.country_code}&videoDuration=medium&maxResults={min(MAX_API_COUNT, self.videos_count)}'
        url = first_url

        for idx, chunk in enumerate(chunks):
            print(f"{counter+1} time -> {idx+1} chunk out of {len(chunks)}")
            inp = requests.get(url)
            if not inp.status_code == 200:
                raise Exception(f"Status code is {inp.status_code}, reason is: {inp.json()}")
            try:
                resp = inp.json()
                count = 0
                list_ids = []
                for video in resp['items']:
                    if count >= chunk:
                        break
                    video_id = video['id']['videoId']
                    list_ids.append(video_id)
                    count+=1
                self.get_videos_by_ids(list_ids, videos_count)

                print(f"{counter+1} time -> Total number of videos: {len(self.videos)}")

                try:
                    next_page_token = resp['nextPageToken']
                    url = first_url + '&pageToken={}'.format(next_page_token)
                except:
                    break
            except requests.JSONDecodeError as e:
                raise Exception(f"Resonse output is not json: {e}")
    
    def get_videos_by_ids(self, video_ids, videos_count):
        base_videos_url = 'https://www.googleapis.com/youtube/v3/videos?'
        url = base_videos_url+'key={}&id={}&part=snippet,id,contentDetails,statistics'.format(self.api_key, ','.join(video_ids))
        inp = requests.get(url)
        if not inp.status_code == 200:
            raise Exception(f"Status code is {inp.status_code}, reason is: {inp.json()}")
            
        try:
            resp = inp.json()

            for video in resp['items']:
                video_id = video['id']
                if not video_id in self.all_video_ids:
                    self.videos[video_id] = Video(video_id, video).video_formatted
                    self.download_transcript(video_id)

                    if (len(self.videos) == min(MAX_API_COUNT, videos_count)):
                        break
        except requests.JSONDecodeError as e:
            raise Exception(f"Resonse output is not json: {e}")

    def download_transcript(self, video_id):
        try:
            video_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[self.language])
            self.videos[video_id]['transcript'] = video_list
        except TranscriptsDisabled as e:
            # We do not include videos without a transcript
            self.videos.pop(video_id)
        except NoTranscriptFound as e:
            # We do not include videos without a transcript
            self.videos.pop(video_id)

    def build_comments(self, dataset_folder, comments_count=100):
        dataset_files = [f for f in os.listdir(dataset_folder) if 
                         os.path.isfile(os.path.join(dataset_folder, f)) and f.endswith('.json')]

        for file in dataset_files:
            with open(os.path.join(dataset_folder, file), 'r') as f:
                print(f"Working on file {file}")
                self.videos = json.loads(f.read())

            for video_id in list(self.videos.keys()):
                if int(self.videos[video_id]['commentCount']) > 0 and \
                    ( 
                        not self.videos[video_id].get('comments') or \
                        len(self.videos[video_id]['comments']) == 0 or \
                        (
                            len(self.videos[video_id]['comments']) < comments_count and \
                            int(self.videos[video_id]['commentCount']) > comments_count*2
                        )
                    ):
                    print(f"Fetching comments of {video_id}")
                    self.videos[video_id]['comments'] = self.get_comments_in_video(video_id, comments_count)
            
            self.save(os.path.join(dataset_folder, file))

    def get_comments_in_video(self, video_id, comments_count):
        base_search_url = 'https://www.googleapis.com/youtube/v3/commentThreads?'

        chunks = self.get_chunks_list(comments_count, YOUTUBE_COMMENTS_CHUNK)

        list_comments = self.videos[video_id].get('comments') or []
        already_exist_comment_ids = [comment['id'] for comment in list_comments]
        first_url = base_search_url+ f'key={self.api_key}&type=video&part=snippet,id&order=relevance&videoId={video_id}&maxResults={comments_count}'
        url = first_url

        for idx, chunk in enumerate(chunks):
            inp = requests.get(url)
            if not inp.status_code == 200:
                if inp.status_code != 404:
                    raise Exception(f"Status code is {inp.status_code}, reason is: {inp.json()}")
                else:
                    print(f"Video {video_id} not found. Returning empty list")
                    return []
            try:
                resp = inp.json()
                count = 0
                for comment in resp['items']:
                    if count >= chunk:
                        break
                    comment_id = comment['id']
                    if not comment_id in already_exist_comment_ids:
                      list_comments.append(Comments(comment_id, comment).comment_formatted)
                    count+=1

                try:
                    next_page_token = resp['nextPageToken']
                    url = first_url + '&pageToken={}'.format(next_page_token)
                except:
                    break
            except requests.JSONDecodeError as e:
                raise Exception(f"Resonse output is not json: {e}")
            
        return list_comments

    def load_ids(self, folder_name):
        all_video_ids = []
        dataset_files = [f for f in os.listdir(folder_name) if 
                         os.path.isfile(os.path.join(folder_name, f)) and f.endswith('.json')]

        for file in dataset_files:
            with open(os.path.join(folder_name, file), 'r') as f:
                dataset_file = json.loads(f.read())
                all_video_ids.extend(list(dataset_file.keys()))
        return len(dataset_files), all_video_ids

    def save(self, file_name):
        with open(file_name, 'w') as output:
            json.dump(self.videos, output, indent=4, sort_keys=True)