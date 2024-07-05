from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled,NoTranscriptFound
from youtube_transcript_api.formatters import JSONFormatter
from youtubesearchpython import VideosSearch
from pythonryd import *
import requests
import isodate
import json

MAX_API_COUNT=500
YOUTUBE_VIDEOS_CHUNK=50

class Video:
    def __init__(self, video_id, video_obj):
        base_video_url = 'https://www.youtube.com/watch?v='
        dur = isodate.parse_duration(video_obj['contentDetails']['duration'])
        statistics = video_obj['statistics']
        video_with_dislikes = ryd_getvideoinfo(video_id)
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

class TranscriptDataset:
    def __init__(self, api_key, videos_count=100, language='en', country_code='us') -> None:
        self.videos = {}
        self.all_video_ids = []
        self.language = language
        self.country_code = country_code
        self.videos_count = videos_count
        self.api_key = api_key
        self.build_dataset()

    @staticmethod
    def get_chunks_list(total_count, chunk):
        chunks = []
        for i in range(0, total_count, chunk):
            chunks.append(min(i+chunk, total_count) - i)
        return chunks

    def build_dataset(self):
        counter = 0
        file_counter = 0
        while file_counter*min(MAX_API_COUNT, self.videos_count) < self.videos_count or len(self.videos) > 0:
            print(f"{counter+1} time - Fecthing videos")
            self.get_videos_in_channel(counter)
            counter+=1
            if len(self.videos) >= min(MAX_API_COUNT, self.videos_count):
                print(f"Reached {len(self.videos)} videos. Saving...")
                self.all_video_ids.extend(list(self.videos.keys()))
                self.save(f"dataset_{file_counter}.json")
                self.videos = {}
                file_counter +=1

    def get_videos_in_channel(self, counter):
        base_search_url = 'https://www.googleapis.com/youtube/v3/search?'

        chunks = self.get_chunks_list(min(MAX_API_COUNT, self.videos_count), YOUTUBE_VIDEOS_CHUNK)

        first_url = base_search_url+ f'key={self.api_key}&type=video&part=snippet,id&order=relevance&relevanceLanguage={self.language}&regionCode={self.country_code}&videoDuration=medium&maxResults={min(MAX_API_COUNT, self.videos_count)}'
        url = first_url

        for idx, chunk in enumerate(chunks):
            print(f"{counter+1} time -> {idx+1} chunk out of {len(chunks)}")
            inp = requests.get(url)
            if not inp.status_code == 200:
                raise Exception(f"Status code is {inp.status_code}")
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
                self.get_videos_by_ids(list_ids)

                print(f"{counter+1} time -> Total number of videos: {len(self.videos)}")

                try:
                    next_page_token = resp['nextPageToken']
                    url = first_url + '&pageToken={}'.format(next_page_token)
                except:
                    break
            except requests.JSONDecodeError as e:
                raise Exception(f"Resonse output is not json: {e}")
    
    def get_videos_by_ids(self, video_ids):
        base_videos_url = 'https://www.googleapis.com/youtube/v3/videos?'
        url = base_videos_url+'key={}&id={}&part=snippet,id,contentDetails,statistics'.format(self.api_key, ','.join(video_ids))
        inp = requests.get(url)
        if not inp.status_code == 200:
            raise Exception(f"Status code is {inp.status_code}")
            
        try:
            resp = inp.json()

            for video in resp['items']:
                video_id = video['id']
                if not video_id in self.all_video_ids:
                    self.videos[video_id] = Video(video_id, video).video_formatted
                    self.download_transcript(video_id)

                    if (len(self.videos) == min(MAX_API_COUNT, self.videos_count)):
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

    def save(self, file_name):
        with open(file_name, 'w') as output:
            json.dump(self.videos, output, indent=4, sort_keys=True)