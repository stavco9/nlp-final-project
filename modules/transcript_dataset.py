from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from youtube_transcript_api.formatters import JSONFormatter
from youtubesearchpython import VideosSearch
import requests
import isodate
import json

YOUTUBE_VIDEOS_CHUNK=25

class TranscriptDataset:
    def __init__(self, api_key, channels = [], count_per_channel=100, languages = ['en']) -> None:
        self.videos = {}
        self.channels = channels
        self.languages = languages
        self.count_per_channel = count_per_channel
        self.api_key = api_key
        self.build_dataset()

    @staticmethod
    def get_chunks_list(total_count, chunk):
        chunks = []
        for i in range(0, total_count, chunk):
            chunks.append(min(i+chunk, total_count) - i)
        return chunks

    def build_dataset(self):
        for channel in self.channels:
            self.get_videos_in_channel(channel)

    def get_videos_in_channel(self, channel_id):
        base_search_url = 'https://www.googleapis.com/youtube/v3/search?'

        chunks = self.get_chunks_list(self.count_per_channel, YOUTUBE_VIDEOS_CHUNK)

        first_url = base_search_url+'key={}&channelId={}&type=video&part=snippet,id&order=date&maxResults={}'.format(self.api_key, channel_id, self.count_per_channel)
        url = first_url

        for chunk in chunks:
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

                try:
                    next_page_token = resp['nextPageToken']
                    url = first_url + '&pageToken={}'.format(next_page_token)
                except:
                    break
            except requests.JSONDecodeError as e:
                raise Exception(f"Resonse output is not json: {e}")
    
    def get_videos_by_ids(self, video_ids):
        base_video_url = 'https://www.youtube.com/watch?v='
        base_videos_url = 'https://www.googleapis.com/youtube/v3/videos?'
        url = base_videos_url+'key={}&id={}&part=snippet,id,contentDetails,statistics'.format(self.api_key, ','.join(video_ids))
        inp = requests.get(url)
        if not inp.status_code == 200:
            raise Exception(f"Status code is {inp.status_code}")
            
        try:
            resp = inp.json()

            for video in resp['items']:
                video_id = video['id']
                dur = isodate.parse_duration(video['contentDetails']['duration'])
                video['contentDetails']['durationSeconds'] = dur.total_seconds()
                video['videoLink'] = base_video_url + video_id
                self.videos[video_id] = video
                self.download_transcript(video_id)
        except requests.JSONDecodeError as e:
            raise Exception(f"Resonse output is not json: {e}")

    def download_transcript(self, video_id):
        try:
            video_list = YouTubeTranscriptApi.get_transcript(video_id, languages=self.languages)
            self.videos[video_id]['transcript'] = video_list
        except TranscriptsDisabled as e:
            self.videos[video_id]['transcript'] = []
    
    def save(self, file_name):
        with open(file_name, 'w') as output:
            json.dump(self.videos, output, indent=4, sort_keys=True)