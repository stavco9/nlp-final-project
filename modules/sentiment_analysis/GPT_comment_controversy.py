# get a lst of comments and ask GPT4o to say how controversial they are in a scale from 0 to 10

import os
from openai import OpenAI
import re


def init_openai():
    client
    organization = os.getenv("OPENAI_ORGANIZATION")
    PROJECT_ID = os.getenv("OPENAI_ADIR_STAV_PROJECT")
    project_auth_token = os.getenv("OPENAI_PROJECT_AUTH")

    print(organization)
    print(PROJECT_ID)
    print(project_auth_token)

    client = OpenAI(
        organization=organization,
        project=PROJECT_ID,
        api_key=project_auth_token
    )
    return client

def get_disagreement_rating(client, comments):
    # Ensure the comments list is not empty and does not exceed 200 comments
    if not comments:
        return "No comments provided."
    if len(comments) > 200:
        return "Exceeded the limit of 200 comments."
 # Prepare the messages to send to the OpenAI API
    messages = [
        {"role": "system", "content": "You are an assistant that evaluates the level of disagreement between comments. The result shoud be a rating from 0 to 10 and nothing more."},
        {"role": "user", "content": (
            "I have a list of comments from a YouTube video. Please rate the level of disagreement "
            "between these comments on a scale of 0 to 10, where 0 means no disagreement and 10 means "
            "extreme disagreement. Here are the comments:\n\n" + "\n".join([f"Comment {i+1}: {comment}" for i, comment in enumerate(comments)])
        )}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=50,
            temperature=0.5,
        )
        rating = response.choices[0].message.content
        rating = int(re.search(r"\d+", rating).group())
        return rating

    except Exception as e:
        return f"An error occurred: {e}"

'''
# Example usage
comments = [
    "I love this video, it's so informative!",
    "This video is misleading and incorrect.",
    "Great content, keep it up!",
    "I disagree with the points made in this video.",
    "This is the best video I've seen on this topic.",
    # Add more comments as needed
]

rating = get_disagreement_rating(comments)
print(rating)
'''

