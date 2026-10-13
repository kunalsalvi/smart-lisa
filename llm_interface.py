import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def getOpenAIAPIClient():
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=OPENAI_API_KEY)
    return client
