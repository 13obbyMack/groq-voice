import os
from dotenv import load_dotenv
import pyaudio
from enum import Enum

load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not ELEVENLABS_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY environment variable not set")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")

# Available character voices
AVAILABLE_VOICES = [
    "George_Carlin",
    "Max_Payne",
    "Obi_Wan_Kenobi",
    "David_Goggins",
    "Duke_Nukem",
    "Scary_Terry",
    "Sterling_Archer",
    "Rick_Sanchez"
]

# class Voices(Enum):
# 
#     George_Carlin = "WUskcuzAbpFeh46oYy0B"
#     Max_Payne = "vosASqmKV6UsA6rHMkqP"
#     Obi_Wan_Kenobi = "Yj8J8mqj4zaLvpq7MLEt"
#     David_Goggins = "Z3lN8xrIGAbOXleAdIeQ"
#     Duke_Nukem = "Q3V0aigkBauXSZTYJsmr"
#     Scary_Terry = "yAZetOJ1I6kTvOehOOGp"
#     Sterling_Archer = "J2tWojuB5wEBXrVezwXm"
#     Rick_Sanchez = "q7gnMYP8uD2QRVYhvQzt"

VOICE_ID = Voices.George_Carlin
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
SILENCE_THRESHOLD = 200  # Adjust this threshold as needed
SILENCE_DURATION = 2  # Duration of silence to stop recording in seconds
PRE_SPEECH_BUFFER_DURATION = 0.5  # 500ms of audio to keep before speech detection