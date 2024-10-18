import os
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chat_models.base import BaseChatModel
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Generator, Any, Dict
from config import AVAILABLE_VOICES

# Load environment variables
load_dotenv()
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')

# Function to select character voice and load corresponding prompt
def select_character_voice():
    print("Available voices:")
    for i, voice in enumerate(AVAILABLE_VOICES, 1):
        print(f"{i}. {voice}")
    choice = int(input("Select a character by number: ")) - 1
    if choice < 0 or choice >= len(AVAILABLE_VOICES):
        print("Invalid choice. Defaulting to George Carlin.")
        choice = 0
    voice_name = AVAILABLE_VOICES[choice]

    # Load character prompt from local storage using an absolute path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_file = os.path.join(current_dir, "prompts", f"{voice_name.replace(' ', '_')}.txt")
    
    if os.path.exists(prompt_file):
        with open(prompt_file, 'r') as file:
            character_prompt = file.read()
    else:
        print(f"Prompt file not found: {prompt_file}")
        character_prompt = ""

    return voice_name, character_prompt

# Function to get optional custom guidance from user
def get_custom_guidance():
    instructions_prompt = input("Enter any custom guidance (or press Enter to skip): ")
    return instructions_prompt

# Function to get ElevenLabs voice ID
def get_voice_id(voice_name):
    url = "https://api.elevenlabs.io/v1/voices"
    headers = {
        "Accept": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    data = response.json()

    speaker_voice_mapping = {
        "George_Carlin": "George_Carlin",
        "Max_Payne": "Max_Payne",
        "Obi_Wan_Kenobi": "Obi_Wan_Kenobi",
        "David_Goggins": "David_Goggins",
        "Duke_Nukem": "Duke_Nukem",
        "Scary_Terry": "Scary_Terry",
        "Sterling_Archer": "Sterling_Archer",
        "Rick_Sanchez": "Rick_Sanchez"
    }

    voice_name = speaker_voice_mapping.get(voice_name)

    for voice in data['voices']:
        if voice['name'] == voice_name:
            return voice['voice_id']

    return None

# Langchain agent
LLM = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.75)

# Select character and get custom guidance
if __name__ == "__main__":
    voice_name, character_prompt = select_character_voice()
    custom_guidance = get_custom_guidance()

    # Retrieve the corresponding voice_id
    voice_id = get_voice_id(voice_name)
    if voice_id:
        print(f"Voice ID for {voice_name}: {voice_id}")
    else:
        print(f"No matching voice found for {voice_name}")

    # Combine prompts to create the system prompt
    SYSTEM_PROMPT = f"""
    Do not include anything in parentheses in your response.
    {character_prompt}

    {custom_guidance}
    """
else:
    SYSTEM_PROMPT = """
    You are a personal assistant that is helpful.
    You are part of a realtime voice to voice interaction with the human.
    Make your responses sound natural, like a human. Respond with fill words like 'hmm', 'ohh', and similar wherever relevant to make your responses sound natural.
    """

CHAT_MEMORY = ConversationBufferMemory(
    memory_key='chat_history', return_messages=True
)


class Agent:
    def __init__(
        self,
        llm: BaseChatModel = LLM,
        system_prompt: str = SYSTEM_PROMPT,
        chat_memory=CHAT_MEMORY
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.chat_memory = chat_memory
        self.memory_key = chat_memory.memory_key
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name=self.memory_key, optional=True),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        self.agent = self.prompt | self.llm | StrOutputParser()

    def _return_response(self, llm_input: Dict) -> Generator[str, Any, None]:
        response = self.agent.invoke(llm_input)
        self.chat_memory.save_context({'input': llm_input['input']}, {'output': response})
        return response
    
    def _stream_response(self, llm_input: Dict) -> Generator[str, Any, None]:
        stream = self.agent.stream(llm_input)
        response = ""
        for chunk in stream:
            response += chunk
            yield chunk

        self.chat_memory.save_context({'input': llm_input['input']}, {'output': response})
    
    def chat(self, query: str, streaming: bool=False):
        llm_input = {
            'input': query,
            'chat_history': self.chat_memory.load_memory_variables({})[self.memory_key]
        }

        if streaming:
            return self._stream_response(llm_input)
        else:
            return self._return_response(llm_input)

if __name__ == "__main__":
    agent = Agent(system_prompt=SYSTEM_PROMPT)

    while True:
        query = input("Chat: ")
        print("Response:\n")
        for token in agent.chat(query, streaming=True):
            print(token, end='', flush=True)
        print("\n")