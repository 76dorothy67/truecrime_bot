import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

logging.basicConfig(
    filename="bot.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def text_to_speech(text: str, folder: str = "data/audio") -> str:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        logging.error("OPENAI_API_KEY not found in .env file")
        raise ValueError("Missing API key")

    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(folder, f"crime_story_{timestamp}.mp3")

    try:
        client = OpenAI(api_key=api_key)
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text,
        ) as response:
            response.stream_to_file(file_path)

        logging.info(f"Audio saved to {file_path}")
        return file_path

    except Exception as e:
        logging.error(f"TTS generation failed: {e}", exc_info=True)
        raise
