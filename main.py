import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from tts import text_to_speech

logging.basicConfig(
    filename="bot.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def load_prompt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def generate_story(prompt_text: str, api_key: str) -> str:
    prompt = PromptTemplate.from_template(prompt_text)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=api_key)
    story = llm.invoke(prompt.format())
    return story.content


def save_story(story: str, folder: str = "data/stories") -> str:
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(folder, f"crime_story_{timestamp}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(story)
    return file_path


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OPENAI_API_KEY not found in .env file")
        raise ValueError("Missing API key")

    try:
        prompt_text = load_prompt("prompts/crime_prompt.txt")
        story = generate_story(prompt_text, api_key)
        file_path = save_story(story)
        logging.info(f"Story generated and saved to {file_path}")
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
    try:
        audio_path = text_to_speech(story)
        logging.info(f"Generated audio file: {audio_path}")
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
