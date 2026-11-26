
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv


load_dotenv()

# Defining OpenAI LLM Model
llm_open_ai = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.7,
)

