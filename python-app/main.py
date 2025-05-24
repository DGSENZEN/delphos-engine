import getpass
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai
os.environ["GOOGLE_API_KEY"]

embed_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

data = None

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API key: ")
    
# Initializing the models.
    
gemini_text_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=2048,
    timeout=None,
    max_retries=2
    )