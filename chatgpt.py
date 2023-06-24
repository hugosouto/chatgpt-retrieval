import os 
import sys

from langchain.document_loaders import TextLoader 
from langchain.document_loaders import DirectoryLoader 
from langchain.indexes import VectorstoreIndexCreator 
from langchain.llms import OpenAI 
from langchain.chat_models import ChatOpenAI 

# Read API keyfrom the environment variable OPENAI_API_KEY
APIKEY = os.getenv("OPENAI_API_KEY")

# query = sys.argv[1]
query = "Em quais minitérios foi dividido o Ministério da Economia no governo Lula?"

loader = TextLoader('data.txt') 
# loader = DirectoryLoader(".", glob="*.txt")  
index = VectorstoreIndexCreator().from_loaders([loader])

print(index.query(query, llm=ChatOpenAI(model="gpt-4")))