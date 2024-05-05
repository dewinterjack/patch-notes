import dotenv
dotenv.load_dotenv()
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_openai import ChatOpenAI
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.pydantic_v1 import BaseModel, Field
import pprint

model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

class Article(BaseModel):
    title: str = Field(description="The title of the article")
    summary: str = Field(description="A brief summary of the article")

structured_llm = model.with_structured_output(Article)

def extract(content: str):
    return structured_llm.invoke(content)

def scrape(urls):
    urls = ["https://www.espn.com", "https://lilianweng.github.io/posts/2023-06-23-agent/"]
    loader = AsyncHtmlLoader(urls)
    docs = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        docs, tags_to_extract=["span"]
    )
    print("Extracting content with LLM")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=0
    )
    splits = splitter.split_documents(docs_transformed)
    extracted_content = extract(content=splits[0].page_content)
    pprint.pprint(extracted_content)
    return extracted_content

urls = ["https://www.espn.com", "https://lilianweng.github.io/posts/2023-06-23-agent/"]

extracted_content = scrape(urls)

