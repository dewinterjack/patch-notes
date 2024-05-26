from dotenv import load_dotenv
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.pydantic_v1 import BaseModel, Field
from enum import Enum
from typing import List
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class SubjectCategory(Enum):
    CHAMPION = "champion"
    ITEM = "item"
    RUNE = "rune"
    SPELL = "spell"
    OBJECTIVE = "objective"
    JUNGLE = "jungle"
    RANKED = "ranked"
    MISC = "misc"


class Subject(BaseModel):
    name: str = Field(description="The name of the subject")
    category: SubjectCategory = Field(
        description="Tags describing the subject", default=SubjectCategory.MISC
    )


class ChangeTags(Enum):
    BUGFIX = "bugfix"
    BUFF = "buff"
    NERF = "nerf"
    ADJUSTMENT = "adjustment"
    GAMEPLAY = "gameplay"
    OBJECTIVES = "objectives"
    JUNGLE = "jungle"
    MINIONS = "minions"
    GOLD = "gold"
    EXPERIENCE = "experience"
    RANKED = "ranked"
    QUALITY_OF_LIFE = "quality_of_life"
    BASE_STATS = "base_stats"
    SCALING = "scaling"
    TECHNICAL = "technical"
    SERVERS = "servers"
    COOLDOWN = "cooldown"
    RATIO = "ratio"
    ARENA = "arena"
    ARAM = "aram"
    TFT = "tft"
    MISC = "misc"


class Change(BaseModel):
    title: str = Field(description="The title of the change")
    summary: str = Field(description="A brief summary of the change")
    subject: Subject = Field(description="The subject receiving the change")
    tags: List[ChangeTags] = Field(description="Tags describing the change")


class List_of_Changes(BaseModel):
    changes: List[Change] = Field(description="The list of changes")


def extract(content: str):
    return structured_llm.invoke(content)


def scrape(urls):
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

    return extracted_content


def generate_patch_url(major, minor):
    return (
        f"https://www.leagueoflegends.com/en-us/news/game-updates/"
        f"patch-{major}-{minor}-notes/"
    )

model = ChatOpenAI(temperature=0, model="gpt-4o")
structured_llm = model.with_structured_output(List_of_Changes)

urls = [generate_patch_url(14, minor) for minor in range(8, 11)]

extracted_content = scrape(urls)

# documents = [Document(page_content=change.summary, metadata={"title": change.title}) for change in extracted_content.changes]
# embeddings = OpenAIEmbeddings()
# vectorstore = Chroma.from_documents(documents, embedding=embeddings)
# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# qa_chain = RetrievalQA.from_chain_type(
#     llm=model,
#     chain_type="stuff",
#     retriever=retriever
# )


# query = "What are the recent changes related to Skarner?"
# answer = qa_chain.invoke(query)
# print(answer)

prompt_str = """Answer the question below using the context:

Context: {context}

Question: {question}

Answer: """
prompt = ChatPromptTemplate.from_template(prompt_str)

chain = prompt | model | StrOutputParser()

result = chain.invoke({ "context": extracted_content, "question": "What are the recent changes related to Skarner?"})
print(result)

