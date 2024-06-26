import asyncio
from dotenv import load_dotenv
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_openai import ChatOpenAI
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.pydantic_v1 import BaseModel, Field
from enum import Enum
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)

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


model = ChatOpenAI(temperature=0, model="gpt-4o")
structured_llm = model.with_structured_output(List_of_Changes)


def scrape(urls):
    print(f"URLS: {urls}")
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

    return splits[0].page_content


def generate_patch_url(chainInput):
    patch_versions = chainInput["patch_versions"]
    urls = []
    for major, minor in patch_versions:
        urls.append(
            f"https://www.leagueoflegends.com/en-us/news/game-updates/"
            f"patch-{major}-{minor}-notes/"
        )
    return urls


urlGeneratorRunnable = RunnableLambda(generate_patch_url)
scrapeRunnable = RunnableLambda(scrape)
retriever = RunnableParallel(
    {
        "context": urlGeneratorRunnable | scrapeRunnable | structured_llm,
        "question": RunnablePassthrough(),
    }
)

prompt_str = """Answer the question below using the context:
Context: {context}

Question: {question}

Answer: """
prompt = ChatPromptTemplate.from_template(prompt_str)

chain = retriever | prompt | model | StrOutputParser()


async def run_chain():
    result = await chain.ainvoke(
        {
            "patch_versions": [[14, 8], [14, 9], [14, 10]],
            "question": "What are the recent changes related to Skarner?",
        }
    )
    print(result)


asyncio.run(run_chain())
