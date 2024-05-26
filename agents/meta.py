from enum import Enum
from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from helpers.create_agent import create_agent
from helpers.create_tool import create_tool
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


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


def generate_patch_url(patch_versions):
    urls = []
    for major, minor in patch_versions:
        urls.append(
            f"https://www.leagueoflegends.com/en-us/news/game-updates/"
            f"patch-{major}-{minor}-notes/"
        )
    return urls


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


def structure_content(raw_data, model):
    structured_model = model.with_structured_output(List_of_Changes)
    structured_data = structured_model.invoke(raw_data)
    return structured_data


system_prompt = "You are an expert in analyzing League of Legends meta changes based on patch notes."
agent = create_agent("MetaAnalysisAgent", [], system_prompt)
model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")


async def analyse_meta(patch_versions, question):
    print("Starting analyse_meta function")
    urls = generate_patch_url(patch_versions)
    print(f"Generated URLs: {urls}")

    raw_data = scrape(urls)
    print(
        f"Scraped raw data: {raw_data[:500]}..."
    )  # Print the first 500 characters for brevity

    structured_data = structure_content(raw_data, model)
    print(f"Structured data: {structured_data}")

    relevant_changes = [
        change for change in structured_data.changes if "Skarner" in change.subject.name
    ]
    print(f"Relevant changes: {relevant_changes}")

    response_content = "\n".join(
        [f"{change.title}: {change.summary}" for change in relevant_changes]
    )
    print(f"Response content: {response_content}")

    result = await agent.invoke(
        {
            "messages": [
                {"role": "system", "content": f"Context: {response_content}"},
                {"role": "user", "content": question},
            ]
        }
    )
    print(f"Agent result: {result}")
    return result
