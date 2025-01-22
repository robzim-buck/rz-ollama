import os
import uvicorn
from dotenv import dotenv_values
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException
from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.nasa.toolkit import NasaToolkit
from langchain_community.utilities.nasa import NasaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_openai import OpenAI
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Optional, List


from sqlmodel import Field, SQLModel, Session, create_engine, select, delete


class Hero(SQLModel, table=True):
    name: str = Field(primary_key=True)
    secret_name: str
    age: Optional[int] = None


class MyResult(SQLModel):
    result: str

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

KEYS = dotenv_values('.key')
TAVILY_KEY = KEYS['TAVILY_KEY']

MISTRAL_API_KEY = KEYS['MISTRAL_KEY']
chat_llm = ChatMistralAI(
    model="mistral-8x-22b",
    temperature=0,
    max_retries=2,
    # other params...
)

ollama_chat_llm = ChatOllama(
    model="llama3.1",
    temperature=0,
    # other params...
)
# if "MISTRAL_API_KEY" not in os.environ:
#     os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter your Mistral API key: ")

openai_llm = OpenAI(temperature=0, openai_api_key=KEYS["OPENAI_API_KEY"])



os.environ["TAVILY_API_KEY"] = TAVILY_KEY


def do_nasa_search(query:str=None):
  nasa = NasaAPIWrapper()
  toolkit = NasaToolkit.from_nasa_api_wrapper(nasa)
  agent = initialize_agent(
      toolkit.get_tools(), openai_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
  )
  res = agent.run(query)
  return res


def do_search(query:str=None):
    # yield 'searching'
    # search = tool(query)
    search = TavilySearchResults(max_results=5)
    search_results = search.invoke(query)
    # print(search_results)
    # tools = [search]
    return search_results


"""

http://localhost:11434/api/chat -d '{
  "model": "llama3",
  "messages": [
    { "role": "user", "content": "why is the sky blue?" }
  ]
}'

"""


def get_langchain_tools(max_result_count:int=10):
    tool = TavilySearchResults(max_results=max_result_count)
    # search_results = search.invoke(query)
    # print(search_results)
    # tools = [search]
    return tool


@app.get('/tavily_search')
def doit(prompt:str=None):
  res = do_search(query=prompt)
  return JSONResponse(content=res)


@app.get('/nasa')
def nasa(prompt:str=None):
  return do_nasa_search(query=prompt)


@app.get('/duckduckgo_search')
def duckduckgosearch(prompt:str=None):
  search = DuckDuckGoSearchRun()
  res = search.invoke(prompt)
  return res


@app.get('/serper_search')
def serper_search(prompt:str=None):
  search = GoogleSerperAPIWrapper(serper_api_key=KEYS["SERPER_KEY"])
  res = search.run(query=prompt)
  return res


@app.get('/ollama_search/{details}')
def ollama_search(prompt:str=None, details:bool | None = None):
    # themodel = 'arguer'
    messages = [
        (
            "system",
            "You are an angry man from Brooklyn New York.  Limit your answers to one sentence.  Disagree with the user in your answers.  Suggest that the user avoid taking action every time.  Tell the user they are wrong in your answers.  Use the phrase 'yo stupid' in your answers every time.  Use the phrase 'management believes' in your answers every time.",
        ),
        ("human", prompt),
    ]
    ai_msg =  ollama_chat_llm.invoke(messages)
    if details:
        return ai_msg
    return ai_msg.content


@app.get('/coolguy/{details}')
def chat(prompt:str=None, details:bool | None = None):
  messages = [
      (
          "system",
          "You are a cool surfer from Venice Beach, California.  Respond by starting with the phrase 'hey dude'.  Limit your response to one sentence.",
      ),
      ("human", prompt),
  ]
  ai_msg =  ollama_chat_llm.invoke(messages)
  if details:
    return ai_msg
  return ai_msg.content




@app.get('/translate_to/{language}/{details}')
def translate_to(prompt:str=None, language:str=None, details:bool | None = None):
  messages = [
      (
          "system",
          f"You are a helpful assistant that translates English to {language}. Translate the user sentence.",
      ),
      ("human", prompt),
  ]
  ai_msg =  ollama_chat_llm.invoke(messages)
  if details:
    return ai_msg
  return ai_msg.content

engine = create_engine("mysql+pymysql://robz:robz@zimserver.zimmelman.org/scratch", echo=True)


SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


@app.get("/hero/{name}")
def get_hero(name:str=None) -> Hero:
    if name:
        try:
            with Session(engine) as session:
                print(type(name), name)
                my_select = select(Hero).where(Hero.name == name)
                my_hero = session.exec(my_select).first()
                return my_hero
        except Exception as _e:
            print(f'Exception {_e} getting hero from db')


@app.delete("/hero/{name}")
def delete_hero(name:str=None):
    try:
        with Session(engine, autoflush=True) as session:
            my_ret = select(Hero).where(Hero.name == name)
            my_found_hero_name = session.exec(my_ret).first().name
            if my_found_hero_name is None:
                print("not found")
                return Hero(id=0, name='ERROR', secret_name='ERROR', age=9999)
    except Exception as _e:
        print(f'Exception {_e} HERO NOT FOUND!!! deleting hero from db')
        return Hero(id=0, name='ERROR', secret_name='ERROR', age=9999)
    else:
        try:
            hero_ret = select(Hero).where(Hero.name == name)
            hero_ret = session.exec(hero_ret)
            my_delete = delete(Hero).where(Hero.name == name)
            my_hero = session.exec(my_delete)
            print('my_hero ', my_hero)
            session.commit()
            return f"deleted {name}"
        except Exception as _e:
            print(f'Exception {_e} deleting hero from db')
            return Hero(id=0, name='ERROR', secret_name='ERROR', age=9999)


@app.post("/hero/{name}/{secretname}/{age}")
def create_hero(name:str=None, secretname:str=None, age:int=None, ) -> str:
    myhero = Hero(name=name, secret_name=secretname, age=age)
    try:
        session = Session(engine, autoflush=True)
        session.add(myhero)
    except Exception as _e:
        print(f'Exception {_e} geting session')
    try:
        res = session.commit()
        print(f"result of commit = {res}")
        return f"Created {name}"
    except Exception as _e:
        print(f'Exception {_e} adding hero to db')
        return Hero(id=0, name='ERROR', secret_name='ERROR', age=9999)



@app.put("/hero/{name}/{secretname}/{age}", response_model=Hero)
def update_hero(name:str=None, secretname:str=None, age:int=None) -> str:
    try:
        with Session(engine) as session:
            my_select = select(Hero).where(Hero.name == name)
            my_hero = session.exec(my_select).first()
            my_hero.age = age
            my_hero.secret_name = secretname
            session.add(my_hero)
            session.commit()
            session.refresh(my_hero)
            return my_hero
    except Exception as _e:
        print(f'Exception {_e} adding hero to db')
        return Hero(id=0, name='ERROR', secret_name='ERROR', birthday='0001-01-01', age=9999)


@app.get("/heroes", response_model=List[Hero])
def heroes() -> List[Hero]:
    try:
        with Session(engine) as session:
            statement = select(Hero)
            hero = session.exec(statement).all()
            print(hero)
            return hero
    except Exception as _e:
        raise HTTPException(400, 'Hero Not Found')


uvicorn.run(app=app)
