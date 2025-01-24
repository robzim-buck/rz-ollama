import json
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
# from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter, RecursiveJsonSplitter
from langchain_openai import ChatOpenAI
from pathlib import Path

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from mistralai import Mistral
# from mistralai.client import mistral_client

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, JSONLoader
from pprint import pprint as pp
from dotenv import dotenv_values
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from argparse import ArgumentParser

# _ = load_dotenv(find_dotenv()) # read local .env file
DOTENV = dotenv_values('.key')

OPENAI_API_KEY=DOTENV['OPENAI_API_KEY']
chat_history = []
myurl = 'https://www.zimmelman.org'


# OPENAI_MODEL = 'gpt-3.5-turbo-0125'
OPENAI_MODEL = 'gpt-4o'
# exit(0)
#initialize the LLM we'll use
openai_llm_chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=OPENAI_MODEL)

# initi the mistral model
mistral_ai_chat = ChatMistralAI(model="mistral-large-latest", api_key=DOTENV['MISTRAL_KEY'])

my_mistral_client = Mistral(api_key=DOTENV['MISTRAL_KEY'])

# OPENAI_MODEL = 'gpt-3.5-turbo-0125'
# OPENAI_MODEL = 'gpt-4o-mini'
OPENAI_MODEL = 'gpt-4-turbo'


CHAT_MODEL = openai_llm_chat


def zendesk_metadata_func(record: dict, metadata: dict) -> dict:
    metadata = {'assignee': 'None', 'requester': 'None', 'submitter': 'None'}
    try:
        assignee = record.get("assignee")['email'] if record and 'assignee' in record.keys() else 'None'
        metadata["assignee"] = assignee
    except Exception as e:
        print(f'exception {e} getting assignee for record')
    try:
        requester = record.get("requester")['email'] if record and 'requester' in record.keys() else 'None'
        metadata["requester"] = requester
    except Exception as e:
        print(f'exception {e} getting requester for record')
    try:
        submitter = record.get("submitter")['email'] if record and 'submitter' in record.keys() else 'None'
        metadata["submitter"] = submitter
    except Exception as e:
        print(f'exception {e} getting submitter for record')
        # pp(record)
    metadata['source'] = record['plain_body'] + " " + assignee + " " + requester + " " +submitter # replace source with good stuff
    return metadata


def new_load_json_with_metadata(file_path: str):
    loader = JSONLoader(
    file_path=file_path,
    jq_schema='.comments[]',
    is_content_key_jq_parsable=False,
    content_key='body',
    text_content=True,
    metadata_func=zendesk_metadata_func,
    json_lines=True)
    data = loader.load()
    return data



def old_load_json(file_path: str):
    summaries = ""
    with open('./zendesk_tix_reallyfixed.json') as ifile:
        data = ifile.read().split('\n')
        return data
        for d in data:
            if d:
                try:
                    js = json.loads(d)
                except Exception as e:
                    print(f'exception {e} getting json')
                    continue
                if js:
                    try:
                        submitter_name = js['submitter']['name']
                        submitter_email = js['submitter']['email']
                    except Exception as _e:
                        submitter_name = submitter_email  = ' '
                    try:
                        requester_name = js['requester']['name']
                        requester_email = js['requester']['email']
                    except Exception as _e:
                        requester_name = requester_email = ' '
                    try:
                        assignee_name = js['assignee']['name']
                        assignee_email = js['assignee']['email']
                    except Exception as _e:
                        assignee_name = assignee_email = ' '
                    priority = js['priority']
                    tagchain = " tags: ".join([_ for _ in js['tags']])
                    commentchain = " comments: ".join([_['body'] for _ in js['comments']])
                    summaries += commentchain.replace('\n',' ')
                    summaries += " submitter name: " + submitter_name + \
                        ' submitter email: ' + submitter_email if submitter_email else ' ' + \
                            ' requester: ' + \
                            requester_name + ' requester email: ' + \
                    requester_email if requester_email else ' ' +  \
                        ' assignee: ' + assignee_name + \
                            ' assignee email: ' + \
                        assignee_email if assignee_email else ' '  + ' ' + tagchain
                    summaries += ' priority: ' + priority if priority else ' ' + ' ' +  \
                        ' subject: ' + js['raw_subject'] + \
                            ' description: ' + js['description']
        return summaries


def zendesk_metadata(record: dict, metadata: dict) -> dict:
    metadata["page_content"] = record
    return metadata


def split_documents(documents):
    doc_string = json.dumps(documents)
    text_splitter = RecursiveCharacterTextSplitter(separators=documents,
                                                   keep_separator=False,
                                                   is_separator_regex=False)
    split_json = text_splitter.split_text(text=doc_string)
    return split_json



def new_split_json_list(documents):
    text_splitter = RecursiveJsonSplitter(max_chunk_size=5000)
    split_json = text_splitter.split_text(text=documents)
    return split_json



def split_json_list(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,
                                 chunk_overlap=2000,
                                 keep_separator=False,
                                 is_separator_regex=False,
                                 strip_whitespace=True)
    split_json = text_splitter.split_text(text=documents)
    return split_json


def split_character_text(documents):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=100,
        chunk_overlap=8,
        length_function=len
    )
    split_pages = text_splitter.split_text(documents)
    return split_pages


def split_text(documents):
    text_splitter = TextSplitter(chunk_size=800,
                                 chunk_overlap=80,
                                 keep_separator=False,
                                 is_separator_regex=False,
                                 strip_whitespace=True)
    split_text = text_splitter.split_text(text=documents)
    return split_text



def load_pdf_documents():

    # Load PDF
    loaders = [
        PyPDFLoader("/Users/robzimmelman/Documents/Advanced Bash Shell Scripting Guide.pdf"),
        PyPDFLoader("/Users/robzimmelman/Documents/automatetheboringstuffwithpython_new.pdf"),
        PyPDFLoader("/Users/robzimmelman/Documents/linuxbasicsforhackers.pdf"),
        PyPDFLoader("/Users/robzimmelman/Documents/thelinuxcommandline.pdf"),
        PyPDFLoader("/Users/robzimmelman/Documents/thinkpython.pdf")
    ]
    docs = []

    for loader in loaders:
        docs.extend(loader.load())
    return docs

def print_output(docs):
    for doc in docs:
        print('The output is: {}. \n\nThe metadata is {} \n\n'.format(doc.page_content, doc.metadata))


def create_vector_db_for_pdf(documents):
    vectordb = FAISS.from_documents(documents, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small"))
    res = vectordb.save_local("pdf_faiss2_index")
    return res



def create_mistral_vector_db_for_json(documents):
    my_mistral_embeddings = my_mistral_client.embeddings.create(model="mistral-embed",
                                                                inputs=documents)
    vectordb = FAISS.from_texts(documents, my_mistral_embeddings)
    res = vectordb.save_local("zendesk_faiss2_index")
    return res


def create_vector_db_for_json(documents):
    vectordb = FAISS.from_texts(documents, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small"))
    res = vectordb.save_local(f"{OPENAI_MODEL}zendesk_faiss2_index")
    return res


def new_create_vector_db_for_json(documents):
    vectordb = FAISS.from_documents(documents, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small"))
    # vectordb = FAISS.from_texts(documents, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small"))
    res = vectordb.save_local(f"{OPENAI_MODEL}_new_zendesk_faiss2_index")
    return res



def get_mistral_vector_db_retriever_for_json():
    embeddings_model = MistralAIEmbeddings()
    new_db = FAISS.load_local("zendesk_faiss2_index", embeddings_model, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    return retriever


def new_get_vector_db_retriever_for_json():
    embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    new_db = FAISS.load_local(f"{OPENAI_MODEL}_new_zendesk_faiss2_index", embeddings_model, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever(search_type="similarity", search_kwargs={"k": 50})
    return retriever

def get_vector_db_retriever_for_json():
    embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    new_db = FAISS.load_local(f"{OPENAI_MODEL}zendesk_faiss2_index", embeddings_model, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    return retriever


def get_vector_db_retriever_for_pdf():
    embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    new_db = FAISS.load_local("pdf_faiss2_index", embeddings_model, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    return retriever



def get_retriever_with_history(retriever, llm):
    system_prompt = """Given the chat history and a recent user question \
    generate a new standalone question \
    that can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed or otherwise return it as is. \
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    retriever_with_history = create_history_aware_retriever(
        llm, retriever, prompt
    )
    return retriever_with_history


def get_rag_chain(retriever_with_history):
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\



    
    If the user asks what you know about, tell them you know about the following documents
        Advanced Bash Shell Scripting Guide.pdf,
        automatetheboringstuffwithpython_new.pdf,
        linuxbasicsforhackers.pdf,
        thelinuxcommandline.pdf,
        thinkpython.pdf.




    {context}"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(mistral_ai_chat, qa_prompt)
    rag_chain = create_retrieval_chain(retriever_with_history, question_answer_chain)
    return rag_chain


def get_rag_chain_for_json(retriever_with_history):
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    The user is a higly-skilled systems administrator and I.T. Specialist. \
    Use the following pieces of retrieved context to answer the question. \
    Use a lengthy technical explaination if it is appropriate. \
    If the user asks you for a list, always include all items that match in the list. \
    If the user asks for a list of items, include 100 items when possible. \
    If the user asks what you know about, tell them you know about Zendesk Tickets. \
    If the user asks how many tickets there are, tell them there are 6973 Zendesk Tickets. \
    If the user asks how many urgent tickets there are, tell them there are 48 Urgent Tickets. \
    Think step-by-step.  If the user asks for a list, always include all items that match in the list. \
    If the user asks for a mathematical result other than the number of zendesk tickets or urgent tickets, \
    create a variable to count the items, then step by step, report the result. Show your work. \
    If don't know the answer and can't figure out the answer step-by-step, just say 'I'm sorry Dave, I'm afraid I can't do that' or 'I don't know'. \

    {context}"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(CHAT_MODEL, qa_prompt)
    rag_chain = create_retrieval_chain(retriever_with_history, question_answer_chain)
    return rag_chain




def ask_question(rag_chain, question=None):
    # question = "What are list comprehensions?"
    ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
    chat_history.extend([HumanMessage(content=question), ai_msg_1["answer"]])
    print("\n\n", ai_msg_1["answer"])
    return ai_msg_1["answer"]




# def do_pdf_conversation():
#     # docs = load_pdf_documents()
#     # split_docs = split_character_text(documents=docs)
#     # res = create_vector_db_for_pdf(split_docs)
#     # pp(res)
#     my_retriever_with_history = get_retriever_with_history(retriever=get_vector_db_retriever_for_pdf(), llm=openai_llm_chat)
#     my_rag_chain = get_rag_chain(retriever_with_history=my_retriever_with_history)
#     myq = ""
#     while myq != "bye":
#         myq = input("\n\nAsk me something about linux, bash or python  ....  ")
#         ask_question(rag_chain=my_rag_chain, question=myq)
#         # ask_question(rag_chain=my_rag_chain, question="are they cool?")
    

def create_json_vector_db():
    docs = old_load_json('./zendesk_tix_test.json')
    split_docs = split_json_list(documents=docs)
    create_vector_db_for_json(split_docs)





# def do_json_conversation_with_metadata(urgent_count:int=0, tix_count:int=0):
def do_json_conversation_with_metadata():
    # docs = load_json('./zendesk_tix_reallyfixed.json')
    # split_docs = split_json_list(documents=docs)
    # create_vector_db_for_json(split_docs)
    my_retriever_with_history = get_retriever_with_history(retriever=new_get_vector_db_retriever_for_json(), llm=openai_llm_chat)
    my_rag_chain = get_rag_chain_for_json(retriever_with_history=my_retriever_with_history)
    myq = ''
    while myq != 'bye':
        myq = input("\n\nAsk me something about Zendesk Tickets ...  ")
        ask_question(rag_chain=my_rag_chain, question=myq)
        # ask_question(rag_chain=my_rag_chain, question="are they cool?")



def fix_json_file():
    with open('./zendesk_tix.json') as ifile:
        data = ifile.read().split('\n')
        with open('./zendesk_tix_fixed.json', 'w') as ofile:
            for d in data:
                ofile.write(f"{d},\n")
    


def unfix_json_file():
    with open('./zendesk_tix_fixed.json') as ifile:
        data = ifile.read().split('\n')
        with open('./zendesk_tix_reallyfixed.json', 'w') as ofile:
            for d in data:
                ofile.write(f"{d.rstrip(",")}\n")
    

def main():
    # d = new_load_json_with_metadata('zendesk_tix_new.json')
    # #add ticket number to page content
    # for i, _ in enumerate(d):
    #     _.page_content = f"Ticket Number {i+1} : " +  _.metadata['source'] + " "  + _.page_content
    # new_create_vector_db_for_json(documents=d)

    # info = old_load_json('zendesk_tix_reallyfixed.json')
    # info = [json.loads(_) for _ in info if _]
    # urgent = len([_ for _ in info if _['priority'] == 'urgent'])
    # print(f"{urgent=}")
    do_json_conversation_with_metadata()



if __name__ == '__main__':
    main()