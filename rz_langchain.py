# from langchain.document_loaders import WebBaseLoader
from langchain_community.document_loaders import WebBaseLoader
from langsmith import wrappers
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import hub
from langchain.chains.retrieval_qa.base import RetrievalQA

import urllib3
urllib3.disable_warnings()
QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")

myurl = 'https://www.zimmelman.org'

def main():
    # loader = WebBaseLoader("https://www.gutenberg.org/files/1727/1727-h/1727-h.htm")
    # loader = WebBaseLoader("https://www.zimmelman.org")
    loader = WebBaseLoader("http://www.galileoco.com")
    data = loader.load()
    print(dir(wrappers))
    # client = wrappers.wrap_openai(Ollama)
    llm = Ollama(model="mistral:latest",
                    callback_manager= CallbackManager([StreamingStdOutCallbackHandler()]),
                    temperature=0.9)
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    all_splits = text_splitter.split_documents(data)


    oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents=all_splits[:1000], embedding=oembed)

    # question="Who is Neleus and who is in Neleus' family?"
    question="Who is Rob Zimmelman?"
    docs = vectorstore.similarity_search(question)
    len(docs)
    qachain=RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
    res = qachain.invoke({"query": question})
    print(res['result'])
    # print(llm.invoke("why is the sky blue"))
    return
    
    # ollama = Ollama(
    #     base_url='http://localhost:11434',
    #     model="llama3"
    # )
    # prompt = PromptTemplate(input_variables=["topic"],
    #                         template="Give me 2 interesting facts about {topic}?")
    # chain = LLMChain(llm=llm,
    #                 prompt=prompt,
    #                 verbose=False)

    # chain.invoke("New York City")


    # myurl = "https://www.cnn.com/"
    # myurl = "https://www.msn.com/"
    # myurl = "https://www.zimmelman.org/"
    # myurl = "https://buck.co/"
    # myurl = "https://news.google.com/"
    # myurl = "http://core-tools.buck.local:7000/adobe_groups"
    # myurl = "http://core-tools.buck.local:3000/activeselfservelicenses"
    # myurl = "http://core-tools.buck.local:7000/zendesk_query?querystring=type%3Aticket%20status%3Aopen" # takes too long, messy data?
    # myurl = "http://core-tools.buck.local:7000/buck_rlm_license_info"
    # myurl = "http://core-tools.buck.local:7000/leolicense"

    # get data from the web
    myloader = WebBaseLoader(myurl)
    mydata = myloader.load() 
    my_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    all_splits = my_text_splitter.split_documents(mydata)

    print('getting vector store')
    vector_store = Chroma.from_documents(collection_name='test', documents=all_splits,
                                         embedding=OllamaEmbeddings(show_progress=True))

    print('got vector store')


    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=vector_store.as_retriever(),
                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    qlist = [f"Please summarize {myurl}.",
              f"How many products are listed on {myurl}?",
               "How many licenses are in use?",
                "What is the most popular product?" ]
    

    for item in qlist:
        prompt = {"query": item}
        # question = f"Who are the top users on {myurl}?.  Provide answers as a list."
        # question = f"What are the top groupnames on {myurl}?.  Provide answers as a list." # for adobe groups
        # question = f"Produce a summary of the description fields on {myurl}?." # for zendesk, clean up this data
        # question = 
        # question = f"What are the top headlines on {myurl}?.  Provide answers as a list."
        # summary = f"What are the highlights of {myurl}?"
        # summary = f"How many tickets are listed on {myurl}?"  # zendesk
        # summary = f"How many products are listed on {myurl}?" # rlm

        # print("Latest Headlines")
        # qa_chain({"query": question})
        # print("Summary")
        print('\n\n')
        qa_chain(prompt)


if __name__ == '__main__':
    main()


