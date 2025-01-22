# from langchain_community.llms import Ollama
from langchain_ollama.llms import OllamaLLM as Ollama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from sys import argv

# 1. Create the model
llm = Ollama(model='llama3')
embeddings = OllamaEmbeddings(model='znbang/bge:small-en-v1.5-f32')

# 2. Load the PDF file and create a retriever to be used for providing context
loader = PyPDFLoader(argv[1])
pages = loader.load_and_split()
doc_index = pages.index
try:
    # store = DocArrayInMemorySearch.from_documents(documents=pages, embedding=embeddings)
    store = DocArrayInMemorySearch(doc_index=doc_index, embedding=embeddings)
except Exception as _e:
    print(f'exception {_e} building store')
retriever = store.as_retriever()

# 3. Create the prompt template
template = """
Answer the question based only on the context provided.

Context: {context}

Question: {question}
"""

prompt = PromptTemplate.from_template(template)

def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)

# 4. Build the chain of operations
chain = (
  {
    'context': retriever | format_docs,
    'question': RunnablePassthrough(),
  }
  | prompt
  | llm
  | StrOutputParser()
)

# 5. Start asking questions and getting answers in a loop
while True:
  question = input('What do you want to learn from the document?\n')
  print()
  res = chain.invoke({'question': question})
  print(res)
  print()   