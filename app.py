from flask import Flask, render_template, request
from rz_chat import get_rag_chain_for_pdf, ask_question, get_rag_chain_for_json, get_retriever_with_history, get_vector_db_retriever_for_json, \
    openai_llm, get_vector_db_retriever_for_pdf

my_json_retriever_with_history = get_retriever_with_history(retriever=get_vector_db_retriever_for_json(), llm=openai_llm)
my_pdf_retriever_with_history = get_retriever_with_history(retriever=get_vector_db_retriever_for_pdf(), llm=openai_llm)

pdf_chain = get_rag_chain_for_pdf(retriever_with_history=my_pdf_retriever_with_history)
json_chain = get_rag_chain_for_json(retriever_with_history=my_json_retriever_with_history)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/linux_python_chat', methods=['GET', 'POST'])
def linux_python():
    message = request.form['msg']
    return ask_question(rag_chain=pdf_chain, question=message)


@app.route('/chat', methods=['GET', 'POST'])
def chat():
    message = request.form['msg']
    return ask_question(rag_chain=json_chain, question=message)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)