import os
from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import gradio as gr
import pandas as pd
import sys
from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for)


app = Flask(__name__)
os.environ["OPENAI_API_KEY"] = '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'

@app.route('/')
#def construct_index(directory_path):
#    max_input_size = 4096
#    num_outputs = 512
#    max_chunk_overlap = 50
#    chunk_size_limit = 600

#    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

#    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

#    documents = SimpleDirectoryReader(directory_path).load_data()

#    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

#    index.save_to_disk('index.json')

#    return index

#df['ada_embedding'] = df.ada_embedding.apply(eval).apply(np.array)

def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    return response.response




#Not necessary in live code (gradio as gr)
iface = gr.Interface(fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="Custom-trained AI Chatbot")

#index = construct_index("docs")
iface.launch(share=True)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/hello', methods=['POST'])
def hello():
   name = request.form.get('name')

   if name:
       print('Request for hello page received with name=%s' % name)
       return render_template('hello.html', name = name)
   else:
       print('Request for hello page received with no name or blank name -- redirecting')
       return redirect(url_for('index'))

if __name__ == '__main__':
   app.run()
