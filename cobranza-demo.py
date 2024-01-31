# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Second
import streamlit as st

import vertexai
from google.cloud import documentai
from google.api_core.client_options import ClientOptions

from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.llms import VertexAI
#from langchain.chains.question_answering import load_qa_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import LLMChain

from vertexai.preview.generative_models import GenerativeModel, Part

# Constantes
PROJECT_ID = "cecl-genai-demos"
LOCATION = "us-central1"
MODEL_NAME = "gemini-pro"
    
# Cargo el documento con Document AI
def read_document(file):
    # Constantes para inicializar el doc processor de DocAI
    DOCAI_LOCATION = "us"
    PROCESSOR_ID = "a3c30cf2a12207e9"
    PROCESSOR_VERSION = 'rc'

    # Inicializo el cliente de DocAI
    opts = ClientOptions(api_endpoint=f"{DOCAI_LOCATION}-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    # Inicializo el nombre del processor
    processor_name = client.processor_version_path(
        PROJECT_ID, DOCAI_LOCATION, PROCESSOR_ID, PROCESSOR_VERSION
    )

    # Leo el archivo en memoria
    document_content = file.getvalue()

    # Cargo el documento en el formato raw para enviar a la API
    raw_document = documentai.RawDocument(
        content=document_content, mime_type="application/pdf")

    # Proceso el documento
    request = documentai.ProcessRequest(name=processor_name, raw_document=raw_document)
    result = client.process_document(request=request)
    document = result.document
    return document.text


# Inicializo el cliente de Vertex
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Inicializo el modelo de Vertex
model = GenerativeModel("gemini-pro")

# Defino el template del prompt
prompt = PromptTemplate.from_template("""
Question: {input}

{call_transcription}

Answer:""")

# Instancio la chain de QA de Langchain
#chain = create_stuff_documents_chain(
#    llm=VertexAI(model_name=MODEL_NAME, temperature=0, max_output_tokens=1024),
#    prompt=prompt)

# Instancio la chain de Langchain
llm = VertexAI(model_name=MODEL_NAME, temperature=0, max_output_tokens=1024)
chain = LLMChain(
    llm=llm,
    prompt=prompt
)

# Dibujo la ventana
st.title("📝 Demo Cobranza Gemini")
uploaded_file = st.file_uploader("Suba un documento", type=("pdf", "md"))
#question = st.text_input(
#    "Indique un prompt para analizar el llamado",
#    placeholder="Clasifique la intención de pago del cliente en Muy Baja, Baja, Media, Alta o Muy Alta",
#    disabled=not uploaded_file,
#)
question = st.text_area(
    "Indique un prompt para analizar el llamado",
    height=3,
    placeholder="Clasifique la intención de pago del cliente en Muy Baja, Baja, Media, Alta o Muy Alta",
    label_visibility="visible")

# Logica de la ventana
if not uploaded_file:
    st.info("Por favor cargue un archivo")

if not question:
    st.info("Por favor indique una pregunta")

if uploaded_file and question:
    text = read_document(uploaded_file)

    st.write("### Transcripción del llamado")
    st.write(text)

    doc =  Document(page_content=text, metadata={"source": "local"})
    print(doc)
    docs = [doc]

    #reply = str(chain.run(input_documents=docs, question=question))
    #reply = str(chain.invoke({
    #    "input": question,
    #    "context": docs
    #}))
    #reply = str(chain.run(input=question, call_transcription=text))
    #reply = llm(prompt.format(input=question, call_transcription=text))
    
    responses = model.generate_content(
        question + "\n\n" + text,
        generation_config = {
            "max_output_tokens": 2048,
            "temperature": 0.9,
            "top_p": 1
        },
        stream=True,
    )
    reply = ""
    for response in responses:
        reply += response.text

    print("*** RESPUESTA\n" + reply)

    st.write("### Respuesta")
    st.write(reply)