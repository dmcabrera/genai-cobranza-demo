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

# Imports
import streamlit as st
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
from google.cloud import documentai
from google.api_core.client_options import ClientOptions

# Constantes
PROJECT_ID = "cecl-genai-demos" # @TODO: Cambiar ID del proyecto
LOCATION = "us-central1"
MODEL_NAME = "gemini-pro"
    
# Cargo el documento con Document AI
def read_document(file):
    # Constantes para inicializar el doc processor de DocAI
    DOCAI_LOCATION = "us"
    PROCESSOR_ID = "a3c30cf2a12207e9" # @TODO: Cambiar ID del DocAI processor
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

# Dibujo la ventana
st.title("📝 Demo Cobranza Gemini")
uploaded_file = st.file_uploader("Suba un documento", type=("pdf", "md"))
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
    # Leo el documento
    text = read_document(uploaded_file)

    # Muestro el documento
    st.write("### Transcripción del llamado")
    st.write(text)

    # Genero la respuesta
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
    print("\n" + reply)

    # Muestro la respuesta
    st.write("### Respuesta")
    st.write(reply)