import openai
import os
import pandas as pd
from flask import Flask
from flask import request
from langchain import LLMChain
from openai.embeddings_utils import distances_from_embeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import (AIMessage, HumanMessage, SystemMessage)
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.prompts import PromptTemplate
import warnings


warnings.filterwarnings('ignore')

# Defino las variables a utilizar y cargo la base de conocimiento obtenida del scraping
openai.api_key = os.getenv("OPENAI_API_KEY")
max_tokens = 500
pkl = f'../Scraping/Procesado/dfUnificado.pkl'
df = pd.read_pickle(pkl)
modelo = 'gpt-4'
max_len = 3000
size = "ada"
preguntas = []
app = Flask(__name__)
templateSystem = """Eres un asistente virtual de un hospital, que responde dudas sobre el cáncer de mama y permite agendar citas para exámenes médicos o consultas médicas. 
                 Nunca rompas el personaje. Me proporcionarás respuestas basadas en la información dada. Si la respuesta no está incluida, di exactamente \'Hmm, no estoy seguro\' 
                 y detente. Al contexto debes llamarlo \'información proporcionada por el Ministerio de Salud\'. Debes continuar el diálogo, revisa los mensajes anteriores antes 
                 de responder. Niega responder cualquier pregunta que no esté relacionada con la información. Aquí está la pregunta:{input}"""
templateAgendar = """Eres un asistente virtual de un hospital que tu respuesta será procesada por un software. 
                 Nunca rompas el personaje. Debes analizar si el humano está solicitando una consulta con el doctor, o si quiere agendar para un examen. En ese caso SOLO deberas
                 responder: "AGENDAR", y nada más.
                 Si no quiere agendar un examen médico, o una consulta con el doctor, debes responder que no entendiste su pregunta.
                 Aquí está el diálogo de tu conversación con el humano:{input}"""


prompt_infos = [
    {
        "name": "Agendar",
        "description": "Es bueno para responder consultas sobre el agendamiento de exámenes o consultas médicas.",
        "prompt_template": templateAgendar
    },
    {
        "name": "general",
        "description": "Es bueno para todas las respuestas, excepto si la persona quiere agendar una consulta o examen.",
        "prompt_template": templateSystem
    }
    ]
MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising\
it will ultimately lead to a better response from the language model.\
The original input consists of a Context enclosed in \""", a Question, and a Dialogue enclosed in \"".\
If you are going to select "destination" as "Agendar", \
in "next_inputs," you MUST provide only the all "Dialogue" received from the input.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \ name of the prompt to use or "general"
    "next_inputs": string \ a potentially modified version of the original input, but if you are going to select destination as "Agendar", the "next_inputs" parameter MUST only include the all "Dialogue" received from the input.
}}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt \
names specified below OR it can be "general" if the input is not\
well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>"""
messages = [SystemMessage(content=templateSystem)]
template_string = '\n\nContext: """{context}"""\n\n---\n\nDialogue: ""{preguntas}""\n\n---\n\nQuestion: {question}'
prompt_template = ChatPromptTemplate.from_template(template_string)

def create_context(question, df, max_len=1800, size="ada"):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

llm = ChatOpenAI(temperature=0, model=modelo, verbose=True)
destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    destination_chains[name] = chain

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt, verbose=True)
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str, verbose=True
)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(), verbose=True,
)

router_chain = LLMRouterChain.from_llm(llm, router_prompt, verbose=True)
chain = MultiPromptChain(router_chain=router_chain,
                         destination_chains=destination_chains,
                         default_chain=default_chain, verbose=True
                        )

def procesarRespuesta(respuesta):
    respuesta = respuesta.replace("Assistant: ", "")
    respuesta = respuesta.replace("IA: ", "")
    return respuesta

@app.route('/envioPreguntaSalud', methods=['POST'])
def envioPreguntaSalud():
    datosIn = request.get_json()
    preguntasrespuestas = datosIn.get("conversacion")
    preguntas = []
    for unapregunta in preguntasrespuestas:
        role = unapregunta["role"]
        content = unapregunta["content"]
        if role == 'user':
            preguntas.append(f'Human: {content}')
        else:
            preguntas.append(f'IA: {content}')
    question = datosIn.get("pregunta")
    preguntas.append(f'Human: {question}')
    preguntas_formateado = '""\n' + '", "'.join(preguntas) + '\n""'
    context = create_context(question, df, max_len=max_len, size=size)
    pregunta = f'\n\nContext: {context}\n\n---\n\nDialogue:{preguntas_formateado}\n\n---\n\nQuestion: {question}'
    print(pregunta)
    respuesta = chain.run(pregunta)
    respuesta = procesarRespuesta(respuesta)
    return {'mensaje': respuesta}

app.run(port=5051)