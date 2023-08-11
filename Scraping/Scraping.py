import os
import numpy as np
from pdfminer.high_level import extract_text
import pandas as pd
import openai
import tiktoken

max_tokens = 500
modelo = 'gpt-4'

def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie

def split_into_many(tokenizer, text, max_tokens=max_tokens):
    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    # Add the last chunk to the list of chunks
    if chunk:
        chunks.append(". ".join(chunk) + ".")

    return chunks

def leerInsumos():
    rutaInsumos = 'Insumos'
    archivoScrapeado = 'Procesado/Unificado.txt'
    with open(archivoScrapeado, "a", encoding="utf-8") as archivoScrapeado:
        # Itera sobre los archivos en la carpeta
        for nombre_archivo in os.listdir(rutaInsumos):
            # Ruta completa del archivo
            ruta_archivo = os.path.join(rutaInsumos, nombre_archivo)

            # Verifica si es un archivo PDF
            if nombre_archivo.lower().endswith(".pdf"):
                try:
                    # Extrae el texto del archivo PDF
                    texto_pdf = extract_text(ruta_archivo)
                    print(f'Leyendo: {ruta_archivo}')
                    # Escribe el texto en el archivo unificado
                    archivoScrapeado.write(texto_pdf)
                except Exception as e:
                    print(f"Error al leer el archivo PDF: {nombre_archivo}")
                    print(str(e))

def procesarArchivoUnificado():
    # Create a list to store the text files
    texts = []
    # Get all the text files in the text directory
    for file in os.listdir("Procesado"):
        # Open the file and read the text
        with open("Procesado/" + file, "r", encoding="UTF-8") as f:
            text = f.read()

            # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
            texts.append((file[11:-4].replace('-', ' ').replace('_', ' ').replace('#update', ''), text))

    # Create a dataframe from the list of texts
    df = pd.DataFrame(texts, columns=['fname', 'text'])

    # Set the text column to be the raw text with the newlines removed
    df['text'] = df.fname + ". " + remove_newlines(df.text)
    csv = f'Procesado/scrapedUnificado.csv'
    df.to_csv(csv)
    df.head()

    # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
    tokenizer = tiktoken.get_encoding("cl100k_base")

    df = pd.read_csv(csv, index_col=0)
    df.columns = ['title', 'text']

    # Tokenize the text and save the number of tokens to a new column
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    # Visualize the distribution of the number of tokens per row using a histogram
    df.n_tokens.hist()

    shortened = []

    # Loop through the dataframe
    for row in df.iterrows():

        # If the text is None, go to the next row
        if row[1]['text'] is None:
            continue

        # If the number of tokens is greater than the max number of tokens, split the text into chunks
        if row[1]['n_tokens'] > max_tokens:
            shortened += split_into_many(tokenizer, row[1]['text'])

        # Otherwise, add the text to the list of shortened texts
        else:
            shortened.append(row[1]['text'])

    df = pd.DataFrame(shortened, columns=['text'])
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    df.n_tokens.hist()

    df['embeddings'] = df.text.apply(
        lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
    embeddings = f'Procesado/embeddingsUnificado.csv'
    df.to_csv(embeddings)
    df.head()

    df = pd.read_csv(embeddings, index_col=0)
    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
    df.head()
    pkl = f'Procesado/dfUnificado.pkl'
    df.to_pickle(pkl)

# leerInsumos()#leo el/los pdf y creo un archivo txt con todo el texto
# procesarArchivoUnificado()#procesa el txt y crea los archivos necesario para terminar en un pkl con todo el contenido de los txt