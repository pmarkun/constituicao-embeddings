import pandas as pd
import numpy as np
import re
from transformers import GPT2TokenizerFast
from tqdm import tqdm
import openai
import pickle
import os


openai.api_key = os.getenv("OPENAI_APIKEY")

class LawAnalyzer:
    def __init__(self, project, debug=False):
        self.debug = debug
        self.project = project
        df_filename = "data/" + project + "_df.csv" #ToFix
        embedding_filename = "data/" + project + "_embedding.pickle"

        self.df = self.load_df(df_filename)

        self.embeddings = self.load_embeddings(embedding_filename)
        self.config = {
            'MAX_SECTION_LEN' : 1024,
            'SEPARATOR' : "\n* ",
            'SEPARATOR_LEN' : 4,
            "EMBEDDING_MODEL" : "text-embedding-ada-002",
            "COMPLETION_MODEL" : {
                "model" : "text-davinci-003",
                "temperature": 0.0,
                "max_tokens": 300
                }
        }

    def get_embedding(self, text):
        result = openai.Embedding.create(
        model=self.config['EMBEDDING_MODEL'],
        input=text
        )
        return result["data"][0]["embedding"]

    def compute_embeddings(self):
        """
        Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
        
        Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
        """
        return {
            idx: self.get_embedding(r.content) for idx, r in tqdm(self.df.iterrows())
        }
    
    def load_df(self, df_filename):
        df = pd.read_csv(df_filename)
        return df

    def load_embeddings(self, embedding_filename):
        """
        Read the document embeddings and their keys from a CSV.
        
        fname is the path to a CSV with exactly these named columns: 
            "title", "heading", "0", "1", ... up to the length of the embedding vectors.
        """
        if os.path.exists(embedding_filename):
            with open(embedding_filename, "rb") as embedding_file:
                embeddings = pickle.load(embedding_file)

            if self.debug:
                print("Embedding loaded...")

        else: 
            with open(embedding_filename, "wb") as embedding_file:
                embeddings = self.compute_embeddings()
                pickle.dump(self.embeddings, embedding_filename)

            if self.debug:
                print("Embedding created...")

        return(embeddings) 
        
    def vector_similarity(self, x, y):
        """
        Returns the similarity between two vectors.
        
        Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
        """
        return np.dot(np.array(x), np.array(y))

    def build_query_similarity(self, query):
        """
        Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
        to find the most relevant sections. 
        
        Return the list of document sections, sorted by relevance in descending order.
        """
        query_embedding = self.get_embedding(query)
        
        document_similarities = sorted([
            (self.vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in self.embeddings.items()
        ], reverse=True)
        
        return document_similarities

    def similar(self, question):
        most_relevant_document_sections = self.build_query_similarity(question)
        
        chosen_sections = {}
        chosen_sections_len = 0
        chosen_sections['sections'] = []
        chosen_sections['indexes'] = []
        
        for _, section_index in most_relevant_document_sections:
            # Add contexts until we run out of space.        
            document_section = self.df.loc[section_index]
            
            chosen_sections_len += document_section.tokens + self.config['SEPARATOR_LEN']
            if chosen_sections_len > self.config['MAX_SECTION_LEN']:
                break
                
            chosen_sections['sections'].append(self.config['SEPARATOR'] + document_section.no_artigo + " " + document_section.content.replace("\n", " "))
            chosen_sections['indexes'].append(str(section_index))
        
        if self.debug:
            print("## Chosen sections ##")
            print(chosen_sections)
        
        return(chosen_sections)

    def buildPrompt(self, question, prompt_template=None, extra=None):

        if prompt_template and os.path.isfile(f"prompts/{prompt_template}.txt"):
            template = open(f"prompts/{prompt_template}.txt", "r").read()
        else:
            template = open(f"prompts/default.txt", "r").read()
        
        chosen_sections = self.similar(question)
                
        context = "".join(chosen_sections['sections'])
    
        prompt = template.replace("<<CONTEXT>>",context).replace("<<QUESTION>>",question)

        if extra:
            for tag in extra.keys():
                prompt = prompt.replace(f"<<{tag}>>", extra[tag])
        
        if self.debug:
            print("## Prompt ##")
            print(prompt)

        return prompt

    def buildFromText(self, tokenizer=GPT2TokenizerFast.from_pretrained("gpt2")):
        # Abrir o arquivo e ler as linhas
        complete_text =  open(f"data/{self.project}.txt", 'r').read()
        articles = re.split("\n\nArt. [0-9\-A]*[º|\.] ",complete_text)
        no_articles = ["Art. " + n for n in re.findall("\n\nArt. ([0-9\-A]*)[º|\.] ",complete_text)]
        no_articles.insert(0, "PREAMBULO")
        tokens = [len(tokenizer(t)['input_ids']) for t in tqdm(articles)]

        df = pd.DataFrame({
            "no_article": no_articles,
            "content": articles,
            "tokens" : tokens
        })

        df.to_csv(f"data/{self.project}_df.csv", index=False)
        return(df)

    def answer(self, query, **kwargs):
        
        prompt = self.buildPrompt(query, **kwargs)

        response = openai.Completion.create(
                    prompt=prompt,
                    **self.config['COMPLETION_MODEL']
                )

        answer = response["choices"][0]["text"].strip(" \n")
        if "Eu não sei." in answer:
            answer += "\n" + self.answer(query, prompt_template="uncertain")
        return answer
