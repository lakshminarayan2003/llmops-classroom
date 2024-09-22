from sentence_transformers import SentenceTransformer
import json
import time

import numpy as np
import pandas as pd
import requests
import redis
from redis.commands.search.field import (
    NumericField,
    TagField,
    TextField,
    VectorField,
)
import phoenix as px
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Load the pre-trained model and tokenizer from Hugging Face
model_name = "facebook/bart-large-cnn"  # Example model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
session = px.launch_app()
# Initialize the pipeline
qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
embedder = SentenceTransformer('msmarco-distilbert-base-v4')

client = redis.Redis(host="172.20.0.3", port=6379, decode_responses=True)


queries = [
    "Can you suggest me a best bikes for kids of 3 to 4 years old and also give me a brief about the same",
]

encoded_queries = embedder.encode(queries)

def create_query_table(query, queries, encoded_queries, extra_params=None):
    """
    Creates a query table.
    """
    results_list = []
    for i, encoded_query in enumerate(encoded_queries):
        result_docs = (
            client.ft("idx:bikes_vss")
            .search(
                query,
                {"query_vector": np.array(encoded_query, dtype=np.float32).tobytes()}
                | (extra_params if extra_params else {}),
            )
            .docs
        )
        for doc in result_docs:
            vector_score = round(1 - float(doc.vector_score), 2)
            results_list.append(
                {
                    "query": queries[i],
                    "score": vector_score,
                    "id": doc.id,
                    "brand": doc.brand,
                    "model": doc.model,
                    "description": doc.description,
                }
            )

    # Optional: convert the table to Markdown using Pandas
    queries_table = pd.DataFrame(results_list)
    queries_table.sort_values(
        by=["query", "score"], ascending=[True, False], inplace=True
    )
    # queries_table["query"] = queries_table.groupby("query")["query"].transform(
    #     lambda x: [x.iloc[0]] + [""] * (len(x) - 1)
    # )
    # queries_table["description"] = queries_table["description"].apply(
    #     lambda x: (x[:497] + "...") if len(x) > 500 else x
    # )
    return queries_table

query = (
    Query("(*)=>[KNN 3 @vector $query_vector AS vector_score]")
    .sort_by("vector_score")
    .return_fields("vector_score", "id", "brand", "model", "description")
    .dialect(2)
)

table = create_query_table(query, queries, encoded_queries)
print(table)
context_texts = table['description'].to_list(
)

azure_openai_api_key = '<Your open ai api key>'
azure_openai_endpoint = 'endpoint_name'
azure_openai_deployment_name = 'deployement name'
import openai
openai.api_type = "azure"
openai.api_base = azure_openai_endpoint  # Your Azure OpenAI endpoint
openai.api_version = "2023-03-15-preview"  # The current Azure OpenAI version
openai.api_key = azure_openai_api_key
client = openai.AzureOpenAI(
        azure_endpoint=azure_openai_endpoint,
        azure_deployment="llmops_CT_GPT4o",
        api_key=azure_openai_api_key,
        api_version="2023-09-01-preview",

    )

from phoenix.otel import register
import phoenix as px
# px.launch_app()
tracer_provider = register(
  project_name="my-llm-app",
  endpoint="http://localhost:4317/v1/traces"
)
def answer_query_with_context(user_query, context_texts):
    # Concatenate the retrieved contexts
    from openinference.instrumentation.openai import OpenAIInstrumentor
    (session := px.launch_app()).view()
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    context = " ".join(context_texts)
    
    # Define the prompt for GPT-4
    prompt = (
        f"Context: {context}\n\n"
        f"Question: {user_query}\n\n"
        "Answer based only on the context provided. If the answer is not in the context, say 'I don't know'."
    )

    # Call the Azure OpenAI API using the deployed model
    response = client.chat.completions.create(
        model='llmops_CT_GPT4o',  # Your deployment name in Azure
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.5,
    )


    # Extract the generated answer from the response
    print(response.choices[0].message.content)
    generated_answer = response.choices[0].message.content
    # generated_answer = response.choices[0].message['content'].strip()

    # Handle cases where the model responds with "I don't know"
    if "I don't know" in generated_answer or not generated_answer:
        return "I don't know"
    else:
        return generated_answer

# Example call to the function
answer = answer_query_with_context(user_query=queries[0], context_texts=context_texts)
print(answer)
# # context = " ".join(context_texts)
# def answer_query_with_context(user_query, context_texts):

#     context = " ".join(context_texts)
    
    
#     input_text = (
#         f"Context: {context}\n\n"
#         f"Question: {user_query}\n\n"
#         "Answer based only on the context provided. If the answer is not in the context, say 'I don't know'."
#     )

#     response = qa_pipeline(input_text, max_length=150, clean_up_tokenization_spaces=True)
    

#     generated_answer = response[0]['generated_text'].strip()
    

#     if "I don't know" in generated_answer or not generated_answer or generated_answer.lower() in ["no", "i don't know"]:
#         return "I don't know"
#     else:
#         return generated_answer
# answer_query_with_context(user_query=queries[0],context_texts=context_texts) 
# px.active_session().url
