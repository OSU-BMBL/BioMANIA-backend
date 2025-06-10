from biochatter.llm_connect import GptConversation
from biochatter.api_agent import APIAgent, ScanpyQueryBuilder, ScanpyFetcher, ScanpyInterpreter

import os
import scanpy
from scanpy.datasets import pbmc3k, krumsiek11

from dotenv import load_dotenv
load_dotenv()

# Jiahang (TODO): these prelim settings should be organized.
scanpy.settings.datasetdir = os.environ.get("DATA")
scanpy.settings.figdir = os.environ.get("FIG")
if not os.path.exists(scanpy.settings.figdir):
    os.makedirs(scanpy.settings.figdir)



system_prompt = """
You are a professional bioinformatician. 
1. You have access to the data object named `data`.
2. Please only use the provided tools. Do not use any tools that are not provided.
"""
# Create an API agent
query_builder_conv = GptConversation(
    model_name=os.environ.get("MODEL"), 
    prompts={
        "primary_model_prompts": system_prompt
    }
)
interpreter_conv = GptConversation(
    model_name=os.environ.get("MODEL"), 
    prompts={
        "primary_model_prompts": system_prompt
    }
)

scanpy_agent = APIAgent(
    query_builder=ScanpyQueryBuilder(
        conversation=query_builder_conv,
    ),
    fetcher=ScanpyFetcher(),
    interpreter=ScanpyInterpreter( # Jiahang (TODO): explain codes, args, etc. see biomania.
        conversation=interpreter_conv,
    )
)

# Execute a query
# correct
question = "visualize the t-SNE plot of the cells"
# sc.tl.tsne, key_added = "tsne", incorrect
# question = "visualize the t-SNE plot of the cells, where cells are colored by the louvain clustering."
# sc.pl.tsne, color = 'louvain', incorrect
# question = "visualize the t-SNE plot of the cells, where cells are colored by the louvain clustering label."
data = krumsiek11()
# data = pbmc3k()
scanpy_agent.execute(question, data=data)

pass

# Jiahang (TODO, severe): when arguments scale up, arg prediction mess up.