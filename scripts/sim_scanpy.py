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
3. When predicting arguments, please only predict the arguments that are required by the user query. If an argument is not relevant to user query, please leave it as default and do not predict it.
"""
# Create an API agent for OncoKB
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
question = "visualize the t-SNE plot of the cells"
# question = "visualize the t-SNE plot of the cells, where cells are colored by the louvain clustering."
data = krumsiek11()
# data = pbmc3k()
scanpy_agent.execute(question, data=data)

pass

# Jiahang (TODO, severe): when arguments scale up, arg prediction mess up.