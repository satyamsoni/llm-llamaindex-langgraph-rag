#!/usr/bin/env python3

import sys
from pymilvus import connections
from dotenv import load_dotenv
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from langgraph.graph import StateGraph
import warnings,logging
from colorama import Fore, Back, Style, init

init(autoreset=True)
load_dotenv()
BOLD = '\033[1m'
warnings.filterwarnings("ignore")

# --- LangGraph State ---
class State(dict):
	question: str
	answer: str

def format_response(state: State):
	"""Enforce response < 200 chars"""
	text = state["answer"]
	state["answer"] = text[:200] if len(text) > 200 else text
	return state

class LLM:
	def __init__(self):
		bnrSty=Back.RED + Fore.BLACK + BOLD
		print(bnrSty+" *------------------------------------------------------------------------------* ")
		print(bnrSty+" |                LLM Chat+RAG [ llamaIndex,langGraph, Milvus ]                 | ")
		print(bnrSty+" |                             by : Satyam Swarnkar                             | ")
		print(bnrSty+" *------------------------------------------------------------------------------* ")
		self.MILVUS_HOST=os.environ.get("MILVUS_HOST")
        self.MILVUS_PORT=os.environ.get("MILVUS_PORT")
        self.MILVUS_ALIAS=os.environ.get("MILUS_ALIAS")
        self.RAG_COLLECTION=os.environ.get("RAG_COLLECTION")
        self.EMBEDDING_MODEL=os.environ.get("EMBEDDING_MODEL")
        self.LLM_MODEL=os.environ.get("LLM_MODEL")
		# Connect to Milvus
		connections.connect(self.MILVUS_ALIAS, host=self.MILVUS_HOST, port=self.MILVUS_PORT)
		self.vector_store = MilvusVectorStore(
			collection_name=self.RAG_COLLECTION,
			dim=768
		)

		# Create index context
		storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
		self.index = VectorStoreIndex.from_vector_store(
			vector_store=self.vector_store,
			storage_context=storage_context,
			embed_model=OllamaEmbedding(model_name=self.EMBEDDING_MODEL)
		)

		# LLM
		self.ollama = Ollama(model=self.LLM_MODEL)
		# LangGraph for formatting
		graph = StateGraph(State)
		graph.add_node("format", format_response)
		graph.set_entry_point("format")
		self.formatter = graph.compile()

	def query(self, question: str) -> str:
		# Step 1: Retrieve context
		query_engine = self.index.as_query_engine(llm=self.ollama)
		response = query_engine.query(question)
		print("Number of documents in index:", len(self.index.docstore.docs))
		print(response)
		# Step 2: LangGraph formatting
		state = {"question": question, "answer": str(response)}
		result = self.formatter.invoke(state)
		return result["answer"]

# --- Example Usage ---
if __name__ == "__main__":
	app = LLM()
	while True:
		q = input("\n🧠 > ")
		if q.lower() in ["exit", "quit"]:
			sys.exit(0)
		ans = app.query(q)
		print("🤖:", ans)
