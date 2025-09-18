#!/usr/bin/env python3

import sys
from pymilvus import connections
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from langgraph.graph import StateGraph
import warnings,logging
from colorama import Fore, Back, Style, init

init(autoreset=True)
BOLD = '\033[1m'
EMBEDDING_MODEL="nomic-embed-text"
LLM_MODEL = "mistral"
RAG_COLLECTION="rag_docs"
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
		# Connect to Milvus
		connections.connect("default", host="127.0.0.1", port="19530")
		self.vector_store = MilvusVectorStore(
			collection_name=RAG_COLLECTION,
			dim=768
		)

		# Create index context
		storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
		self.index = VectorStoreIndex.from_vector_store(
			vector_store=self.vector_store,
			storage_context=storage_context,
			embed_model=OllamaEmbedding(model_name=EMBEDDING_MODEL)
		)

		# LLM
		self.llm = Ollama(model=LLM_MODEL)

		# LangGraph for formatting
		graph = StateGraph(State)
		graph.add_node("format", format_response)
		graph.set_entry_point("format")
		self.formatter = graph.compile()

	def query(self, question: str) -> str:
		# Step 1: Retrieve context
		query_engine = self.index.as_query_engine(llm=self.llm)
		response = query_engine.query(question)

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
