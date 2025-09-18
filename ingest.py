#!/usr/bin/env python3

import os, sys, shutil, logging, warnings
from tqdm import tqdm
from colorama import Fore, Back, init,Style
from pymilvus import connections, MilvusException
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

#Ignore Logging
warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("llama_index").setLevel(logging.ERROR)

init(autoreset=True)
BOLD = '\033[1m'

# Global vars
INGEST_DIR = "ingest"
PROCESSED_DIR = "docs"
EMBEDDING_MODEL="nomic-embed-text"

# Ingest Class
class Ingest:
    BATCH_SIZE=0
    def __init__(self, num):
        bnrSty=Back.RED + Fore.BLACK + BOLD
        print(bnrSty+" *------------------------------------------------------------------------------* ")
        print(bnrSty+" |                    LLM Ingest RAG [ llamaIndex, Milvus ]                     | ")
        print(bnrSty+" |                             by : Satyam Swarnkar                             | ")
        print(bnrSty+" *------------------------------------------------------------------------------* ")
        try:
            self.BATCH_SIZE=int(num)
        except (TypeError, ValueError):
            self.BATCH_SIZE=1000
        self.start()
    def start(self):
        """Process documents one by one with progress bar"""
        all_files = [os.path.join(INGEST_DIR, f) for f in os.listdir(INGEST_DIR)]
        all_files.sort()
        batch_files = all_files[:self.BATCH_SIZE]

        if not batch_files:
            print(Fore.GREEN+" All documents are ingested.")
            return

        print(Fore.YELLOW+" Vector Store : Connecting", end="\r")
        try:
            connections.connect("default", host="127.0.0.1", port="19530")
            self.vector_store = MilvusVectorStore(
                collection_name="rag_docs",
                dim=1024,   # embedding size
            )
            print(Fore.GREEN+" Vector Store : ✔️         ", end="\r")
        except MilvusException as e:
            print(Fore.RED+" Vector Store : ❌ Error     ", end="\r")
            print("\n")
        except Exception as e:
            print(Fore.RED+" Vector Store : ❌ General error", end="\r")
            print("\n")

        self.embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL)
        print(Fore.YELLOW+f" Found {len(batch_files)} docs for ingestion")

        success, failed = 0, 0
        with tqdm(batch_files, desc="Ingesting", unit="doc") as pbar:
            for f in pbar:
                # Show current file name (basename only)
                pbar.set_postfix_str(os.path.basename(f))
                if self.embed_rag(f):
                    success += 1
                else:
                    failed += 1

        print(Fore.GREEN+f" ✅ Completed: {success} | {Fore.RED}❌ Failed: {failed}")

    def embed_rag(self, file_path: str):
        # Process document-> embed + insert into Milvus + move file"""
        try:
            documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
            index = VectorStoreIndex.from_documents(
                documents,
                vector_store=self.vector_store,
                embed_model=self.embed_model
            )
            index.storage_context.persist("./storage")

            # Move file
            shutil.move(file_path, os.path.join(PROCESSED_DIR, os.path.basename(file_path)))
            return True
        except Exception as e:
            print(Fore.RED+f" ❌ Failed {file_path}: {e}")
            return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <number_of_docs>")
    else:
        app = Ingest(sys.argv[1])