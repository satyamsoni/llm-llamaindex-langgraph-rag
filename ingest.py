#!/usr/bin/env python3

import os, sys, shutil, logging, warnings
from dotenv import load_dotenv
from tqdm import tqdm
from colorama import Fore, Back, init,Style
from pymilvus import connections, MilvusException,utility,Collection, FieldSchema, CollectionSchema, DataType
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex

#Ignore Logging
warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("llama_index").setLevel(logging.ERROR)

load_dotenv()
init(autoreset=True)
BOLD = '\033[1m'

# Ingest Class
class Ingest:
    BATCH_SIZE=0
    vector_store=""
    def __init__(self, num):
        bnrSty=Back.RED + Fore.BLACK + BOLD
        print(bnrSty+" *------------------------------------------------------------------------------* ")
        print(bnrSty+" |                    LLM Ingest RAG [ llamaIndex, Milvus ]                     | ")
        print(bnrSty+" |                             by : Satyam Swarnkar                             | ")
        print(bnrSty+" *------------------------------------------------------------------------------* ")
        self.MILVUS_HOST=os.environ.get("MILVUS_HOST")
        self.MILVUS_PORT=os.environ.get("MILVUS_PORT")
        self.MILVUS_ALIAS=os.environ.get("MILUS_ALIAS")
        self.RAG_COLLECTION=os.environ.get("RAG_COLLECTION")
        self.EMBEDDING_MODEL=os.environ.get("EMBEDDING_MODEL")
        self.INGEST_DIR=os.environ.get("INGEST_DIR")
        self.PROCESSED_DIR=os.environ.get("PROCESSED_DIR")
        try:
            self.BATCH_SIZE=int(num)
        except (TypeError, ValueError):
            self.BATCH_SIZE=1000
        self.start()
    def start(self):
        #Process documents
        all_files = [os.path.join(self.INGEST_DIR, f) for f in os.listdir(self.INGEST_DIR)]
        all_files.sort()
        batch_files = all_files[:self.BATCH_SIZE]

        if not batch_files:
            print(Fore.GREEN+"  ✅ All documents are ingested.")
            return

        print(Fore.YELLOW+" Vector Store : Connecting", end="\r")
        try:
            connections.connect(self.MILVUS_ALIAS, host=self.MILVUS_HOST, port=self.MILVUS_PORT)
            if not utility.has_collection(self.RAG_COLLECTION):
                self.createCollection()
            self.vector_store = MilvusVectorStore( collection_name=self.RAG_COLLECTION, dim=768)
            print(Fore.GREEN+" Vector Store : ✔️         ", end="\r")
            print("\n")
        except MilvusException as e:
            print(Fore.RED+" Vector Store : ❌ Error     ", end="\r")
            print("\n")
        except Exception as e:
            print(Fore.RED+" Vector Store : ❌ General error", end="\r")
            print(e)
            print("\n")

        self.embed_model = OllamaEmbedding(model_name=self.EMBEDDING_MODEL)
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
        try:
            documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
            
            # Create a storage context connected to Milvus
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

            # Build index and send embeddings to Milvus
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=self.embed_model
            )
            coll = Collection(self.RAG_COLLECTION, self.MILVUS_ALIAS)
            coll.load()
            print(f"🔢 Entities in '{coll}':", coll.num_entities)
            # Persist metadata locally
            #index.storage_context.persist("./storage")

            # Move the processed file
            #shutil.move(file_path, os.path.join(self.PROCESSED_DIR, os.path.basename(file_path)))
            return True

        except Exception as e:
            print(Fore.RED + f" ❌ Failed {file_path}: {e}")
            return False

    def createCollection(self):
        fields = [
            FieldSchema(name="doc_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
        ]
        schema = CollectionSchema(fields,description="RAG Data")
        Collection(name=self.RAG_COLLECTION, schema=schema)
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <number_of_docs>")
    else:
        app = Ingest(sys.argv[1])