#!/usr/bin/env python3
import sys,os
from dotenv import load_dotenv
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType

load_dotenv()

class MilvusShell:
    def __init__(self):
        self.alias = os.environ.get("MILVUS_ALIAS")
        port=os.environ.get("MILVUS_PORT")
        host=os.environ.get("MILVUS_HOST")
        connections.connect(self.alias, host=host, port=port)
        print(f"✅ Connected to Milvus at {host}:{port}")

    def list_collections(self):
        cols = utility.list_collections(using=self.alias)
        print("📂 Collections:", cols)

    def drop_collection(self, name):
        if utility.has_collection(name, using=self.alias):
            utility.drop_collection(name, using=self.alias)
            print(f"✅ Collection '{name}' dropped")
        else:
            print(f"⚠️ Collection '{name}' not found")

    def info_collection(self, name):
        if not utility.has_collection(name, using=self.alias):
            print(f"⚠️ Collection '{name}' not found")
            return
        coll = Collection(name=name, using=self.alias)
        print("ℹ️ Info:", coll.describe())

    def count_entities(self, name):
        if not utility.has_collection(name, using=self.alias):
            print(f"⚠️ Collection '{name}' not found")
            return
        coll = Collection(name=name, using=self.alias)
        """index_params = {
            "metric_type": "L2",        # or "COSINE" / "IP"
            "index_type": "IVF_FLAT",   # other options: HNSW, IVF_SQ8, IVF_PQ
            "params": {"nlist": 1024}   # number of clusters, adjust based on data size
        }

        coll.create_index(field_name="embedding", index_params=index_params)
        """
        coll.load()
        print(f"🔢 Entities in '{name}':", coll.num_entities)
    def create(self,name):
        fields = [
            FieldSchema(name="doc_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
        ]
        schema = CollectionSchema(fields)
        Collection(name, schema)
        print(f"🔢 Collection created :'{name}'")
    def run(self):
        print("💡 Type 'help' for commands")
        while True:
            cmd = input("milvus> ").strip().split()
            if not cmd:
                continue
            action = cmd[0].lower()

            if action in ["exit", "quit"]:
                print("👋 Bye!")
                sys.exit(0)
            elif action == "help":
                print("Commands: list, info <name>, count <name>, drop <name>,create <name>, exit")
            elif action == "list":
                self.list_collections()
            elif action == "info" and len(cmd) > 1:
                self.info_collection(cmd[1])
            elif action == "count" and len(cmd) > 1:
                self.count_entities(cmd[1])
            elif action == "drop" and len(cmd) > 1:
                self.drop_collection(cmd[1])
            elif action == "create" and len(cmd) > 1:
                self.create(cmd[1])
            else:
                print("⚠️ Unknown command. Type 'help'.")


if __name__ == "__main__":
    shell = MilvusShell()
    shell.run()
