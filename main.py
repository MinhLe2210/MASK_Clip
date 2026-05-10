import os
import sys
from pymilvus import connections, utility


def main():
    milvus_host = os.getenv("MILVUS_HOST")
    milvus_port = os.getenv("MILVUS_PORT", "19530")
    milvus_database = os.getenv("MILVUS_DATABASE", "default")
    collection_name = os.getenv(
        "COLLECTION_NAME",
        "ai_detector_images_deduplicate",
    )

    if not milvus_host:
        print("ERROR: MILVUS_HOST is not set.")
        sys.exit(1)

    print(f"Connecting to Milvus: host={milvus_host}, port={milvus_port}, database={milvus_database}")
    print(f"Target collection: {collection_name}")

    connections.connect(
        alias="default",
        host=milvus_host,
        port=milvus_port,
        db_name=milvus_database,
    )

    try:
        if utility.has_collection(collection_name):
            print(f"Collection exists. Dropping: {collection_name}")
            utility.drop_collection(collection_name)
            print(f"Dropped collection: {collection_name}")
        else:
            print(f"Collection does not exist: {collection_name}")

    finally:
        connections.disconnect(alias="default")
        print("Disconnected from Milvus.")


if __name__ == "__main__":
    main()