import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from chromadb.utils import embedding_functions
from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = Path(__file__).resolve().parents[1]
KB_DIR = BASE_DIR / "0_Data" / "5_Knowledge_Base"
KB_SOURCE_DIR = KB_DIR / "source"
CHROMA_DIR = KB_DIR / "chroma_db"
CHROMA_EXPORT_DIR = CHROMA_DIR / "store"
CHROMA_RUNTIME_DIR = (
    Path(os.getenv("LOCALAPPDATA", tempfile.gettempdir())) / "AI_for_AI_Sec" / "chroma_db_runtime"
)
COLLECTION_NAME = "ai_sec_knowledge_base"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2 (via Chroma ONNXMiniLM_L6_V2)"
EMBEDDING_BACKEND_FILE = "embedding_backend.json"
TFIDF_VECTORIZER_FILE = "tfidf_vectorizer.pkl"

SOURCE_FILES = [
    KB_SOURCE_DIR / "mitre_atlas_knowledge.json",
    KB_SOURCE_DIR / "owasp_knowledge.json",
]


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def flatten_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [flatten_value(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        parts = []
        for key, inner_value in value.items():
            inner_text = flatten_value(inner_value)
            if inner_text:
                parts.append(f"{key}: {inner_text}")
        return "\n".join(parts)
    return str(value)


def build_document(item: Dict[str, Any]) -> str:
    lines: List[str] = [
        f"dataset: {item.get('dataset', '')}",
        f"id: {item.get('id', '')}",
        f"item_index: {item.get('item_index', '')}",
    ]

    if item.get("source_title"):
        lines.append(f"source_title: {item['source_title']}")
    if item.get("source_name"):
        lines.append(f"source_name: {item['source_name']}")

    for field in [
        "original_text",
        "requirement_text",
        "business_value",
    ]:
        if item.get(field):
            lines.append(f"{field}: {item[field]}")

    if item.get("implicit_risk_hints"):
        lines.append("implicit_risk_hints: " + ", ".join(item["implicit_risk_hints"]))

    if item.get("prevention_and_mitigation_strategies_subsections"):
        lines.append("prevention_and_mitigation_strategies:")
        lines.append(flatten_value(item["prevention_and_mitigation_strategies_subsections"]))

    if item.get("procedure"):
        procedure_lines = []
        for idx, step in enumerate(item["procedure"], start=1):
            procedure_lines.append(f"step_{idx}:")
            for key in ["tactic", "technique", "description", "mitigations"]:
                if key in step and step[key]:
                    procedure_lines.append(f"{key}: {flatten_value(step[key])}")
        lines.append("procedure:")
        lines.append("\n".join(procedure_lines))

    return "\n".join(lines)


def build_metadata(item: Dict[str, Any], source_path: Path) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "dataset": str(item.get("dataset", "")),
        "id": str(item.get("id", "")),
        "item_index": int(item.get("item_index", -1)),
        "global_id": int(item.get("global_id", -1)),
        "source_file": source_path.name,
    }

    if "source_name" in item:
        metadata["source_name"] = str(item.get("source_name") or "")
    if "source_title" in item:
        metadata["source_title"] = str(item.get("source_title") or "")

    return metadata


def prepare_chunks() -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for source_file in SOURCE_FILES:
        for item in load_json_list(source_file):
            chunk_id = f"{item.get('dataset', 'unknown')}-{item.get('global_id', len(chunks))}"
            chunks.append(
                {
                    "id": chunk_id,
                    "document": build_document(item),
                    "metadata": build_metadata(item, source_file),
                }
            )
    return chunks


def get_embedding_backend(documents: List[str]) -> Dict[str, Any]:
    try:
        embedding_function = embedding_functions.ONNXMiniLM_L6_V2()
        embeddings = embedding_function(documents)
        return {
            "backend": "onnx_minilm_l6_v2",
            "embeddings": embeddings,
            "query_encoder": embedding_function,
            "vectorizer": None,
        }
    except Exception as exc:
        print(f"MiniLM ONNX backend unavailable, falling back to TF-IDF. Reason: {exc}")
        vectorizer = TfidfVectorizer(stop_words="english")
        embeddings = vectorizer.fit_transform(documents).astype("float32").toarray().tolist()
        return {
            "backend": "tfidf",
            "embeddings": embeddings,
            "query_encoder": None,
            "vectorizer": vectorizer,
        }


def save_backend_artifacts(backend_name: str, vectorizer: Any) -> None:
    metadata = {"backend": backend_name}
    (CHROMA_RUNTIME_DIR / EMBEDDING_BACKEND_FILE).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if vectorizer is not None:
        with (CHROMA_RUNTIME_DIR / TFIDF_VECTORIZER_FILE).open("wb") as f:
            pickle.dump(vectorizer, f)


def load_backend_artifacts() -> Dict[str, Any]:
    metadata_path = CHROMA_RUNTIME_DIR / EMBEDDING_BACKEND_FILE
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing embedding backend metadata: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    backend_name = metadata["backend"]
    vectorizer = None
    if backend_name == "tfidf":
        with (CHROMA_RUNTIME_DIR / TFIDF_VECTORIZER_FILE).open("rb") as f:
            vectorizer = pickle.load(f)

    return {"backend": backend_name, "vectorizer": vectorizer}


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def export_runtime_db() -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    reset_directory(CHROMA_EXPORT_DIR)
    for item in CHROMA_RUNTIME_DIR.iterdir():
        destination = CHROMA_EXPORT_DIR / item.name
        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)


def ensure_runtime_db_from_export() -> None:
    if CHROMA_RUNTIME_DIR.exists():
        return
    if not CHROMA_EXPORT_DIR.exists():
        raise FileNotFoundError(
            f"Chroma runtime DB not found and no exported DB exists at {CHROMA_EXPORT_DIR}."
        )
    reset_directory(CHROMA_RUNTIME_DIR)
    for item in CHROMA_EXPORT_DIR.iterdir():
        destination = CHROMA_RUNTIME_DIR / item.name
        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)


def recreate_collection(client: chromadb.PersistentClient, collection_name: str):
    existing_names = {collection.name for collection in client.list_collections()}
    if collection_name in existing_names:
        client.delete_collection(collection_name)
    return client.create_collection(name=collection_name)


def build_chroma_db() -> chromadb.Collection:
    chunks = prepare_chunks()
    print(f"Prepared {len(chunks)} chunks from {len(SOURCE_FILES)} files.")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    reset_directory(CHROMA_RUNTIME_DIR)

    client = chromadb.PersistentClient(path=str(CHROMA_RUNTIME_DIR))
    collection = recreate_collection(client, COLLECTION_NAME)
    documents = [chunk["document"] for chunk in chunks]
    backend_info = get_embedding_backend(documents)
    save_backend_artifacts(backend_info["backend"], backend_info["vectorizer"])

    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        documents=documents,
        metadatas=[chunk["metadata"] for chunk in chunks],
        embeddings=backend_info["embeddings"],
    )
    export_runtime_db()
    print(f"Stored {collection.count()} chunks in Chroma runtime DB at {CHROMA_RUNTIME_DIR}.")
    print(f"Exported Chroma DB files to {CHROMA_EXPORT_DIR}.")
    print(f"Embedding backend used: {backend_info['backend']}")
    return collection


def test_chroma_db(
    query: str = "How to prevent prompt injection attacks?",
) -> Dict[str, Any]:
    ensure_runtime_db_from_export()
    client = chromadb.PersistentClient(path=str(CHROMA_RUNTIME_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    backend_info = load_backend_artifacts()

    knowledge_count = collection.count()
    first_record = collection.get(limit=1, include=["documents", "metadatas"])
    first_content = first_record["documents"][0] if first_record["documents"] else None
    first_metadata = first_record["metadatas"][0] if first_record["metadatas"] else None

    if backend_info["backend"] == "onnx_minilm_l6_v2":
        query_embedding = embedding_functions.ONNXMiniLM_L6_V2()([query])[0]
    else:
        query_embedding = (
            backend_info["vectorizer"].transform([query]).astype("float32").toarray().tolist()[0]
        )

    query_result = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    result = {
        "knowledge_count": knowledge_count,
        "first_content": first_content,
        "first_metadata": first_metadata,
        "embedding_backend": backend_info["backend"],
        "query": query,
        "query_result": query_result,
    }
    return result


def main() -> None:
    build_chroma_db()
    test_result = test_chroma_db()

    print("\n=== Test Summary ===")
    print(f"Knowledge count: {test_result['knowledge_count']}")
    print(f"Embedding backend: {test_result['embedding_backend']}")
    print(f"First metadata: {json.dumps(test_result['first_metadata'], ensure_ascii=False)}")
    print(f"First content preview: {str(test_result['first_content'])[:500]}")

    print("\n=== Query Test ===")
    print(f"Query: {test_result['query']}")
    query_documents = test_result["query_result"].get("documents", [[]])
    query_metadatas = test_result["query_result"].get("metadatas", [[]])
    query_distances = test_result["query_result"].get("distances", [[]])

    if query_documents and query_documents[0]:
        for idx, doc in enumerate(query_documents[0], start=1):
            metadata = query_metadatas[0][idx - 1] if query_metadatas and query_metadatas[0] else {}
            distance = query_distances[0][idx - 1] if query_distances and query_distances[0] else None
            print(f"Result {idx}: distance={distance}, metadata={json.dumps(metadata, ensure_ascii=False)}")
            print(doc[:500])
            print("-" * 80)
    else:
        print("No results returned for the query.")


if __name__ == "__main__":
    main()
