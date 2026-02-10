"""
Document RAG Module for Athletics AI Analyst

Provides semantic search over:
1. PDF documents (rulebooks, qualification standards) - stored in Azure Blob
2. Azure Blob parquet data (rankings, benchmarks, athletes)

All data is stored in Azure Blob Storage for cloud deployment.

Uses sentence-transformers for embeddings.
Vector embeddings stored as JSON in Azure Blob (no external DB needed).

Dependencies:
    pip install sentence-transformers pypdf2 azure-storage-blob
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from io import BytesIO

# Lazy import for sentence_transformers (heavy dependency, delays startup by 30-45s)
# Don't import at module level - import only when needed in VectorStore
EMBEDDINGS_AVAILABLE = None  # Will be set on first use
_SentenceTransformer = None  # Lazy-loaded class reference

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    try:
        import pypdf as PyPDF2
        PDF_AVAILABLE = True
    except ImportError:
        PDF_AVAILABLE = False

try:
    from azure.storage.blob import BlobServiceClient, ContainerClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

# Azure Blob configuration
CONTAINER_NAME = "personal-data"
AZURE_FOLDER = "athletics"
DOCUMENTS_BLOB_FOLDER = f"{AZURE_FOLDER}/documents"
EMBEDDINGS_BLOB_PATH = f"{AZURE_FOLDER}/embeddings/document_embeddings.json"

# Local cache paths (for offline/development)
BASE_DIR = Path(__file__).parent
LOCAL_DOCUMENTS_DIR = BASE_DIR / "documents"
LOCAL_CACHE_DIR = BASE_DIR / "cache"

# Ensure local directories exist
LOCAL_DOCUMENTS_DIR.mkdir(exist_ok=True)
LOCAL_CACHE_DIR.mkdir(exist_ok=True)


class DocumentProcessor:
    """Process PDF documents into searchable chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text from PDF with page numbers."""
        if not PDF_AVAILABLE:
            return []

        chunks = []
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)

                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    if text:
                        # Split into chunks
                        page_chunks = self._chunk_text(text, page_num, pdf_path)
                        chunks.extend(page_chunks)
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")

        return chunks

    def _chunk_text(self, text: str, page_num: int, source: str) -> List[Dict]:
        """Split text into overlapping chunks."""
        chunks = []
        words = text.split()

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)

            if len(chunk_text.strip()) > 50:  # Skip very short chunks
                chunks.append({
                    'text': chunk_text,
                    'page': page_num,
                    'source': os.path.basename(source),
                    'chunk_id': f"{os.path.basename(source)}_p{page_num}_{i}"
                })

        return chunks


class AzureBlobHelper:
    """Helper class for Azure Blob Storage operations."""

    def __init__(self):
        self.connection_string = self._get_connection_string()
        self.blob_service = None
        self.container_client = None

        if self.connection_string and AZURE_AVAILABLE:
            try:
                self.blob_service = BlobServiceClient.from_connection_string(self.connection_string)
                self.container_client = self.blob_service.get_container_client(CONTAINER_NAME)
            except Exception as e:
                print(f"Azure connection error: {e}")

    def _get_connection_string(self) -> Optional[str]:
        """Get Azure connection string from environment or Streamlit secrets."""
        conn_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        if not conn_str:
            try:
                import streamlit as st
                if hasattr(st, 'secrets') and 'AZURE_STORAGE_CONNECTION_STRING' in st.secrets:
                    conn_str = st.secrets['AZURE_STORAGE_CONNECTION_STRING']
            except:
                pass
        return conn_str

    def upload_json(self, blob_path: str, data: dict) -> bool:
        """Upload JSON data to Azure Blob."""
        if not self.container_client:
            return False
        try:
            json_data = json.dumps(data)
            blob_client = self.container_client.get_blob_client(blob_path)
            blob_client.upload_blob(json_data, overwrite=True)
            return True
        except Exception as e:
            print(f"Upload error: {e}")
            return False

    def download_json(self, blob_path: str) -> Optional[dict]:
        """Download JSON data from Azure Blob."""
        if not self.container_client:
            return None
        try:
            blob_client = self.container_client.get_blob_client(blob_path)
            data = blob_client.download_blob().readall()
            return json.loads(data)
        except Exception as e:
            print(f"Download error: {e}")
            return None

    def upload_pdf(self, local_path: str, blob_name: str) -> bool:
        """Upload PDF to Azure Blob documents folder."""
        if not self.container_client:
            return False
        try:
            blob_path = f"{DOCUMENTS_BLOB_FOLDER}/{blob_name}"
            blob_client = self.container_client.get_blob_client(blob_path)
            with open(local_path, 'rb') as f:
                blob_client.upload_blob(f, overwrite=True)
            return True
        except Exception as e:
            print(f"PDF upload error: {e}")
            return False

    def download_pdf(self, blob_name: str) -> Optional[bytes]:
        """Download PDF from Azure Blob."""
        if not self.container_client:
            return None
        try:
            blob_path = f"{DOCUMENTS_BLOB_FOLDER}/{blob_name}"
            blob_client = self.container_client.get_blob_client(blob_path)
            return blob_client.download_blob().readall()
        except Exception as e:
            print(f"PDF download error: {e}")
            return None

    def list_documents(self) -> List[str]:
        """List all PDF documents in Azure Blob."""
        if not self.container_client:
            return []
        try:
            blobs = self.container_client.list_blobs(name_starts_with=DOCUMENTS_BLOB_FOLDER)
            return [b.name.replace(f"{DOCUMENTS_BLOB_FOLDER}/", "") for b in blobs if b.name.endswith('.pdf')]
        except Exception as e:
            print(f"List error: {e}")
            return []

    def is_available(self) -> bool:
        """Check if Azure Blob is available."""
        return self.container_client is not None


class VectorStore:
    """
    Vector store for document embeddings using Azure Blob Storage.

    Stores embeddings as JSON in Azure Blob, no external vector DB needed.
    Uses cosine similarity for search.
    """

    def __init__(self):
        self.model = None
        self._model_loaded = False
        self.azure = AzureBlobHelper()
        self.embeddings_cache = None
        self.local_cache_path = LOCAL_CACHE_DIR / "embeddings_cache.json"

        # Don't load model here - defer to first use (lazy loading)
        # This speeds up dashboard startup by 30-45 seconds

        # Load existing embeddings (fast - just JSON)
        self._load_embeddings()

    def _ensure_model_loaded(self) -> bool:
        """Lazily load the embedding model on first use."""
        global EMBEDDINGS_AVAILABLE, _SentenceTransformer

        if self._model_loaded:
            return self.model is not None

        self._model_loaded = True

        # Lazy import of sentence_transformers
        if EMBEDDINGS_AVAILABLE is None:
            try:
                from sentence_transformers import SentenceTransformer
                _SentenceTransformer = SentenceTransformer
                EMBEDDINGS_AVAILABLE = True
            except ImportError:
                EMBEDDINGS_AVAILABLE = False

        if EMBEDDINGS_AVAILABLE and _SentenceTransformer:
            try:
                self.model = _SentenceTransformer('all-MiniLM-L6-v2')
                return True
            except Exception as e:
                print(f"Model load error: {e}")
                return False

        return False

    def _load_embeddings(self):
        """Load embeddings from Azure Blob or local cache."""
        # Try Azure first
        if self.azure.is_available():
            data = self.azure.download_json(EMBEDDINGS_BLOB_PATH)
            if data:
                self.embeddings_cache = data
                # Also save locally for offline access
                self._save_local_cache()
                return

        # Fall back to local cache
        if self.local_cache_path.exists():
            try:
                with open(self.local_cache_path, 'r') as f:
                    self.embeddings_cache = json.load(f)
            except:
                self.embeddings_cache = {'documents': [], 'embeddings': [], 'metadata': []}
        else:
            self.embeddings_cache = {'documents': [], 'embeddings': [], 'metadata': []}

    def _save_embeddings(self):
        """Save embeddings to Azure Blob and local cache."""
        # Save to Azure
        if self.azure.is_available():
            self.azure.upload_json(EMBEDDINGS_BLOB_PATH, self.embeddings_cache)

        # Also save locally
        self._save_local_cache()

    def _save_local_cache(self):
        """Save embeddings to local cache file."""
        try:
            with open(self.local_cache_path, 'w') as f:
                json.dump(self.embeddings_cache, f)
        except Exception as e:
            print(f"Local cache save error: {e}")

    def add_documents(self, chunks: List[Dict]) -> int:
        """Add document chunks to the vector store."""
        if not self._ensure_model_loaded():
            return 0

        texts = [c['text'] for c in chunks]
        metadatas = [{'page': c['page'], 'source': c['source'], 'chunk_id': c['chunk_id']} for c in chunks]

        # Generate embeddings
        embeddings = self.model.encode(texts).tolist()

        # Add to cache
        self.embeddings_cache['documents'].extend(texts)
        self.embeddings_cache['embeddings'].extend(embeddings)
        self.embeddings_cache['metadata'].extend(metadatas)

        # Save to Azure
        self._save_embeddings()

        return len(chunks)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        return dot_product / (magnitude1 * magnitude2)

    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for relevant document chunks using cosine similarity."""
        if not self._ensure_model_loaded() or not self.embeddings_cache.get('documents'):
            return []

        # Generate query embedding
        query_embedding = self.model.encode([query])[0].tolist()

        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings_cache['embeddings']):
            sim = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, sim))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top results
        results = []
        for idx, score in similarities[:n_results]:
            results.append({
                'text': self.embeddings_cache['documents'][idx],
                'source': self.embeddings_cache['metadata'][idx]['source'],
                'page': self.embeddings_cache['metadata'][idx]['page'],
                'score': score
            })

        return results

    def get_stats(self) -> Dict:
        """Get collection statistics."""
        doc_count = len(self.embeddings_cache.get('documents', []))
        # Check embeddings state: None = not loaded yet, True = available, False = unavailable
        if EMBEDDINGS_AVAILABLE is None:
            model_status = 'not_loaded'
        elif EMBEDDINGS_AVAILABLE:
            model_status = 'all-MiniLM-L6-v2'
        else:
            model_status = 'unavailable'
        return {
            'status': 'azure' if self.azure.is_available() else 'local',
            'count': doc_count,
            'embeddings_model': model_status
        }

    def clear(self):
        """Clear all embeddings."""
        self.embeddings_cache = {'documents': [], 'embeddings': [], 'metadata': []}
        self._save_embeddings()


class AthleticsRAG:
    """
    Combined RAG system for athletics data and documents.

    Searches:
    1. Vector store (PDF documents)
    2. Parquet data (via data_connector)
    """

    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.data_connector_available = False

        # Try to import data connector
        try:
            from data_connector import (
                get_ksa_rankings, get_ksa_athletes,
                get_benchmarks_data, get_data_mode
            )
            self.get_ksa_rankings = get_ksa_rankings
            self.get_ksa_athletes = get_ksa_athletes
            self.get_benchmarks_data = get_benchmarks_data
            self.get_data_mode = get_data_mode
            self.data_connector_available = True
        except ImportError:
            pass

    def index_documents(self, folder_path: str = None) -> Dict:
        """
        Index all PDF documents - from local folder OR Azure Blob.

        Args:
            folder_path: Local folder path, or None to use Azure Blob documents
        """
        results = {'processed': 0, 'chunks': 0, 'errors': [], 'source': 'local'}

        # Try Azure Blob first if no folder specified
        if folder_path is None:
            azure_docs = self.vector_store.azure.list_documents()
            if azure_docs:
                results['source'] = 'azure'
                for doc_name in azure_docs:
                    try:
                        # Download PDF from Azure
                        pdf_bytes = self.vector_store.azure.download_pdf(doc_name)
                        if pdf_bytes:
                            # Process PDF from bytes
                            chunks = self._process_pdf_bytes(pdf_bytes, doc_name)
                            if chunks:
                                added = self.vector_store.add_documents(chunks)
                                results['processed'] += 1
                                results['chunks'] += added
                    except Exception as e:
                        results['errors'].append(f"{doc_name}: {str(e)}")
                return results

        # Fall back to local folder
        if folder_path is None:
            folder_path = str(LOCAL_DOCUMENTS_DIR)

        for pdf_file in Path(folder_path).glob("*.pdf"):
            try:
                chunks = self.doc_processor.extract_text_from_pdf(str(pdf_file))
                if chunks:
                    added = self.vector_store.add_documents(chunks)
                    results['processed'] += 1
                    results['chunks'] += added
            except Exception as e:
                results['errors'].append(f"{pdf_file.name}: {str(e)}")

        return results

    def _process_pdf_bytes(self, pdf_bytes: bytes, source_name: str) -> List[Dict]:
        """Process PDF from bytes (downloaded from Azure)."""
        if not PDF_AVAILABLE:
            return []

        chunks = []
        try:
            reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))

            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text:
                    page_chunks = self.doc_processor._chunk_text(text, page_num, source_name)
                    chunks.extend(page_chunks)
        except Exception as e:
            print(f"Error processing {source_name}: {e}")

        return chunks

    def upload_document(self, local_path: str) -> Dict:
        """Upload a local PDF to Azure Blob and index it."""
        result = {'success': False, 'message': '', 'chunks': 0}

        if not os.path.exists(local_path):
            result['message'] = f"File not found: {local_path}"
            return result

        blob_name = os.path.basename(local_path)

        # Upload to Azure
        if self.vector_store.azure.upload_pdf(local_path, blob_name):
            # Now index it
            chunks = self.doc_processor.extract_text_from_pdf(local_path)
            if chunks:
                added = self.vector_store.add_documents(chunks)
                result['success'] = True
                result['message'] = f"Uploaded and indexed {blob_name}"
                result['chunks'] = added
            else:
                result['success'] = True
                result['message'] = f"Uploaded {blob_name} but no text extracted"
        else:
            result['message'] = "Failed to upload to Azure Blob"

        return result

    def search_documents(self, query: str, n_results: int = 5) -> str:
        """Search PDF documents and return formatted context."""
        results = self.vector_store.search(query, n_results)

        if not results:
            return ""

        context_parts = ["DOCUMENT SEARCH RESULTS:"]
        for r in results:
            context_parts.append(f"\n[{r['source']}, Page {r['page']}]")
            context_parts.append(r['text'][:500])  # Limit chunk size

        return "\n".join(context_parts)

    def search_data(self, query: str) -> str:
        """Search parquet data and return formatted context."""
        if not self.data_connector_available:
            return ""

        context_parts = []
        query_lower = query.lower()

        # Search for athlete data
        try:
            df = self.get_ksa_rankings()
            if df is not None and not df.empty:
                athlete_col = next((c for c in ['competitor', 'Competitor', 'full_name'] if c in df.columns), None)
                event_col = next((c for c in ['event', 'Event'] if c in df.columns), None)
                result_col = next((c for c in ['result', 'Result'] if c in df.columns), None)

                # Event-specific search
                event_keywords = ['100m', '200m', '400m', '800m', '1500m', 'jump', 'throw',
                                'shot', 'discus', 'javelin', 'relay', 'hurdles']

                for kw in event_keywords:
                    if kw in query_lower and event_col:
                        event_data = df[df[event_col].str.lower().str.contains(kw, na=False)]
                        if not event_data.empty and athlete_col and result_col:
                            context_parts.append(f"\nKSA {kw.upper()} ATHLETES:")
                            for _, row in event_data.head(10).iterrows():
                                context_parts.append(f"- {row[athlete_col]}: {row[result_col]}")
                            break
        except Exception:
            pass

        # Search benchmarks
        try:
            benchmarks = self.get_benchmarks_data()
            if benchmarks is not None and not benchmarks.empty:
                if any(term in query_lower for term in ['qualify', 'standard', 'medal', 'entry']):
                    event_col = next((c for c in ['Event', 'event'] if c in benchmarks.columns), None)
                    if event_col:
                        context_parts.append("\nQUALIFICATION STANDARDS:")
                        for _, row in benchmarks.head(10).iterrows():
                            row_info = [f"{row[event_col]}"]
                            for col in ['Gold Standard', 'Silver Standard', 'Bronze Standard']:
                                if col in row.index and pd.notna(row[col]):
                                    row_info.append(f"{col}: {row[col]}")
                            context_parts.append("- " + ", ".join(row_info))
        except Exception:
            pass

        return "\n".join(context_parts) if context_parts else ""

    def search(self, query: str) -> Tuple[str, List[str]]:
        """
        Combined search across documents and data.

        Returns:
            Tuple of (context_string, list_of_sources)
        """
        sources = []
        context_parts = []

        # Search documents
        doc_context = self.search_documents(query)
        if doc_context:
            context_parts.append(doc_context)
            sources.append("PDF Documents")

        # Search data
        data_context = self.search_data(query)
        if data_context:
            context_parts.append(data_context)
            sources.append("Azure Blob Data")

        return "\n\n".join(context_parts), sources

    def get_status(self) -> Dict:
        """Get RAG system status."""
        # Count documents in Azure and local
        azure_docs = self.vector_store.azure.list_documents() if self.vector_store.azure.is_available() else []
        local_docs = list(LOCAL_DOCUMENTS_DIR.glob("*.pdf"))

        return {
            'embeddings_available': EMBEDDINGS_AVAILABLE,
            'azure_available': AZURE_AVAILABLE and self.vector_store.azure.is_available(),
            'pdf_available': PDF_AVAILABLE,
            'data_connector': self.data_connector_available,
            'data_mode': self.get_data_mode() if self.data_connector_available else 'unavailable',
            'vector_store': self.vector_store.get_stats(),
            'azure_documents': azure_docs,
            'local_documents': [f.name for f in local_docs],
            'total_documents': len(azure_docs) + len(local_docs)
        }


# Convenience functions
_rag_instance = None

def get_rag() -> AthleticsRAG:
    """Get singleton RAG instance."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = AthleticsRAG()
    return _rag_instance


def search(query: str) -> Tuple[str, List[str]]:
    """Quick search function."""
    return get_rag().search(query)


def index_documents() -> Dict:
    """Index all documents in the documents folder."""
    return get_rag().index_documents()


def get_status() -> Dict:
    """Get RAG system status."""
    return get_rag().get_status()


# Import pandas for data operations
try:
    import pandas as pd
except ImportError:
    pd = None


if __name__ == "__main__":
    # Test the RAG system
    print("Athletics RAG System Status:")
    print("-" * 40)

    status = get_status()
    for key, value in status.items():
        print(f"{key}: {value}")

    print("\n" + "-" * 40)
    print("Testing search...")

    context, sources = search("What is the 100m qualification standard?")
    print(f"Sources: {sources}")
    print(f"Context length: {len(context)} chars")
