import pandas as pd
import numpy as np
from tqdm import tqdm
import time
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("SentenceTransformers not available, using TF-IDF fallback")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models

    QDRANT_AVAILABLE = True
except ImportError:
    print("Qdrant not available, using in-memory fallback")
    QDRANT_AVAILABLE = False

# CrewAI imports
try:
    from crewai import Agent, Task, Crew
    from crewai.tools import BaseTool, tool

    CREWAI_AVAILABLE = True
except ImportError:
    print("CrewAI not available - using fallback implementation")
    CREWAI_AVAILABLE = False

import uuid
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class TravelRAGIndexer:
    def __init__(self, collection_name="travel_reviews"):
        self.collection_name = collection_name
        self.documents = []  # Always initialize documents list

        if QDRANT_AVAILABLE:
            # Initialize Qdrant client
            self.client = QdrantClient(":memory:")
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embedding_size = 384
            else:
                self.model = TfidfVectorizer(max_features=1000, stop_words='english')
                self.embedding_size = 1000
                self.tfidf_matrix = None
            self.setup_collection()
        else:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
            else:
                self.model = TfidfVectorizer(max_features=1000, stop_words='english')
                self.tfidf_matrix = None

    def setup_collection(self):
        """Setup Qdrant collection"""
        if not QDRANT_AVAILABLE:
            return

        try:
            # Try to delete existing collection first
            try:
                self.client.delete_collection(collection_name=self.collection_name)
            except:
                pass

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_size,
                    distance=models.Distance.COSINE
                )
            )
            print(f"Collection '{self.collection_name}' created successfully")
        except Exception as e:
            print(f"Collection setup error: {e}")

    def index_documents(self, csv_path="processed_chunks.csv", limit_rows=None):
        """Index documents from CSV file with improved batching"""
        print("Loading processed chunks...")

        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"Error: {csv_path} not found. Run data_ingest.ipynb first!")
            return 0

        if limit_rows:
            df = df.head(limit_rows)
            print(f"Limited to {limit_rows} chunks for faster testing")

        print(f"Processing {len(df)} chunks...")
        texts = df['text'].fillna('').astype(str).tolist()

        if QDRANT_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE:
            # Use Qdrant with SentenceTransformers with improved batching
            print("Generating embeddings with SentenceTransformers...")

            # Process in smaller batches with progress
            batch_size = 32
            points = []
            start_time = time.time()

            for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False, device='cpu')

                for j, (_, row) in enumerate(df.iloc[i:i + batch_size].iterrows()):
                    rating = self._safe_rating(row['rating'])

                    point = models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=batch_embeddings[j].tolist(),
                        payload={
                            "text": str(row['text']),
                            "rating": rating,
                            "review_id": int(row['review_id']) if pd.notna(row['review_id']) else i + j,
                            "chunk_id": int(row['chunk_id']) if pd.notna(row['chunk_id']) else 0,
                            "source": str(row['source']),
                            "category": "hotel_review"
                        }
                    )
                    points.append(point)

            # Upload to Qdrant in batches
            upload_batch_size = 100
            print(f"\nUploading {len(points)} points to Qdrant...")

            for i in tqdm(range(0, len(points), upload_batch_size), desc="Uploading batches"):
                batch = points[i:i + upload_batch_size]
                self.client.upsert(collection_name=self.collection_name, points=batch)

            elapsed = time.time() - start_time
            print(f"Indexing completed in {elapsed:.2f} seconds")

        elif SENTENCE_TRANSFORMERS_AVAILABLE:
            # Use SentenceTransformers with in-memory storage
            print("Using SentenceTransformers with in-memory storage...")
            embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=16, device='cpu')

            self.documents = []  # Reset documents
            for idx, (_, row) in enumerate(df.iterrows()):
                rating = self._safe_rating(row['rating'])

                self.documents.append({
                    "text": str(row['text']),
                    "rating": rating,
                    "review_id": int(row['review_id']) if pd.notna(row['review_id']) else idx,
                    "chunk_id": int(row['chunk_id']) if pd.notna(row['chunk_id']) else 0,
                    "source": str(row['source']),
                    "category": "hotel_review",
                    "embedding": embeddings[idx]
                })
        else:
            # Use TF-IDF fallback
            print("Using TF-IDF fallback...")
            self.tfidf_matrix = self.model.fit_transform(texts)

            self.documents = []  # Reset documents
            for idx, (_, row) in enumerate(df.iterrows()):
                rating = self._safe_rating(row['rating'])

                self.documents.append({
                    "text": str(row['text']),
                    "rating": rating,
                    "review_id": int(row['review_id']) if pd.notna(row['review_id']) else idx,
                    "chunk_id": int(row['chunk_id']) if pd.notna(row['chunk_id']) else 0,
                    "source": str(row['source']),
                    "category": "hotel_review",
                    "idx": idx
                })

        print(f"Indexed {len(df)} document chunks successfully!")
        return len(df)

    def _safe_rating(self, rating):
        """Safely convert rating to integer"""
        try:
            return int(float(rating)) if pd.notna(rating) else 3
        except (ValueError, TypeError):
            return 3

    def retrieve(self, query, top_k=5, rating_filter=None):
        """Retrieve relevant documents based on query"""
        if not query or query.strip() == "":
            return []

        if QDRANT_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE:
            # Use Qdrant search with updated API
            query_embedding = self.model.encode([query])

            search_filter = None
            if rating_filter:
                search_filter = models.Filter(
                    must=[models.FieldCondition(key="rating", match=models.MatchValue(value=rating_filter))]
                )

            # Use query_points instead of deprecated search method
            try:
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding[0].tolist(),
                    limit=top_k,
                    query_filter=search_filter
                )
                points = results.points
            except AttributeError:
                # Fallback to old search method if query_points not available
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding[0].tolist(),
                    limit=top_k,
                    query_filter=search_filter
                )
                points = results

            return [
                {
                    "text": hit.payload["text"],
                    "rating": hit.payload["rating"],
                    "score": hit.score,
                    "review_id": hit.payload["review_id"],
                    "source": hit.payload.get("source", "unknown"),
                    "category": hit.payload.get("category", "hotel_review")
                }
                for hit in points
            ]

        elif SENTENCE_TRANSFORMERS_AVAILABLE:
            # Use SentenceTransformers with cosine similarity
            query_embedding = self.model.encode([query])[0]
            scores = []

            for doc in self.documents:
                if rating_filter and doc["rating"] != rating_filter:
                    continue
                score = np.dot(query_embedding, doc["embedding"]) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(doc["embedding"])
                )
                scores.append((score, doc))

            # Sort by score and return top_k
            scores.sort(key=lambda x: x[0], reverse=True)

            return [
                {
                    "text": doc["text"],
                    "rating": doc["rating"],
                    "score": score,
                    "review_id": doc["review_id"],
                    "source": doc.get("source", "unknown"),
                    "category": doc.get("category", "hotel_review")
                }
                for score, doc in scores[:top_k]
            ]

        else:
            # Use TF-IDF fallback
            if self.tfidf_matrix is None:
                return []

            query_vec = self.model.transform([query])
            scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

            # Get top indices
            top_indices = scores.argsort()[-top_k:][::-1]

            results = []
            for idx in top_indices:
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    if rating_filter and doc["rating"] != rating_filter:
                        continue
                    results.append({
                        "text": doc["text"],
                        "rating": doc["rating"],
                        "score": scores[idx],
                        "review_id": doc["review_id"],
                        "source": doc.get("source", "unknown"),
                        "category": doc.get("category", "hotel_review")
                    })

            return results[:top_k]

    def get_stats(self):
        """Get indexer statistics"""
        if QDRANT_AVAILABLE:
            try:
                info = self.client.get_collection(self.collection_name)
                return {
                    "total_documents": info.points_count if hasattr(info, 'points_count') else len(self.documents),
                    "embedding_method": "SentenceTransformers" if SENTENCE_TRANSFORMERS_AVAILABLE else "TF-IDF",
                    "storage": "Qdrant",
                    "collection_name": self.collection_name
                }
            except:
                pass

        return {
            "total_documents": len(self.documents),
            "embedding_method": "SentenceTransformers" if SENTENCE_TRANSFORMERS_AVAILABLE else "TF-IDF",
            "storage": "In-Memory",
            "collection_name": self.collection_name
        }


# CrewAI Tools for RAG retrieval
if CREWAI_AVAILABLE:
    # Method 1: Class-based tool (Fixed version)
    class RAGRetrievalTool(BaseTool):
        name: str = "rag_retrieval"
        description: str = "Retrieve relevant hotel reviews based on user query using semantic search"

        # Define indexer as a class variable that will be set externally
        _indexer: Optional[TravelRAGIndexer] = None

        @classmethod
        def set_indexer(cls, indexer: TravelRAGIndexer):
            """Set the indexer instance for the tool"""
            cls._indexer = indexer

        def _run(self, query: str, top_k: int = 5, rating_filter: Optional[int] = None) -> str:
            """Execute the RAG retrieval"""
            if self._indexer is None:
                return "Error: RAG indexer not initialized. Call RAGRetrievalTool.set_indexer() first."

            try:
                results = self._indexer.retrieve(query, top_k=top_k, rating_filter=rating_filter)

                if not results:
                    return f"No relevant reviews found for query: {query}"

                # Format results for agent consumption
                formatted_results = []
                for i, result in enumerate(results, 1):
                    formatted_results.append(
                        f"Review {i} (Rating: {result['rating']}/5, Relevance: {result['score']:.3f}):\n"
                        f"{result['text']}\n"
                    )

                return "\n".join(formatted_results)

            except Exception as e:
                return f"Error during retrieval: {str(e)}"


    # Method 2: Function-based tool (Fixed version)
    def create_rag_tool_function(indexer: TravelRAGIndexer):
        """Create a function-based RAG retrieval tool that returns a callable function"""

        def rag_retrieval_function(query: str, top_k: int = 5, rating_filter: Optional[int] = None) -> str:
            """Retrieve relevant hotel reviews based on user query using semantic search

            Args:
                query: Search query for hotel reviews
                top_k: Number of top results to return (default: 5)
                rating_filter: Filter by specific rating (1-5 stars, optional)

            Returns:
                Formatted string with relevant hotel reviews
            """
            try:
                results = indexer.retrieve(query, top_k=top_k, rating_filter=rating_filter)

                if not results:
                    return f"No relevant reviews found for query: {query}"

                # Format results for agent consumption
                formatted_results = []
                for i, result in enumerate(results, 1):
                    formatted_results.append(
                        f"Review {i} (Rating: {result['rating']}/5, Relevance: {result['score']:.3f}):\n"
                        f"{result['text']}\n"
                    )

                return "\n".join(formatted_results)

            except Exception as e:
                return f"Error during retrieval: {str(e)}"

        return rag_retrieval_function


    # Method 3: Proper CrewAI tool decorator (Alternative approach)
    def create_crewai_rag_tool(indexer: TravelRAGIndexer):
        """Create a proper CrewAI tool using the tool decorator"""

        @tool("rag_retrieval_tool")
        def rag_retrieval_tool(query: str) -> str:
            """Retrieve relevant hotel reviews based on user query using semantic search.

            Args:
                query (str): Search query for hotel reviews

            Returns:
                str: Formatted string with relevant hotel reviews
            """
            try:
                results = indexer.retrieve(query, top_k=5)

                if not results:
                    return f"No relevant reviews found for query: {query}"

                # Format results for agent consumption
                formatted_results = []
                for i, result in enumerate(results, 1):
                    formatted_results.append(
                        f"Review {i} (Rating: {result['rating']}/5, Relevance: {result['score']:.3f}):\n"
                        f"{result['text']}\n"
                    )

                return "\n".join(formatted_results)

            except Exception as e:
                return f"Error during retrieval: {str(e)}"

        return rag_retrieval_tool

if __name__ == "__main__":
    # Initialize indexer
    print("Initializing TravelRAGIndexer...")
    indexer = TravelRAGIndexer()

    try:
        # Index documents (limit for testing)
        num_indexed = indexer.index_documents()  # Limit for faster testing
        print(f"\nIndexing complete! {num_indexed} chunks indexed.")

        # Print stats
        stats = indexer.get_stats()
        print(f"Stats: {stats}")

        # Test retrieval
        print("\n=== Testing Retrieval ===")
        test_queries = [
            "expensive hotel with good location",
            "comfortable bed and clean room",
            "noisy hotel with parking issues",
            "great service and friendly staff"
        ]

        for query in test_queries:
            print(f"\nQuery: '{query}'")
            results = indexer.retrieve(query, top_k=3)

            if results:
                for i, result in enumerate(results, 1):
                    print(f"{i}. Score: {result['score']:.3f} | Rating: {result['rating']}/5")
                    print(f"   Text: {result['text'][:100]}...")
                    print()
            else:
                print("   No results found.")

        # Test rating filter
        print("\n=== Testing Rating Filter (5-star reviews only) ===")
        results = indexer.retrieve("great hotel experience", top_k=3, rating_filter=5)
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result['score']:.3f} | Rating: {result['rating']}/5")
            print(f"   Text: {result['text'][:100]}...")
            print()

        # Test CrewAI tools if available
        if CREWAI_AVAILABLE:
            print("\n=== Testing CrewAI RAG Tools ===")

            # Test Method 1: Class-based tool
            print("\n--- Testing Class-based Tool ---")
            RAGRetrievalTool.set_indexer(indexer)
            rag_tool_class = RAGRetrievalTool()
            class_result = rag_tool_class._run("comfortable hotel with good amenities", top_k=2)
            print("Class-based tool result:")
            print(class_result)

            # Test Method 2: Function-based approach (Fixed)
            print("\n--- Testing Function-based Approach ---")
            rag_function = create_rag_tool_function(indexer)
            func_result = rag_function("luxury hotel with spa facilities", top_k=2)
            print("Function-based approach result:")
            print(func_result)

            # Test Method 3: CrewAI tool decorator (Fixed)
            print("\n--- Testing CrewAI Tool Decorator ---")
            rag_crewai_tool = create_crewai_rag_tool(indexer)
            # To use the tool, you would normally pass it to an agent
            # For testing purposes, we can access the underlying function
            if hasattr(rag_crewai_tool, 'func'):
                crewai_result = rag_crewai_tool.func("boutique hotel with unique design")
            else:
                # Alternative way to test
                crewai_result = "CrewAI tool created successfully - would be used with agents"
            print("CrewAI tool result:")
            print(crewai_result)

            # Test with rating filter using function approach
            print("\n--- Testing with Rating Filter ---")
            filtered_result = rag_function("great service", top_k=3, rating_filter=5)
            print("Filtered results (5-star only):")
            print(filtered_result)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()