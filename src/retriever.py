import numpy as np
from typing import Optional
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .chunking import RecipeChunk


@dataclass
class RetrievalResult:
    """Result from retrieval with score."""
    chunk: RecipeChunk
    score: float
    method: str  # "bm25", "semantic", or "hybrid"


class RecipeRetriever:
    """
    Hybrid retriever supporting BM25 and semantic search.
    """
    
    def __init__(
        self,
        chunks: list[RecipeChunk],
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    ):
        self.chunks = chunks
        self.texts = [chunk.text for chunk in chunks]
        
        # Initialize BM25
        print("Initializing BM25 index...")
        tokenized_texts = [text.lower().split() for text in self.texts]
        self.bm25 = BM25Okapi(tokenized_texts)
        
        # Initialize semantic search
        print(f"Loading embedding model: {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        print("Computing embeddings for all chunks...")
        self.embeddings = self.embedding_model.encode(
            self.texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        # Normalize embeddings for cosine similarity
        self.embeddings = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )
        print("Retriever initialized!")
    
    def search_bm25(
        self,
        query: str,
        top_k: int = 10,
        chunk_ids: Optional[list[int]] = None
    ) -> list[RetrievalResult]:
        """
        Search using BM25 keyword matching.
        
        Args:
            query: Search query
            top_k: Number of results to return
            chunk_ids: Optional list of chunk IDs to search within (for filtering)
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Apply chunk filtering if specified
        if chunk_ids is not None:
            mask = np.zeros(len(scores), dtype=bool)
            mask[chunk_ids] = True
            scores = np.where(mask, scores, -np.inf)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include positive scores
                results.append(RetrievalResult(
                    chunk=self.chunks[idx],
                    score=float(scores[idx]),
                    method="bm25"
                ))
        
        return results
    
    def search_semantic(
        self,
        query: str,
        top_k: int = 10,
        chunk_ids: Optional[list[int]] = None
    ) -> list[RetrievalResult]:
        """
        Search using semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            chunk_ids: Optional list of chunk IDs to search within (for filtering)
        """
        # Encode query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Compute cosine similarity
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        
        # Apply chunk filtering if specified
        if chunk_ids is not None:
            mask = np.zeros(len(similarities), dtype=bool)
            mask[chunk_ids] = True
            similarities = np.where(mask, similarities, -np.inf)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append(RetrievalResult(
                    chunk=self.chunks[idx],
                    score=float(similarities[idx]),
                    method="semantic"
                ))
        
        return results
    
    def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        bm25_weight: float = 0.5,
        chunk_ids: Optional[list[int]] = None
    ) -> list[RetrievalResult]:
        """
        Hybrid search combining BM25 and semantic scores.
        
        Args:
            query: Search query
            top_k: Number of results to return
            bm25_weight: Weight for BM25 scores (semantic weight = 1 - bm25_weight)
            chunk_ids: Optional list of chunk IDs to search within
        """
        # Get more results from each method for better fusion
        bm25_results = self.search_bm25(query, top_k=top_k * 2, chunk_ids=chunk_ids)
        semantic_results = self.search_semantic(query, top_k=top_k * 2, chunk_ids=chunk_ids)
        
        # Normalize scores within each result set
        def normalize_scores(results: list[RetrievalResult]) -> dict[int, float]:
            if not results:
                return {}
            scores = [r.score for r in results]
            min_score, max_score = min(scores), max(scores)
            range_score = max_score - min_score if max_score != min_score else 1
            return {
                r.chunk.id: (r.score - min_score) / range_score
                for r in results
            }
        
        bm25_scores = normalize_scores(bm25_results)
        semantic_scores = normalize_scores(semantic_results)
        
        # Combine scores
        all_chunk_ids = set(bm25_scores.keys()) | set(semantic_scores.keys())
        combined_scores = {}
        
        semantic_weight = 1 - bm25_weight
        for chunk_id in all_chunk_ids:
            bm25_score = bm25_scores.get(chunk_id, 0)
            semantic_score = semantic_scores.get(chunk_id, 0)
            combined_scores[chunk_id] = (
                bm25_weight * bm25_score + semantic_weight * semantic_score
            )
        
        # Sort and return top-k
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:top_k]
        
        return [
            RetrievalResult(
                chunk=self.chunks[chunk_id],
                score=combined_scores[chunk_id],
                method="hybrid"
            )
            for chunk_id in sorted_ids
        ]
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        use_bm25: bool = True,
        use_semantic: bool = True,
        bm25_weight: float = 0.5,
        chunk_ids: Optional[list[int]] = None
    ) -> list[RetrievalResult]:
        """
        Main search method with toggleable retrieval methods.
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_bm25: Enable BM25 search
            use_semantic: Enable semantic search
            bm25_weight: Weight for BM25 in hybrid mode
            chunk_ids: Optional list of chunk IDs to search within
        """
        if use_bm25 and use_semantic:
            return self.search_hybrid(query, top_k, bm25_weight, chunk_ids)
        elif use_bm25:
            return self.search_bm25(query, top_k, chunk_ids)
        elif use_semantic:
            return self.search_semantic(query, top_k, chunk_ids)
        else:
            # No search method enabled, return empty
            return []

