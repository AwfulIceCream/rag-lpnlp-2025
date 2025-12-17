from sentence_transformers import CrossEncoder
from .retriever import RetrievalResult


class RecipeReranker:
    """
    Cross-encoder reranker for improving retrieval results.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the reranker with a cross-encoder model.
        
        Args:
            model_name: HuggingFace model name for cross-encoder
        """
        print(f"Loading reranker model: {model_name}...")
        self.model = CrossEncoder(model_name)
        print("Reranker initialized!")
    
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5
    ) -> list[RetrievalResult]:
        """
        Rerank retrieval results using cross-encoder.
        
        Args:
            query: Original search query
            results: List of retrieval results to rerank
            top_k: Number of top results to return after reranking
        
        Returns:
            Reranked list of RetrievalResult objects
        """
        if not results:
            return []
        
        # Prepare query-document pairs
        pairs = [(query, result.chunk.text) for result in results]
        
        # Get cross-encoder scores
        scores = self.model.predict(pairs)
        
        # Create new results with updated scores
        reranked = []
        for result, score in zip(results, scores):
            reranked.append(RetrievalResult(
                chunk=result.chunk,
                score=float(score),
                method=f"{result.method}+rerank"
            ))
        
        # Sort by new scores and return top-k
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]

