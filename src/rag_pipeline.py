"""
Main RAG Pipeline combining all components.
"""

from dataclasses import dataclass
from typing import Optional

from .chunking import RecipeChunk, load_recipes
from .retriever import RecipeRetriever, RetrievalResult
from .reranker import RecipeReranker
from .metadata import MetadataFilterer, MetadataFilter
from .llm import LLMClient


@dataclass
class RAGResponse:
    """Complete RAG response with answer and sources."""
    answer: str
    sources: list[RetrievalResult]
    filters_applied: Optional[MetadataFilter] = None
    retrieval_method: str = ""


class RecipeRAGPipeline:
    """
    Complete RAG pipeline for recipe question answering.
    """
    
    def __init__(
        self,
        csv_path: str,
        max_recipes: int = 5000,
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            csv_path: Path to recipes CSV file
            max_recipes: Maximum number of recipes to load
            embedding_model: Model for semantic embeddings
            reranker_model: Model for cross-encoder reranking
        """
        # Load recipes
        print("=" * 50)
        print("Initializing Recipe RAG Pipeline")
        print("=" * 50)
        
        self.chunks = load_recipes(csv_path, max_recipes)
        
        # Initialize retriever
        self.retriever = RecipeRetriever(self.chunks, embedding_model)
        
        # Initialize reranker
        self.reranker = RecipeReranker(reranker_model)
        
        # Initialize metadata filterer
        self.metadata_filterer = MetadataFilterer(self.chunks, use_llm=False)
        
        # LLM client (set later with API key)
        self.llm_client: Optional[LLMClient] = None
        
        print("=" * 50)
        print("Pipeline ready!")
        print("=" * 50)
    
    def set_llm(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None
    ):
        """
        Set the LLM client.
        
        Args:
            model: Model identifier
            api_key: API key
            api_base: Optional API base URL
        """
        self.llm_client = LLMClient(model, api_key, api_base)
        self.metadata_filterer.set_llm_client(self.llm_client)
    
    def query(
        self,
        question: str,
        use_bm25: bool = True,
        use_semantic: bool = True,
        use_reranker: bool = True,
        use_metadata_filter: bool = False,
        top_k_retrieval: int = 10,
        top_k_final: int = 5,
        bm25_weight: float = 0.5
    ) -> RAGResponse:
        """
        Process a question through the RAG pipeline.
        
        Args:
            question: User's question
            use_bm25: Enable BM25 retrieval
            use_semantic: Enable semantic retrieval
            use_reranker: Enable cross-encoder reranking
            use_metadata_filter: Enable metadata filtering
            top_k_retrieval: Number of results from retrieval
            top_k_final: Number of final results after reranking
            bm25_weight: Weight for BM25 in hybrid search
        
        Returns:
            RAGResponse with answer and sources
        """
        # Step 1: Metadata filtering (optional)
        chunk_ids = None
        filters = None
        
        if use_metadata_filter:
            chunk_ids, filters = self.metadata_filterer.extract_and_apply(question)
            if not chunk_ids:
                # No recipes match filters
                return RAGResponse(
                    answer="No recipes found matching your filters. Try broadening your search.",
                    sources=[],
                    filters_applied=filters,
                    retrieval_method="none (filtered out)"
                )
        
        # Step 2: Retrieval
        if not use_bm25 and not use_semantic:
            return RAGResponse(
                answer="Please enable at least one retrieval method (BM25 or Semantic).",
                sources=[],
                retrieval_method="none"
            )
        
        results = self.retriever.search(
            query=question,
            top_k=top_k_retrieval,
            use_bm25=use_bm25,
            use_semantic=use_semantic,
            bm25_weight=bm25_weight,
            chunk_ids=chunk_ids
        )
        
        if not results:
            return RAGResponse(
                answer="No relevant recipes found for your query.",
                sources=[],
                filters_applied=filters,
                retrieval_method="bm25" if use_bm25 and not use_semantic else 
                               "semantic" if use_semantic and not use_bm25 else "hybrid"
            )
        
        # Determine retrieval method string
        if use_bm25 and use_semantic:
            retrieval_method = "hybrid"
        elif use_bm25:
            retrieval_method = "bm25"
        else:
            retrieval_method = "semantic"
        
        # Step 3: Reranking (optional)
        if use_reranker and len(results) > 1:
            results = self.reranker.rerank(question, results, top_k=top_k_final)
            retrieval_method += "+rerank"
        else:
            results = results[:top_k_final]
        
        # Step 4: Generate answer with LLM
        if self.llm_client is None:
            # No LLM configured, just return sources
            sources_text = "\n\n".join([
                f"**[{i+1}] {r.chunk.title}**\nScore: {r.score:.3f}"
                for i, r in enumerate(results)
            ])
            return RAGResponse(
                answer=f"LLM not configured. Here are the relevant recipes:\n\n{sources_text}",
                sources=results,
                filters_applied=filters,
                retrieval_method=retrieval_method
            )
        
        # Prepare context for LLM
        context_chunks = [
            (r.chunk.id, r.chunk.title, r.chunk.text)
            for r in results
        ]
        
        try:
            answer = self.llm_client.generate_with_context(question, context_chunks)
        except Exception as e:
            answer = f"Error generating answer: {str(e)}\n\nPlease check your API key and model selection."
        
        return RAGResponse(
            answer=answer,
            sources=results,
            filters_applied=filters,
            retrieval_method=retrieval_method
        )
    
    def get_chunk_by_id(self, chunk_id: int) -> Optional[RecipeChunk]:
        """Get a chunk by its ID."""
        if 0 <= chunk_id < len(self.chunks):
            return self.chunks[chunk_id]
        return None
    
    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        categories = {}
        for chunk in self.chunks:
            cat = chunk.category or "unknown"
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "total_recipes": len(self.chunks),
            "categories": categories
        }

