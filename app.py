import os
import logging
import gradio as gr
from typing import Optional

# Suppress verbose logging from libraries
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Global pipeline instance
pipeline = None


def initialize_pipeline(csv_path: str, max_recipes: int = 5000):
    """Initialize the RAG pipeline."""
    global pipeline
    
    from src.rag_pipeline import RecipeRAGPipeline
    
    pipeline = RecipeRAGPipeline(
        csv_path=csv_path,
        max_recipes=max_recipes
    )
    return pipeline


def get_provider_models(provider: str) -> list[str]:
    """Get available models for a provider."""
    models = {
        "Groq": [
            "groq/llama-3.1-8b-instant",
            "groq/llama-3.3-70b-versatile",
            "groq/mixtral-8x7b-32768",
        ],
        "OpenRouter": [
            "openrouter/google/gemma-2-9b-it:free",
            "openrouter/meta-llama/llama-3.2-3b-instruct:free",
            "openrouter/meta-llama/llama-3.3-70b-instruct:free",
        ],
        "OpenAI": [
            "gpt-4o-mini",
            "gpt-4o",
        ],
        "Ollama (Local)": [
            "ollama/llama3.2",
            "ollama/mistral",
            "ollama/gemma2",
        ],
    }
    return models.get(provider, [])


def update_models(provider: str):
    """Update model dropdown based on provider selection."""
    models = get_provider_models(provider)
    return gr.Dropdown(choices=models, value=models[0] if models else None, allow_custom_value=True)


def process_query(
    question: str,
    api_key: str,
    provider: str,
    model: str,
    use_bm25: bool,
    use_semantic: bool,
    use_reranker: bool,
    use_metadata_filter: bool,
    top_k_retrieve: int,
    top_k_final: int
) -> tuple[str, str]:
    """Process user query through RAG pipeline."""
    global pipeline
    
    if pipeline is None:
        return "⚠️ Pipeline not initialized. Please wait...", ""
    
    if not question.strip():
        return "Please enter a question.", ""
    
    if not api_key.strip() and provider != "Ollama (Local)":
        return "⚠️ Please enter your API key.", ""
    
    # Validate model matches provider, use default if not
    provider_models = get_provider_models(provider)
    if model not in provider_models and provider_models:
        model = provider_models[0]
    
    # Configure LLM
    api_base = None
    if model.startswith("ollama/"):
        api_base = "http://localhost:11434"
    
    try:
        pipeline.set_llm(model=model, api_key=api_key if api_key.strip() else None, api_base=api_base)
    except Exception as e:
        return f"⚠️ Error configuring LLM: {str(e)}", ""
    
    # Run query
    try:
        response = pipeline.query(
            question=question,
            use_bm25=use_bm25,
            use_semantic=use_semantic,
            use_reranker=use_reranker,
            use_metadata_filter=use_metadata_filter,
            top_k_retrieval=top_k_retrieve,
            top_k_final=top_k_final
        )
    except Exception as e:
        return f"⚠️ Error processing query: {str(e)}", ""
    
    # Format sources - show full chunk content
    sources_parts = []
    for i, result in enumerate(response.sources, 1):
        chunk = result.chunk
        
        ingredients_list = "\n".join(f"  - {ing}" for ing in chunk.ingredients[:10])
        if len(chunk.ingredients) > 10:
            ingredients_list += "\n  - ..."
        
        directions_list = "\n".join(f"  {j}. {step}" for j, step in enumerate(chunk.directions[:5], 1))
        if len(chunk.directions) > 5:
            directions_list += "\n  ..."
        
        sources_parts.append(
            f"### [{i}] {chunk.title}\n"
            f"**Score:** {result.score:.3f} | **Method:** {result.method} | **Category:** {chunk.category or 'N/A'}\n\n"
            f"**Ingredients:**\n{ingredients_list}\n\n"
            f"**Directions:**\n{directions_list}"
        )
    
    sources_display = "\n\n---\n\n".join(sources_parts) if sources_parts else "No sources found."
    
    # Add retrieval info
    info_line = f"*Retrieved {len(response.sources)} recipes using {response.retrieval_method}*\n\n"
    
    if response.filters_applied and not response.filters_applied.is_empty():
        filter_parts = []
        if response.filters_applied.category:
            filter_parts.append(f"Category: {response.filters_applied.category}")
        if response.filters_applied.ingredients_include:
            filter_parts.append(f"Include: {', '.join(response.filters_applied.ingredients_include)}")
        if response.filters_applied.ingredients_exclude:
            filter_parts.append(f"Exclude: {', '.join(response.filters_applied.ingredients_exclude)}")
        info_line += f"**Filters:** {' | '.join(filter_parts)}\n\n"
    
    return response.answer, info_line + sources_display


def create_ui():
    """Create the Gradio interface with orange/dark theme."""
    
    # Orange-Black theme
    theme = gr.themes.Base(
        primary_hue="orange",
        secondary_hue="amber",
        neutral_hue="stone",
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="#1a1a1a",
        body_background_fill_dark="#1a1a1a",
        body_text_color="#e5e5e5",
        body_text_color_dark="#e5e5e5",
        block_background_fill="#262626",
        block_background_fill_dark="#262626",
        block_border_width="1px",
        block_border_color="#404040",
        block_border_color_dark="#404040",
        block_title_text_color="#f5f5f5",
        block_label_text_color="#d4d4d4",
        input_background_fill="#333333",
        input_background_fill_dark="#333333",
        input_border_color="#525252",
        input_border_color_dark="#525252",
        button_primary_background_fill="#ea580c",
        button_primary_background_fill_hover="#f97316",
        button_primary_text_color="white",
        button_secondary_background_fill="#404040",
        button_secondary_background_fill_hover="#525252",
        button_secondary_text_color="#e5e5e5",
        shadow_drop="none",
        shadow_spread="0px",
    )
    
    with gr.Blocks(
        title="Recipe RAG QA",
        theme=theme,
        css="""
        .gradio-container { max-width: 1400px !important; }
        .header-banner { 
            background: linear-gradient(135deg, #ea580c 0%, #f97316 50%, #fb923c 100%);
            padding: 28px;
            border-radius: 16px;
            margin-bottom: 24px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(234, 88, 12, 0.3);
        }
        .header-banner h1 { 
            color: white; 
            margin: 0; 
            font-size: 2.2rem; 
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .header-banner p { 
            color: rgba(255,255,255,0.95); 
            margin: 10px 0 0 0; 
            font-size: 1.1rem;
        }
        .section-title { 
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
            color: #f5f5f5;
        }
        .param-label {
            color: #fb923c !important;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .dark { background: #1a1a1a !important; }
        footer { display: none !important; }
        .prose { color: #d4d4d4 !important; }
        .prose h3 { color: #fb923c !important; }
        .prose strong { color: #f5f5f5 !important; }
        .prose a { color: #fb923c !important; }
        """
    ) as demo:
        
        # Header
        gr.HTML("""
        <div class="header-banner">
            <h1>🍳 Recipe RAG Question Answering</h1>
            <p>Ask questions about recipes using AI-powered retrieval and generation</p>
        </div>
        """)
        
        with gr.Row():
            # Left Panel - Configuration
            with gr.Column(scale=1):
                gr.HTML('<div class="section-title">⚙️ Configuration</div>')
                
                # API Settings
                with gr.Group():
                    gr.HTML('<div class="param-label">🔑 API Settings</div>')
                    
                    api_key = gr.Textbox(
                        label="API Key",
                        placeholder="Enter your API key here...",
                        type="password",
                        info="Your API key is not stored and only used for this session"
                    )
                    
                    provider = gr.Dropdown(
                        label="LLM Provider",
                        choices=["Groq", "OpenRouter", "OpenAI", "Ollama (Local)"],
                        value="OpenRouter",
                        info="Select your LLM provider"
                    )
                    
                    model = gr.Dropdown(
                        label="Model",
                        choices=get_provider_models("OpenRouter"),
                        value="openrouter/meta-llama/llama-3.3-70b-instruct:free",
                        info="Select the model to use",
                        allow_custom_value=True
                    )
                
                # Retrieval Parameters
                with gr.Group():
                    gr.HTML('<div class="param-label">📊 Retrieval Parameters</div>')
                    
                    top_k_retrieve = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=10,
                        step=1,
                        label="Top-K chunks to retrieve",
                        info="Initial retrieval before reranking"
                    )
                    
                    top_k_final = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="Top-K chunks after reranking",
                        info="Final chunks to use for answer generation"
                    )
                
                # Retrieval Methods
                with gr.Group():
                    gr.HTML('<div class="param-label">🔍 Retrieval Methods</div>')
                    
                    use_bm25 = gr.Checkbox(
                        label="BM25 (Keyword Search)",
                        value=True,
                        info="Traditional keyword matching"
                    )
                    use_semantic = gr.Checkbox(
                        label="Semantic Search",
                        value=True,
                        info="Meaning-based embeddings"
                    )
                    use_reranker = gr.Checkbox(
                        label="Cross-Encoder Reranker",
                        value=True,
                        info="Improve ranking quality"
                    )
                    use_metadata_filter = gr.Checkbox(
                        label="Metadata Filtering",
                        value=False,
                        info="Filter by category/ingredients"
                    )
                
            
            # Right Panel - Query and Results
            with gr.Column(scale=2):
                gr.HTML('<div class="section-title">💬 Ask a Question</div>')
                
                question = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., How do I make chocolate chip cookies? What's a quick dinner with chicken?",
                    lines=2,
                    max_lines=4
                )
                
                with gr.Row():
                    submit_btn = gr.Button("🔍 Search & Answer", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)
                
                gr.HTML('<div class="section-title">📝 Answer</div>')
                answer = gr.Markdown(
                    value="*Enter a question and click Search to get started...*",
                    elem_classes=["answer-box"]
                )
                
                gr.HTML('<div class="section-title">📚 Retrieved Sources</div>')
                sources = gr.Markdown(
                    value="*Sources will appear here after searching...*",
                    elem_classes=["sources-box"]
                )
        
        # Examples
        with gr.Accordion("💡 Example Queries", open=False):
            gr.Examples(
                examples=[
                    ["How do I make chocolate chip cookies?"],
                    ["What's a good recipe for chicken soup?"],
                    ["Quick and easy pasta dinner"],
                    ["Dessert recipe without nuts"],
                    ["Healthy breakfast ideas with eggs"],
                    ["How to make homemade bread?"],
                ],
                inputs=[question],
                label=""
            )
        
        # Tips
        with gr.Accordion("📖 Tips & Usage Guide", open=False):
            gr.Markdown("""
            ### Getting Started
            1. Enter your API key (get free key from [Groq](https://console.groq.com/) or [OpenRouter](https://openrouter.ai/))
            2. Select a provider and model
            3. Type your question and click Search
            
            ### When to use each retrieval method
            
            | Method | Best For | Example Query |
            |--------|----------|---------------|
            | **BM25** | Specific ingredients | "recipe with cream cheese" |
            | **Semantic** | Conceptual queries | "healthy dinner ideas" |
            | **Both (Hybrid)** | General questions | "how to make cookies" |
            | **Reranker** | Better ranking | Always recommended |
            | **Metadata Filter** | Category search | "soup recipe", "dessert without gluten" |
            
            ### Available Categories
            soup, salad, dessert, bread, main_course, appetizer, breakfast, drink, sauce, side_dish
            """)
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 20px; margin-top: 20px; border-top: 1px solid #404040; color: #a3a3a3;">
            Built for LLM Course Project @ Lviv Polytechnic | RAG Question Answering System
        </div>
        """)
        
        # Event handlers
        provider.change(
            fn=update_models,
            inputs=[provider],
            outputs=[model]
        )
        
        submit_btn.click(
            fn=process_query,
            inputs=[
                question, api_key, provider, model,
                use_bm25, use_semantic, use_reranker, use_metadata_filter,
                top_k_retrieve, top_k_final
            ],
            outputs=[answer, sources]
        )
        
        question.submit(
            fn=process_query,
            inputs=[
                question, api_key, provider, model,
                use_bm25, use_semantic, use_reranker, use_metadata_filter,
                top_k_retrieve, top_k_final
            ],
            outputs=[answer, sources]
        )
        
        clear_btn.click(
            fn=lambda: ("", "*Enter a question and click Search to get started...*", "*Sources will appear here after searching...*"),
            outputs=[question, answer, sources]
        )
        
    
    return demo


# Main entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Recipe RAG QA System")
    parser.add_argument(
        "--data",
        type=str,
        default="data/full_dataset.csv",
        help="Path to recipes CSV file"
    )
    parser.add_argument(
        "--max-recipes",
        type=int,
        default=5000,
        help="Maximum number of recipes to load"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link"
    )
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"Error: Data file not found at {args.data}")
        print("Please provide the correct path using --data argument")
        exit(1)
    
    # Initialize pipeline
    print("Initializing RAG pipeline...")
    initialize_pipeline(args.data, args.max_recipes)
    
    # Create and launch UI
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
