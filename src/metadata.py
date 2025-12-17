import json
import re
from typing import Optional
from dataclasses import dataclass

from .chunking import RecipeChunk


@dataclass
class MetadataFilter:
    """Represents extracted metadata filters from a query."""
    category: Optional[str] = None
    ingredients_include: Optional[list[str]] = None
    ingredients_exclude: Optional[list[str]] = None
    
    def is_empty(self) -> bool:
        """Check if no filters are set."""
        return (
            self.category is None and
            self.ingredients_include is None and
            self.ingredients_exclude is None
        )


def extract_filters_with_llm(query: str, llm_client) -> MetadataFilter:
    """
    Extract metadata filters from query using LLM.
    
    Args:
        query: User's search query
        llm_client: LLM client for filter extraction
    
    Returns:
        MetadataFilter object with extracted filters
    """
    prompt = f"""Analyze this recipe search query and extract any filters.

Query: "{query}"

Extract the following if mentioned:
1. category: One of [soup, salad, dessert, bread, main_course, appetizer, breakfast, drink, sauce, side_dish, other]
2. ingredients_include: List of ingredients that MUST be included
3. ingredients_exclude: List of ingredients that should NOT be included

Respond ONLY with a JSON object (no markdown, no explanation):
{{"category": null or "category_name", "ingredients_include": null or ["ing1", "ing2"], "ingredients_exclude": null or ["ing1"]}}

Examples:
- "chicken soup recipe" -> {{"category": "soup", "ingredients_include": ["chicken"], "ingredients_exclude": null}}
- "dessert without nuts" -> {{"category": "dessert", "ingredients_include": null, "ingredients_exclude": ["nuts"]}}
- "pasta with tomatoes" -> {{"category": null, "ingredients_include": ["pasta", "tomatoes"], "ingredients_exclude": null}}
"""
    
    try:
        response = llm_client.generate(prompt, max_tokens=200)
        
        # Try to parse JSON from response
        # Handle potential markdown code blocks
        json_match = re.search(r'\{[^{}]*\}', response)
        if json_match:
            data = json.loads(json_match.group())
            return MetadataFilter(
                category=data.get("category"),
                ingredients_include=data.get("ingredients_include"),
                ingredients_exclude=data.get("ingredients_exclude")
            )
    except (json.JSONDecodeError, Exception) as e:
        print(f"Failed to extract filters: {e}")
    
    return MetadataFilter()


def extract_filters_rule_based(query: str) -> MetadataFilter:
    """
    Extract metadata filters using rule-based approach (no LLM needed).
    
    Args:
        query: User's search query
    
    Returns:
        MetadataFilter object with extracted filters
    """
    query_lower = query.lower()
    
    # Category detection
    category = None
    category_keywords = {
        "soup": ["soup", "chowder", "stew", "broth"],
        "salad": ["salad", "slaw"],
        "dessert": ["dessert", "cake", "cookie", "pie", "sweet", "candy"],
        "bread": ["bread", "muffin", "biscuit"],
        "main_course": ["main course", "dinner", "lunch", "entree"],
        "appetizer": ["appetizer", "starter", "dip"],
        "breakfast": ["breakfast", "morning", "brunch"],
        "drink": ["drink", "beverage", "cocktail", "smoothie"],
        "sauce": ["sauce", "dressing", "gravy"],
        "side_dish": ["side dish", "side"],
    }
    
    for cat, keywords in category_keywords.items():
        if any(kw in query_lower for kw in keywords):
            category = cat
            break
    
    # Ingredient inclusion (basic detection)
    ingredients_include = []
    include_patterns = [
        r"with\s+(\w+)",
        r"using\s+(\w+)",
        r"(\w+)\s+recipe",
    ]
    
    for pattern in include_patterns:
        matches = re.findall(pattern, query_lower)
        ingredients_include.extend(matches)
    
    # Ingredient exclusion
    ingredients_exclude = []
    exclude_patterns = [
        r"without\s+(\w+)",
        r"no\s+(\w+)",
        r"exclude\s+(\w+)",
    ]
    
    for pattern in exclude_patterns:
        matches = re.findall(pattern, query_lower)
        ingredients_exclude.extend(matches)
    
    return MetadataFilter(
        category=category,
        ingredients_include=ingredients_include if ingredients_include else None,
        ingredients_exclude=ingredients_exclude if ingredients_exclude else None
    )


def apply_filters(
    chunks: list[RecipeChunk],
    filters: MetadataFilter
) -> list[int]:
    """
    Apply metadata filters to chunks and return matching chunk IDs.
    
    Args:
        chunks: List of all recipe chunks
        filters: MetadataFilter to apply
    
    Returns:
        List of chunk IDs that match the filters
    """
    if filters.is_empty():
        return list(range(len(chunks)))
    
    matching_ids = []
    
    for chunk in chunks:
        # Check category
        if filters.category and chunk.category != filters.category:
            continue
        
        # Check required ingredients
        if filters.ingredients_include:
            chunk_ingredients = " ".join(chunk.ingredients).lower()
            chunk_ner = " ".join(chunk.ner).lower()
            combined = chunk_ingredients + " " + chunk_ner
            
            if not all(ing.lower() in combined for ing in filters.ingredients_include):
                continue
        
        # Check excluded ingredients
        if filters.ingredients_exclude:
            chunk_ingredients = " ".join(chunk.ingredients).lower()
            chunk_ner = " ".join(chunk.ner).lower()
            combined = chunk_ingredients + " " + chunk_ner
            
            if any(ing.lower() in combined for ing in filters.ingredients_exclude):
                continue
        
        matching_ids.append(chunk.id)
    
    return matching_ids


class MetadataFilterer:
    """
    Metadata filtering handler for recipe search.
    """
    
    def __init__(self, chunks: list[RecipeChunk], use_llm: bool = False):
        """
        Initialize the filterer.
        
        Args:
            chunks: List of recipe chunks
            use_llm: Whether to use LLM for filter extraction
        """
        self.chunks = chunks
        self.use_llm = use_llm
        self.llm_client = None
    
    def set_llm_client(self, llm_client):
        """Set the LLM client for filter extraction."""
        self.llm_client = llm_client
    
    def extract_and_apply(self, query: str) -> tuple[list[int], MetadataFilter]:
        """
        Extract filters from query and return matching chunk IDs.
        
        Args:
            query: User's search query
        
        Returns:
            Tuple of (matching chunk IDs, extracted filters)
        """
        if self.use_llm and self.llm_client:
            filters = extract_filters_with_llm(query, self.llm_client)
        else:
            filters = extract_filters_rule_based(query)
        
        matching_ids = apply_filters(self.chunks, filters)
        
        return matching_ids, filters

