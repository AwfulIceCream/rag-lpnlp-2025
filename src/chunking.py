"""
Chunking module for processing recipes dataset.
Loads CSV data and converts recipes into searchable chunks.
"""

import ast
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class RecipeChunk:
    """Represents a single recipe chunk with metadata."""
    id: int
    title: str
    ingredients: list[str]
    directions: list[str]
    link: str
    source: str
    ner: list[str]
    category: Optional[str] = None
    
    @property
    def text(self) -> str:
        """Full text representation for embedding/search."""
        ingredients_text = ", ".join(self.ingredients)
        directions_text = " ".join(self.directions)
        return f"{self.title}\n\nIngredients: {ingredients_text}\n\nDirections: {directions_text}"
    
    @property
    def display_text(self) -> str:
        """Formatted text for display in UI."""
        ingredients_list = "\n".join(f"  • {ing}" for ing in self.ingredients)
        directions_list = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(self.directions))
        return f"**{self.title}**\n\n**Ingredients:**\n{ingredients_list}\n\n**Directions:**\n{directions_list}"


def safe_parse_list(value: str) -> list[str]:
    """Safely parse a string representation of a list."""
    if pd.isna(value) or not value:
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if item]
        return []
    except (ValueError, SyntaxError):
        return []


def extract_category(title: str) -> Optional[str]:
    """Extract category from recipe title using keyword matching."""
    title_lower = title.lower()
    
    categories = {
        "soup": ["soup", "chowder", "stew", "broth", "bisque"],
        "salad": ["salad", "slaw", "coleslaw"],
        "dessert": ["cake", "cookie", "pie", "brownie", "candy", "fudge", "pudding", "ice cream", "cupcake", "cheesecake", "tart", "mousse", "custard"],
        "bread": ["bread", "muffin", "roll", "biscuit", "scone", "bagel", "loaf"],
        "main_course": ["chicken", "beef", "pork", "fish", "salmon", "shrimp", "steak", "roast", "casserole", "lasagna", "meatloaf", "turkey"],
        "appetizer": ["dip", "appetizer", "spread", "bruschetta", "wings"],
        "breakfast": ["breakfast", "pancake", "waffle", "omelet", "french toast", "eggs"],
        "drink": ["punch", "cocktail", "smoothie", "lemonade", "tea", "coffee"],
        "sauce": ["sauce", "gravy", "dressing", "marinade"],
        "side_dish": ["rice", "potato", "beans", "corn", "vegetable"],
    }
    
    for category, keywords in categories.items():
        if any(kw in title_lower for kw in keywords):
            return category
    
    return "other"


def load_recipes(
    csv_path: str,
    max_recipes: int = 5000,
    random_seed: int = 42
) -> list[RecipeChunk]:
    """
    Load recipes from CSV and convert to chunks.
    
    Args:
        csv_path: Path to the full_dataset.csv file
        max_recipes: Maximum number of recipes to load
        random_seed: Random seed for reproducible sampling
    
    Returns:
        List of RecipeChunk objects
    """
    print(f"Loading recipes from {csv_path}...")
    
    # Load CSV with sampling for large datasets
    df = pd.read_csv(csv_path, nrows=max_recipes * 2)  # Load more to filter
    
    # Sample if we have more than needed
    if len(df) > max_recipes:
        df = df.sample(n=max_recipes, random_state=random_seed)
    
    chunks = []
    for idx, row in df.iterrows():
        try:
            chunk = RecipeChunk(
                id=len(chunks),
                title=str(row.get('title', '')).strip(),
                ingredients=safe_parse_list(row.get('ingredients', '[]')),
                directions=safe_parse_list(row.get('directions', '[]')),
                link=str(row.get('link', '')),
                source=str(row.get('source', '')),
                ner=safe_parse_list(row.get('NER', '[]')),
                category=extract_category(str(row.get('title', '')))
            )
            
            # Skip empty recipes
            if chunk.title and (chunk.ingredients or chunk.directions):
                chunks.append(chunk)
        except Exception as e:
            continue  # Skip problematic rows
    
    print(f"Loaded {len(chunks)} recipe chunks")
    return chunks


def get_chunk_texts(chunks: list[RecipeChunk]) -> list[str]:
    """Extract text from chunks for embedding."""
    return [chunk.text for chunk in chunks]


def get_chunk_titles(chunks: list[RecipeChunk]) -> list[str]:
    """Extract titles from chunks."""
    return [chunk.title for chunk in chunks]

