# Example Queries for Recipe RAG System

## Queries where BM25 performs better

These queries benefit from exact keyword matching:

1. **"chocolate chip cookie recipe"**
   - Exact match for "chocolate chip cookie"
   - BM25 finds exact term matches quickly

2. **"recipe with cream cheese"**
   - Specific ingredient search
   - BM25 matches "cream cheese" exactly in ingredient lists

3. **"brisket marinade"**
   - Specific cooking term + technique
   - Keyword matching is precise

4. **"buttermilk pancakes"**
   - Specific ingredient in title
   - BM25 matches both words exactly

## Queries where Semantic Search performs better

These queries require understanding meaning:

1. **"something sweet for a party"**
   - No specific keywords to match
   - Semantic understands "sweet" → desserts, cakes, cookies

2. **"healthy dinner ideas"**
   - Concept-based query
   - Semantic understands health-related recipes

3. **"quick meal for busy weeknight"**
   - Abstract concept
   - Semantic finds "easy", "fast", "simple" recipes

4. **"comfort food for cold weather"**
   - Emotional/contextual query
   - Semantic understands soups, stews, warm dishes

5. **"kid-friendly lunch"**
   - Audience-specific query
   - Semantic finds simple, appealing recipes

## Complex queries (Hybrid recommended)

1. **"vegetarian pasta without tomatoes"**
   - Needs semantic (vegetarian concept) + exact filtering (no tomatoes)

2. **"gluten-free dessert with chocolate"**
   - Dietary restriction + specific ingredient

3. **"traditional Christmas cookies"**
   - Cultural context + specific item type

## Metadata filtering examples

1. **"soup recipe"** → filters to category=soup
2. **"dessert without nuts"** → category=dessert, exclude=nuts
3. **"chicken main course"** → category=main_course, include=chicken

## Tips for best results

- Use specific ingredients for BM25
- Use natural language for Semantic
- Enable both (Hybrid) for best coverage
- Enable Reranker for improved ranking
- Use Metadata Filtering for category-specific searches

