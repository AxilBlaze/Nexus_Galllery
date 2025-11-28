from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, FieldCondition, MatchValue,
    Filter, PointStruct, Distance
)
import google.generativeai as genai
import uuid
import os

# -------------------------------
# GEMINI CONFIG
# -------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# -------------------------------
# QDRANT CONFIG
# -------------------------------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = "new_img1"

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# -------------------------------
# EMBEDDING (Google Gemini)
# -------------------------------
print("🚀 Initializing Gemini Embeddings...")
EMBEDDING_MODEL = "models/text-embedding-004"  # 768 dimensions
EMBEDDING_DIM = 768
GEMINI_FLASH = "gemini-2.5-flash"  # For semantic verification
print("✅ Gemini configured!")

def embed(text: str):
    """
    Generates a vector using Google Gemini's text-embedding-004 model.
    This model produces 768-dimensional embeddings optimized for semantic search.
    """
    if not text or not text.strip():
        return [0.0] * EMBEDDING_DIM
    
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        raise Exception(f"Embedding failed: {str(e)}")

def embed_query(text: str):
    """
    Generates a vector for search queries using Gemini.
    Uses task_type="retrieval_query" for optimized query embeddings.
    """
    if not text or not text.strip():
        return [0.0] * EMBEDDING_DIM
    
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    
    except Exception as e:
        print(f"❌ Query embedding error: {e}")
        raise Exception(f"Query embedding failed: {str(e)}")

def semantic_match_all_concepts(query: str, summary: str) -> dict:
    """
    Uses Gemini to verify that ALL concepts from the query are present in the summary.
    Returns: {"match": True/False, "confidence": 0.0-1.0, "reason": "explanation"}
    """
    try:
        model = genai.GenerativeModel(GEMINI_FLASH)
        
        prompt = f"""You are a precise image search validator. Your job is to determine if ALL concepts from the query are present in the summary.

Query: "{query}"
Summary: "{summary}"

Rules:
- Return "MATCH" only if EVERY concept/requirement in the query is present in the summary
- Return "NO_MATCH" if even one concept from the query is missing
- Be strict: partial matches are not enough

Examples:
Query: "old and young man standing"
Summary: "one with a young and old man standing" → MATCH (both old AND young present)
Summary: "one with just a old man standing" → NO_MATCH (young is missing)
Summary: "one with just a young man standing" → NO_MATCH (old is missing)

Query: "beach sunset with family"
Summary: "family vacation at beach during sunset" → MATCH (all three concepts present)
Summary: "beach sunset" → NO_MATCH (family is missing)

Respond in this EXACT format:
RESULT: [MATCH or NO_MATCH]
CONFIDENCE: [0.0 to 1.0]
REASON: [one line explanation]"""

        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Parse response
        lines = text.split('\n')
        result = "NO_MATCH"
        confidence = 0.0
        reason = "Unknown"
        
        for line in lines:
            if line.startswith("RESULT:"):
                result = line.split(":", 1)[1].strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except:
                    confidence = 0.0
            elif line.startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()
        
        return {
            "match": result == "MATCH",
            "confidence": confidence,
            "reason": reason
        }
    
    except Exception as e:
        print(f"❌ Semantic matching error: {e}")
        return {"match": True, "confidence": 0.5, "reason": "Error in validation"}

def gemini_rerank_by_relevance(query: str, candidates: list) -> list:
    """
    Uses Gemini to deeply analyze and re-rank candidates by relevance.
    Reads ALL summaries and scores them against the query.
    
    Args:
        query: The search query
        candidates: List of dicts with 'cloudinary_id', 'summary', 'uuid', 'score'
    
    Returns:
        Re-ranked list with gemini_relevance_score and reasoning
    """
    if not candidates:
        return []
    
    try:
        model = genai.GenerativeModel(GEMINI_FLASH)
        
        # Build the prompt with all candidates
        candidates_text = "\n\n".join([
            f"[{i+1}] ID: {c['cloudinary_id']}\nSummary: {c['summary']}"
            for i, c in enumerate(candidates)
        ])
        
        prompt = f"""You are an expert image search ranking system. Your task is to analyze ALL summaries and rank them by how well they match the query.

QUERY: "{query}"

CANDIDATES:
{candidates_text}

INSTRUCTIONS:
1. Read each summary carefully
2. Score each one from 0.0 to 1.0 based on how well it matches the query
3. Consider:
   - Are ALL query concepts present? (highest score)
   - How closely does the content align with query intent?
   - Semantic relevance and context match
4. Return ONLY the ranking

Respond in this EXACT format (one line per candidate):
[ID] SCORE: [0.0-1.0] | REASON: [brief explanation]

Example:
[cloud-img-001] SCORE: 0.95 | REASON: Contains all query concepts with exact match
[cloud-img-002] SCORE: 0.60 | REASON: Partial match, missing some elements
[cloud-img-003] SCORE: 0.30 | REASON: Weak match, only tangentially related"""

        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Parse Gemini's ranking
        ranking_map = {}
        for line in text.split('\n'):
            if line.strip() and '[' in line:
                try:
                    parts = line.split('SCORE:', 1)
                    if len(parts) < 2:
                        continue
                    
                    id_part = parts[0].strip()
                    cloudinary_id = id_part.strip('[]').strip()
                    
                    score_reason = parts[1].split('|', 1)
                    score = float(score_reason[0].strip())
                    reason = score_reason[1].replace('REASON:', '').strip() if len(score_reason) > 1 else "No reason provided"
                    
                    ranking_map[cloudinary_id] = {
                        "gemini_score": score,
                        "gemini_reason": reason
                    }
                except Exception as parse_err:
                    print(f"⚠️ Parse error on line: {line} - {parse_err}")
                    continue
        
        # Apply Gemini scores to candidates
        for candidate in candidates:
            cid = candidate['cloudinary_id']
            if cid in ranking_map:
                candidate['gemini_relevance_score'] = ranking_map[cid]['gemini_score']
                candidate['gemini_reason'] = ranking_map[cid]['gemini_reason']
            else:
                candidate['gemini_relevance_score'] = candidate.get('score', 0.5)
                candidate['gemini_reason'] = "Not ranked by Gemini"
        
        # Sort by Gemini relevance score (highest first)
        reranked = sorted(candidates, key=lambda x: x['gemini_relevance_score'], reverse=True)
        
        print(f"🎯 Gemini re-ranking complete: {len(reranked)} results")
        return reranked
    
    except Exception as e:
        print(f"❌ Gemini re-ranking error: {e}")
        for candidate in candidates:
            candidate['gemini_relevance_score'] = candidate.get('score', 0.5)
            candidate['gemini_reason'] = "Re-ranking failed, using vector score"
        return candidates

# -------------------------------
# SEARCH LOGIC
# -------------------------------
def search_logic(query_text=None, face_ids=None, top_k=10, min_score=0.0, strict_match=False):
    """
    Search logic using Gemini embeddings for semantic search.
    
    Args:
        query_text: Search query string
        face_ids: List of face IDs to filter by
        top_k: Number of results to return
        min_score: Minimum similarity score threshold
        strict_match: If True, uses Gemini to verify ALL concepts from query are present
    
    Returns:
        List of search results
    """
    # Normalize face_ids to list
    if face_ids:
        if isinstance(face_ids, str):
            face_ids = [face_ids]
        face_ids = [f for f in face_ids if f and f.strip()]
    
    # Build filter for face_ids
    q_filter = None
    if face_ids:
        face_conditions = [
            FieldCondition(
                key="face_ids",
                match=MatchValue(value=fid)
            ) for fid in face_ids
        ]
        q_filter = Filter(must=face_conditions)

    # CASE A: Text Search (Semantic with Gemini)
    if query_text and query_text.strip():
        print(f"🔍 Searching with Gemini: '{query_text}' | Face IDs: {face_ids} | Strict: {strict_match}")
        
        query_vector = embed_query(query_text)
        
        try:
            initial_limit = top_k * 5 if strict_match else top_k
            
            res = client.query_points(
                collection_name=COLLECTION,
                query=query_vector,
                query_filter=q_filter,
                limit=initial_limit,
                score_threshold=min_score,
                with_payload=True
            )
            
            print(f"✅ Found {len(res.points)} initial results")
            
            results = []
            for p in res.points:
                d = p.payload
                
                result = {
                    "uuid": d.get("uuid"),
                    "cloudinary_id": d.get("cloudinary_id"),
                    "summary": d.get("summary"),
                    "face_ids": d.get("face_ids"),
                    "score": float(p.score),
                }
                
                if strict_match:
                    match_result = semantic_match_all_concepts(query_text, d.get("summary", ""))
                    result["strict_match"] = match_result["match"]
                    result["match_confidence"] = match_result["confidence"]
                    result["match_reason"] = match_result["reason"]
                    
                    if match_result["match"]:
                        results.append(result)
                        print(f"  ✓ MATCH: {result['uuid']} - {match_result['reason']}")
                    else:
                        print(f"  ✗ FILTERED: {result['uuid']} - {match_result['reason']}")
                else:
                    results.append(result)
            
            final_results = results[:top_k]
            print(f"🎯 Returning {len(final_results)} final results")
            
            return final_results
        
        except Exception as e:
            print(f"❌ Query error: {e}")
            raise Exception(f"Search error: {str(e)}")

    # CASE B: Only Face ID (Scroll)
    if face_ids:
        print(f"🔍 Scrolling for face_ids: {face_ids}")
        
        try:
            items, _ = client.scroll(
                collection_name=COLLECTION,
                scroll_filter=q_filter,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
            
            print(f"✅ Found {len(items)} results")
            
            results = []
            for p in items:
                d = p.payload
                results.append({
                    "uuid": d.get("uuid"),
                    "cloudinary_id": d.get("cloudinary_id"),
                    "summary": d.get("summary"),
                    "face_ids": d.get("face_ids"),
                    "score": 1.0
                })
            
            return results[:top_k]
        
        except Exception as e:
            print(f"❌ Scroll error: {e}")
            raise Exception(f"Scroll error: {str(e)}")

    # CASE C: No query provided
    raise Exception("Must provide either query or face_ids")

# -------------------------------
# MAIN FUNCTION: CLOUDINARY SEARCH
# -------------------------------
def search_cloudinary(query=None, face_ids=None, strict_match=False):
    """
    Returns Cloudinary IDs using Gemini-powered search AND re-ranking.
    
    WORKFLOW:
    1. Vector search finds initial candidates
    2. Gemini reads ALL summaries deeply
    3. Gemini re-ranks by true relevance to query
    4. Returns only IDs with gemini_relevance_score > 0.7
    
    Args:
        query (str, optional): Search query text
        face_ids (list, optional): List of face IDs to filter by
        strict_match (bool): Enable strict all-concepts matching
    
    Returns:
        list: List of Cloudinary IDs matching the query
    
    Example:
        >>> ids = search_cloudinary(query="beach sunset", strict_match=True)
        >>> print(ids)  # ['img-001', 'img-005', 'img-012']
    """
    # Step 1: Get initial candidates from vector search
    results = search_logic(
        query_text=query, 
        face_ids=face_ids, 
        top_k=100,
        min_score=0.50,
        strict_match=strict_match
    )
    
    # Step 2: Gemini re-ranks by reading all summaries (if query provided)
    if query and results:
        print(f"📊 Gemini re-ranking {len(results)} candidates...")
        results = gemini_rerank_by_relevance(query, results)
        print(f"✅ Re-ranking complete!")
    
    # Step 3: Filter by Gemini relevance score > 0.7 and extract IDs only
    filtered_ids = [
        r["cloudinary_id"]
        for r in results 
        if r.get("cloudinary_id") and r.get("gemini_relevance_score", r.get("score", 0.0)) > 0.7
    ]
    
    print(f"🎯 Filtered to {len(filtered_ids)} results with score > 0.7")
    
    return filtered_ids

# -------------------------------
# STORE FUNCTION
# -------------------------------
def store_image(uuid_val: str, cloudinary_id: str, summary: str, face_ids: list = None):
    """
    Stores image summary with Gemini embeddings.
    
    Args:
        uuid_val (str): Unique identifier for the image
        cloudinary_id (str): Cloudinary ID of the image
        summary (str): Text summary/description of the image
        face_ids (list, optional): List of face IDs present in the image
    
    Returns:
        dict: Status information
    
    Example:
        >>> store_image(
        ...     uuid_val="img-123",
        ...     cloudinary_id="cloud-abc-456",
        ...     summary="Beach sunset with family",
        ...     face_ids=["face-001", "face-002"]
        ... )
    """
    try:
        vector = embed(summary)
        
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "uuid": uuid_val,
                "cloudinary_id": cloudinary_id,
                "summary": summary,
                "face_ids": face_ids
            },
        )
        
        client.upsert(collection_name=COLLECTION, points=[point])
        print(f"✅ Stored with Gemini: {uuid_val}")
        
        return {
            "status": "stored", 
        }
    
    except Exception as e:
        print(f"❌ Store error: {e}")
        raise Exception(f"Store error: {str(e)}")

# -------------------------------
# UTILITY FUNCTIONS
# -------------------------------
def get_count():
    """
    Returns total count of stored images.
    
    Returns:
        int: Number of images in the collection
    """
    try:
        info = client.get_collection(collection_name=COLLECTION)
        return info.points_count
    except Exception as e:
        print(f"❌ Count error: {e}")
        raise Exception(f"Count error: {str(e)}")

def initialize_collection():
    """
    Creates the collection if it doesn't exist.
    """
    try:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE
            ),
        )
        print(f"📦 Collection '{COLLECTION}' created with {EMBEDDING_DIM}D vectors.")
    except Exception as e:
        print(f"Collection exists or error: {e}")

    try:
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name="face_ids",
            field_schema="keyword"
        )
        print("✅ Indexed face_ids field")
    except Exception as e:
        print(f"Index exists or error: {e}")

# Initialize collection on import
initialize_collection()