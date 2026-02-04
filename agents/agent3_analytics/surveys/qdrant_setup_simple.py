#!/usr/bin/env python3
"""
Qdrant collection setup for survey responses storage.
Creates a collection and uploads survey data without embeddings.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import hashlib

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
except ImportError:
    print("Error: qdrant-client not installed")
    sys.exit(1)


class SurveyQdrantManager:
    """Manages Qdrant collection for survey responses."""
    
    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        """Initialize Qdrant client."""
        try:
            self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
            print(f"✓ Connected to Qdrant at {qdrant_host}:{qdrant_port}")
        except Exception as e:
            print(f"✗ Failed to connect to Qdrant: {e}")
            print("  Make sure Qdrant is running: docker-compose -f qdrant/docker-compose.yml up")
            sys.exit(1)
        
        self.embedding_dim = 128  # Simple hash-based embedding dimension
    
    def create_collection(self, collection_name: str = "survey_responses") -> bool:
        """Create a new collection in Qdrant."""
        try:
            # Check if collection exists
            try:
                self.client.get_collection(collection_name)
                print(f"✓ Collection '{collection_name}' already exists")
                return True
            except:
                pass
            
            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                ),
            )
            print(f"✓ Created collection '{collection_name}'")
            return True
        except Exception as e:
            print(f"✗ Failed to create collection: {e}")
            return False
    
    def generate_simple_embedding(self, text: str) -> List[float]:
        """Generate a simple hash-based embedding for text."""
        # Create a deterministic hash-based vector
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert hash bytes to normalized vector
        embedding = []
        for i in range(self.embedding_dim):
            byte_val = hash_bytes[i % len(hash_bytes)]
            # Normalize to [-1, 1]
            embedding.append((byte_val / 127.5) - 1.0)
        
        return embedding
    
    def prepare_survey_vector(self, survey: Dict[str, Any]) -> str:
        """Prepare survey data as text for embedding."""
        text_parts = [
            f"Cel: {survey.get('q1_goal', 'N/A')}",
            f"Czas: {survey.get('q2_duration', 'N/A')}",
            f"Zrozumienie pytań: {survey.get('q3_understanding', 'N/A')}/5",
            f"Jasność odpowiedzi: {survey.get('q4_clarity', 'N/A')}/5",
            f"Dokładność: {survey.get('q5_accuracy', 'N/A')}/5",
            f"Użyteczność: {survey.get('q6_usefulness', 'N/A')}/5",
            f"Zaawansowane pytania: {survey.get('q7_advanced_questions', 'N/A')}/5",
            f"Ogólnie: {survey.get('q8_overall_experience', 'N/A')}/5",
            f"Łatwość użycia: {survey.get('q9_ease_of_use', 'N/A')}/5",
            f"Szybkość: {survey.get('q10_response_time', 'N/A')}/5",
            f"Zrozumienie AI: {survey.get('q11_ai_understanding', 'N/A')}/5",
            f"Chat vs nauczyciel: {survey.get('q12_chat_vs_teacher', 'N/A')}",
            f"Przyszłe użycie: {survey.get('q13_future_use', 'N/A')}",
            f"Pozytywne: {survey.get('q14_positive', 'N/A')}",
            f"Ulepszenia: {survey.get('q15_improvements', 'N/A')}",
            f"Problemy: {survey.get('q16_problems', 'N/A')}",
        ]
        return " ".join(text_parts)
    
    def load_surveys(self, responses_dir: str) -> List[Dict[str, Any]]:
        """Load all survey response files."""
        surveys = []
        responses_path = Path(responses_dir)
        
        if not responses_path.exists():
            print(f"✗ Directory not found: {responses_dir}")
            return surveys
        
        for i in range(1, 31):
            filename = f"survey_response_{i:02d}.json"
            filepath = responses_path / filename
            
            if filepath.exists():
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        survey = json.load(f)
                        surveys.append(survey)
                except Exception as e:
                    print(f"✗ Error loading {filename}: {e}")
        
        print(f"✓ Loaded {len(surveys)} survey responses")
        return surveys
    
    def upload_surveys(self, surveys: List[Dict[str, Any]], collection_name: str = "survey_responses") -> bool:
        """Upload surveys to Qdrant with simple embeddings."""
        if not surveys:
            print("✗ No surveys to upload")
            return False
        
        print(f"Generating embeddings for {len(surveys)} surveys...")
        
        points = []
        for idx, survey in enumerate(surveys, 1):
            try:
                # Generate embedding
                survey_text = self.prepare_survey_vector(survey)
                embedding = self.generate_simple_embedding(survey_text)
                
                # Prepare payload
                payload = {
                    "response_id": survey.get("response_id", f"resp_{idx:03d}"),
                    "student_id": survey.get("student_id"),
                    "timestamp": survey.get("timestamp"),
                    "satisfaction_score": float(survey.get("satisfaction_score", 0)),
                    "q1_goal": survey.get("q1_goal"),
                    "q2_duration": survey.get("q2_duration"),
                    "q8_overall_experience": survey.get("q8_overall_experience"),
                    "q12_chat_vs_teacher": survey.get("q12_chat_vs_teacher"),
                    "q13_future_use": survey.get("q13_future_use"),
                    "q14_positive": survey.get("q14_positive"),
                    "q15_improvements": survey.get("q15_improvements"),
                    "q16_problems": survey.get("q16_problems"),
                }
                
                # Create point
                point = PointStruct(
                    id=idx,
                    vector=embedding,
                    payload=payload
                )
                points.append(point)
                
                if idx % 10 == 0:
                    print(f"  Processed {idx}/{len(surveys)}")
            
            except Exception as e:
                print(f"✗ Error processing survey {idx}: {e}")
        
        # Upload to Qdrant
        try:
            print(f"\nUploading {len(points)} points to Qdrant...")
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            print(f"✓ Uploaded {len(points)} surveys to collection '{collection_name}'")
            return True
        except Exception as e:
            print(f"✗ Failed to upload to Qdrant: {e}")
            return False
    
    def search_surveys(self, query: str, collection_name: str = "survey_responses", limit: int = 5) -> List[Dict[str, Any]]:
        """Search similar surveys using vector similarity."""
        try:
            # Generate query embedding
            query_embedding = self.generate_simple_embedding(query)
            
            # Search in Qdrant using search method
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=limit
            )
            
            return results
        except Exception as e:
            # Try alternative search method
            try:
                from qdrant_client.models import PointIdsList
                query_embedding = self.generate_simple_embedding(query)
                results = self.client.query_points(
                    collection_name=collection_name,
                    query=query_embedding,
                    limit=limit
                )
                return results.points if hasattr(results, 'points') else []
            except Exception as e2:
                print(f"✗ Search error: {e}")
                return []
    
    def get_collection_stats(self, collection_name: str = "survey_responses") -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "points_count": info.points_count,
            }
        except Exception as e:
            print(f"✗ Error getting stats: {e}")
            return {}


def main():
    """Main execution."""
    # Setup paths
    script_dir = Path(__file__).parent
    responses_dir = script_dir / "responses"
    
    # Initialize manager
    manager = SurveyQdrantManager()
    
    # Create collection
    if not manager.create_collection("survey_responses"):
        sys.exit(1)
    
    # Load surveys
    surveys = manager.load_surveys(str(responses_dir))
    if not surveys:
        print("✗ No surveys loaded")
        sys.exit(1)
    
    # Upload to Qdrant
    if not manager.upload_surveys(surveys, "survey_responses"):
        sys.exit(1)
    
    # Get stats
    print("\n📊 Collection Statistics:")
    stats = manager.get_collection_stats("survey_responses")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Example search
    print("\n🔍 Example Search:")
    print("  Query: 'chat był bardzo pomocny i szybki'")
    results = manager.search_surveys(
        "chat był bardzo pomocny i szybki",
        limit=3
    )
    
    if results:
        for i, result in enumerate(results, 1):
            print(f"  Result {i}:")
            print(f"    ID: {result.payload.get('response_id')}")
            print(f"    Satisfaction: {result.payload.get('satisfaction_score')}/5")
            print(f"    Similarity: {result.score:.3f}")
    else:
        print("  No results")
    
    print("\n✅ Qdrant collection setup complete!")
    print("   Use SurveyQdrantManager to search and analyze survey data.")


if __name__ == "__main__":
    main()
