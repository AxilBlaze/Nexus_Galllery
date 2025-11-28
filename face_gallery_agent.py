"""Face UUID Auto-Tagging Tool built on DeepFace + ChromaDB."""
import os
import uuid
import logging
from typing import Dict, List

import chromadb
from deepface import DeepFace

# Configure logging to suppress DeepFace/TensorFlow noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceUUIDTool:
    """
    Automated tool to detect faces and assign consistent UUIDs.
    - If a face is known, returns existing UUID.
    - If a face is new, generates new UUID and stores it.
    """

    def __init__(
        self,
        db_path: str = "./face_db",
        collection_name: str = "unique_faces",
        threshold: float = 0.5,  # Facenet512 cosine threshold
        detector_backend: str = "retinaface",
        duplicate_iou: float = 0.55,
    ) -> None:
        """
        Args:
            db_path: Location for the persistent ChromaDB store.
            collection_name: Embedding collection name.
            threshold: Cosine distance cutoff (lower = stricter).
            detector_backend: DeepFace detector backend to use.
            duplicate_iou: IoU threshold for merging overlapping detections.
        """
        self.threshold = threshold
        self.detector_backend = detector_backend
        self.duplicate_iou_threshold = duplicate_iou

        # Initialize persistent DB
        os.makedirs(db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"FaceUUIDTool initialized. DB: {db_path} | Threshold: {threshold}")

    def run(self, image_path: str) -> List[str]:
        """
        Process an image and return a list of UUIDs for all faces found.
        """
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return []

        logger.info(f"Processing image: {image_path}")
        
        # 1. Detect & Embed
        face_objs = self._embed_faces(image_path)
        
        # 2. Deduplicate (remove overlapping boxes)
        unique_faces = self._deduplicate_faces(face_objs)
        
        if not unique_faces:
            logger.warning("No faces detected.")
            return []

        face_ids = []
        
        # 3. Process each face (Match or Create)
        for i, face_obj in enumerate(unique_faces):
            embedding = face_obj["embedding"]
            
            # Query the DB for the nearest neighbor
            matched_id, dist = self._find_nearest_face(embedding)

            if matched_id and dist < self.threshold:
                # SCENARIO A: Known Face
                logger.info(f"  [Face {i+1}] Match Found: {matched_id} (dist: {dist:.4f})")
                face_ids.append(matched_id)
            else:
                # SCENARIO B: New Face
                new_id = str(uuid.uuid4())
                logger.info(f"  [Face {i+1}] New Face. Generated UUID: {new_id} (dist: {dist:.4f} > {self.threshold})")
                
                # Store in DB
                self.collection.add(
                    ids=[new_id],
                    embeddings=[embedding],
                    metadatas=[{
                        "source_image": os.path.basename(image_path),
                        "created_at": str(os.path.getmtime(image_path))
                    }]
                )
                face_ids.append(new_id)

        return face_ids

    def _embed_faces(self, image_path: str) -> List[Dict]:
        """Return DeepFace embeddings for every detected face."""
        try:
            return DeepFace.represent(
                img_path=image_path,
                model_name="Facenet512",
                detector_backend=self.detector_backend,
                enforce_detection=True,
            )
        except ValueError:
            return []
        except Exception as exc:
            logger.error(f"DeepFace error: {exc}")
            return []

    def _find_nearest_face(self, embedding: List[float]) -> tuple[str | None, float]:
        """Query DB for nearest neighbor. Returns (uuid, distance)."""
        try:
            result = self.collection.query(
                query_embeddings=[embedding],
                n_results=1
            )
            
            if not result['ids'] or not result['ids'][0]:
                return None, 1.0  # Max distance if empty
            
            existing_id = result['ids'][0][0]
            distance = result['distances'][0][0]
            
            return existing_id, distance
        except Exception as exc:
            logger.error(f"Database query failed: {exc}")
            return None, 1.0

    def _deduplicate_faces(self, face_objs: List[Dict]) -> List[Dict]:
        """Merge overlapping detections to avoid duplicate IDs."""
        if not face_objs:
            return []

        # Sort by area (largest faces first)
        def area(face: Dict) -> float:
            r = self._extract_region(face)
            return float(r["w"] * r["h"])

        filtered: List[Dict] = []
        for face in sorted(face_objs, key=area, reverse=True):
            region = self._extract_region(face)
            if region["w"] <= 0 or region["h"] <= 0:
                continue

            # Check IoU against already accepted faces
            if any(
                self._compute_iou(region, self._extract_region(saved))
                >= self.duplicate_iou_threshold
                for saved in filtered
            ):
                continue
            filtered.append(face)
        return filtered

    @staticmethod
    def _extract_region(face_obj: Dict) -> Dict[str, int]:
        """Normalize DeepFace region dictionary."""
        region = face_obj.get("facial_area") or face_obj.get("region") or {}
        return {
            "x": int(region.get("x", 0)),
            "y": int(region.get("y", 0)),
            "w": int(region.get("w", 0)),
            "h": int(region.get("h", 0))
        }

    @staticmethod
    def _compute_iou(region_a: Dict[str, int], region_b: Dict[str, int]) -> float:
        """Compute Intersection over Union (IoU)."""
        ax1, ay1 = region_a["x"], region_a["y"]
        ax2, ay2 = ax1 + region_a["w"], ay1 + region_a["h"]
        bx1, by1 = region_b["x"], region_b["y"]
        bx2, by2 = bx1 + region_b["w"], by1 + region_b["h"]

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, region_a["w"]) * max(0, region_a["h"])
        area_b = max(0, region_b["w"]) * max(0, region_b["h"])
        
        union = area_a + area_b - inter_area
        return 0.0 if union == 0 else inter_area / union


def get_face_ids(image_path: str, db_path: str = "./face_db") -> List[str]:
    """
    Standalone function to process an image and return face UUIDs.
    Initializes the FaceUUIDTool with default settings.
    """
    print(f"Processingggggggggggggggggggggggggggggggggggg image: {image_path}")
    tool = FaceUUIDTool(db_path=db_path)
    return tool.run(image_path)


# if __name__ == "__main__":
#     # Replace with your test image
#     sample_image = "images/sample_face7.jpg"

#     if os.path.exists(sample_image):
#         uuids = get_face_ids(sample_image)
#         print("\n--- Final Result ---")
#         print(f"Image: {sample_image}")
#         print(f"Detected face_id: {uuids}")
#     else:
#         print(f"Please create a dummy image at '{sample_image}' to test.")