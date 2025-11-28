import os
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.plugins.save_files_as_artifacts_plugin import SaveFilesAsArtifactsPlugin
from google.adk.tools.tool_context import ToolContext
import uuid
from google.adk.agents import LlmAgent
from google.genai import types , client
from typing import Dict, Any, List
from PIL import Image
import urllib.parse
from pydantic import BaseModel, Field
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
from agents.face_gallery_agent import get_face_ids
from agents.database import search_cloudinary, store_image
import requests
from typing import List, Optional
import json


# Load environment variables
load_dotenv()

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET')
)

async def save_image_to_local_disk(tool_context: ToolContext, artifact_name: str, output_filename: str) -> str:
    """
    Saves a user-uploaded image artifact to a local folder and returns the file path.
    """
    try:
        # 1. Define where to save images
        save_folder = "downloaded_images"
        os.makedirs(save_folder, exist_ok=True)
        
        # 2. Load the image bytes from the ADK artifact system
        # FIX: We await the function call itself
        artifact = await tool_context.load_artifact(artifact_name)
        
        # Now 'artifact' is the real object, so we can access .inline_data
        image_bytes = artifact.inline_data.data
        
        # 3. Create the full local path
        full_path = os.path.abspath(os.path.join(save_folder, output_filename))
        
        # 4. Write bytes to disk
        with open(full_path, "wb") as f:
            f.write(image_bytes)
            
        return f"SUCCESS: Image saved locally at: {full_path}"

    except Exception as e:
        return f"ERROR: Could not save image. Details: {str(e)}"

class ImageRecord(BaseModel):
    uuid: str = Field(description="The unique UUID generated for the image")
    summary: str = Field(description="The detailed summary text of the image")
    cloudinary_id: str = Field(description="The unique ID generated for the image on cloudinary")
    face_ids: List[str] = Field(description="The list of face UUIDs detected in the image")

# --- Tool Functions ---

def uuid_generator() -> str:
    """Generates a unique identifier (UUID) for the image."""
    return str(uuid.uuid4())


def image_summarizer(filename: str) -> str:
    """
    Locates an image in the 'downloaded_images' folder by filename/uuid, 
    sends it to Gemini for analysis, and returns the summary.
    """
    save_folder = "downloaded_images"
    file_path = os.path.join(save_folder, filename)

    # 1. Search for file
    if not os.path.exists(file_path):
        return f"ERROR: File '{filename}' not found in {save_folder}."

    try:
        # 2. Initialize Client
        # Ensure GOOGLE_API_KEY is set in your environment variables
        ai_client = client.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

        # 3. Load Image locally
        image = Image.open(file_path)

        prompt = """"You are an expert Vision-Language AI specialized in producing extremely detailed, exhaustive, accurate image summaries. Your job is to analyze any given image and generate the most thorough possible description covering every visible detail, even subtle or minor ones, presented as a single continuous paragraph without any headings, subheadings, or structured sections.

IMPORTANT RULE: Never mention things that are missing in the image. Never say 'there is no...', 'not found...', or 'nothing is present.' Simply SKIP any category that is not visible and only describe what actually exists in the image.

CRITICAL OUTPUT FORMAT: Your entire response must be one continuous paragraph. Do not use headings like "Summary:", "Detailed Description:", "Entities:", or any other section titles. Do not use bullet points, numbered lists, or line breaks between different aspects of the description. Write everything as flowing prose in a single unified paragraph.

What to describe (only if visible):

1. Count and Identify All Beings:
- Count how many people are visible and describe each: gender (best guess), age range, skin tone, facial features, expressions, hairstyle, beard, makeup, posture, orientation, and actions.
- Describe any visible animals: type, size, color, behavior, and interaction with people.
- Describe any visible statues, toys, illustrations, posters, or humanoid figures.

2. Clothing Details:
For every visible person, describe clothing in high detail: type of clothing, colors with shades, patterns, logos, text, textures, and all accessories such as watches, bracelets, necklaces, earphones, headphones, jewelry, glasses, bags, belts, and footwear with colors and styles.

3. Food and Objects:
Describe all visible objects: food items, drinks, plates, cups, utensils, mobile phones, laptops, books, papers, tools, toys, furniture, electronics, vehicles, and any readable text on signs, packaging, clothing, or labels.

4. Environment / Location:
Describe the environment only if visible:
- Indoor details: type of room, wall color, textures, decorations, furniture, lighting type, reflections, shadows, and room organization.
- Outdoor details: time of day, weather, sky condition, trees, plants, roads, vehicles, buildings, water bodies, signboards, terrain, and surroundings.

5. Background Details:
Describe all visible elements in the background: buildings, posters, decorations, lights, windows, furniture, scenery, distant people, distant objects, reflections, shadows, or environmental elements.

6. Scene Context:
Describe the possible context if visible: party, gathering, meeting, celebration, casual moment, professional environment, selfie, travel photo, event, activity, or any identifiable scenario.

7. Spatial Arrangement:
Describe where people and objects are positioned relative to each other, foreground vs background layout, and how the scene is structured visually.

8. Image Quality & Technical Notes:
Describe visible technical aspects: image clarity, resolution feel, lighting strength, shadows, camera angle, framing, and focus.

9. Do Not Assume:
Do NOT guess anything not visible. Do NOT list missing elements. If something cannot be seen clearly, describe it as 'unclear' or 'not fully visible' without saying it does not exist.

10. Integration and Flow:
Weave all observations together naturally in paragraph form. Start with an overview, then flow through people, clothing, objects, environment, background, spatial relationships, and technical qualities seamlessly. Use transition words and phrases to connect different aspects of the description naturally.

Always obey the rules: Only describe what exists. Never mention things that are missing. Output everything as one continuous paragraph with no headings or formatting.
"""

        # 4. Call LLM for Summary
        response = ai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, image]
        )
        
        return response.text

    except Exception as e:
        return f"ERROR during summarization: {str(e)}"

def cloudinary_saver(filename: str) -> str:
    """
    Locates an image in the 'downloaded_images' folder by filename/uuid, 
    sends it to cloudinary for upload and returns the cloudinary public_id.
    """
    save_folder = "downloaded_images"
    file_path = os.path.join(save_folder, filename)

    # 1. Search for file
    if not os.path.exists(file_path):
        return f"ERROR: File '{filename}' not found in {save_folder}."

    try:
        response = cloudinary.uploader.upload(file_path)
        print("Upload successful!")
        public_id = response.get('public_id')
        print(f"Public ID: {public_id}")
        return public_id
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def get_face(filename: str) -> str:
    """
    Locates an image in the 'downloaded_images' folder by filename/uuid, 
    sends it to face_gallery_agent for face detection and returns the face ids.
    """
    save_folder = "downloaded_images"
    file_path = os.path.join(save_folder, filename)

    # 1. Search for file
    if not os.path.exists(file_path):
        return f"ERROR: File '{filename}' not found in {save_folder}."
    
    face_ids = get_face_ids(file_path)
    return face_ids

def qdrant_db(uuid: str, summary: str, cloudinary_id: str, face_ids: List[str]) -> str:
    """
    Saves the image in db.
    
    Args:
        uuid: Unique identifier for the image
        summary: Text description/summary of the image
        cloudinary_id: Cloudinary storage ID
        face_ids: List of detected face IDs in the image
    
    Returns:
        str: Success or error message
    """
    
    response = store_image(uuid, cloudinary_id, summary, face_ids)
    
    if response["status"] == "stored":
        return f"✅ Image saved successfully!"
    else:
        return f"❌ Error saving image: {response['error']}"
        
        

def delete_image_from_local_disk(filename: str) -> str:
    """
    Deletes an image from the 'downloaded_images' folder by filename/uuid.
    """
    save_folder = "downloaded_images"
    file_path = os.path.join(save_folder, filename)

    # 1. Search for file
    if not os.path.exists(file_path):
        return f"ERROR: File '{filename}' not found in {save_folder}."

    try:
        os.remove(file_path)
        return f"SUCCESS: Image deleted from {save_folder}."
    except Exception as e:
        return f"ERROR: Could not delete image. Details: {str(e)}"

def get_cloudinary_url(cloudinary_ids: List[str]) -> str:
    """
    Converts array of Cloudinary IDs to their public URLs.
    
    Args:
        cloudinary_ids: List of Cloudinary public IDs
    
    Returns:
        str: JSON string containing array of URL objects with id and url
    """
    try:
        urls = []
        
        for cloudinary_id in cloudinary_ids:
            try:
                url = cloudinary.utils.cloudinary_url(cloudinary_id)[0]
                
                urls.append({
                    "id": cloudinary_id,
                    "url": url
                })
                
            except Exception as e:
                print(f"Error generating URL for {cloudinary_id}: {e}")
                urls.append({
                    "id": cloudinary_id,
                    "url": None,
                    "error": str(e)
                })
        
        return json.dumps(urls, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Failed to process cloudinary IDs: {str(e)}"})



def get_data_from_db(query_summary: Optional[str], face_ids: Optional[List[str]]) -> str:
    """
    Retrieves images from the database based on query and/or face IDs.
    
    Args:
        query_summary: Text description to search for (can be None)
        face_ids: List of face IDs to search for (can be None)
    
    Returns:
        str: JSON string containing array of cloudinary IDs or error message
    """
    
    results = search_cloudinary(query_summary, face_ids)
    
    return results
        
        


save_db_agent = LlmAgent(
    name="SaveDBAgent",
    model="gemini-2.0-flash",
    description="saves the image in db by using qdrant_db tool",
    instruction=(
        "You are a data processor. "
        "1. Call `qdrant_db` to save the data in db."
        "2. Qdrant db will tell whether the data was saved or not. Show the result to the user in a good manner."
    ),
    tools=[qdrant_db],
)

store_image_agent = LlmAgent(
    name="StoreImageAgent",
    model="gemini-2.0-flash",
    description="An agent that returns a strict JSON object with uuid and summary.",
    instruction=(
        "You are a data processor. "
        "1. Call `uuid_generator` to get an ID. "
        "2. Call save_image_to_local_disk to save the image to disk with filename as the uuid. "
        "3. Call `image_summarizer` to get the summary by passing uuid as input. "
        "4. Call `cloudinary_saver` to upload the image to cloudinary and get the public_id. "
        "5. Call `get_face` to get the face_ids by passing uuid as input. "
        "6. Call `delete_image_from_local_disk` to delete the image from disk by passing uuid as input. "
        "7. Send the uuid, summary, cloudinary_id, and face_ids to `SaveDBAgent`."
    ),
    tools=[uuid_generator, image_summarizer, save_image_to_local_disk, cloudinary_saver, get_face, delete_image_from_local_disk],
    sub_agents=[save_db_agent],
    # output_schema=ImageRecord
)

get_image_agent = LlmAgent(
    name="GetImageAgent",
    model="gemini-2.0-flash",
    description="An agent that return images based on user query and reference image.",
    instruction=(
        """
You are GetImageAgent - an intelligent image retrieval specialist with advanced image processing capabilities.

YOUR MISSION: Retrieve images from the database based on user query, reference image, or both.

AVAILABLE TOOLS:

Image Processing Tools:
1. uuid_generator - Generates a unique UUID for image processing
2. save_image_to_local_disk - Saves uploaded image to local storage using UUID
3. get_face - Extracts face IDs from saved image using UUID as filename
4. delete_image_from_local_disk - Removes temporary image file after processing

Database & Retrieval Tools:
5. get_data_from_db - Searches database using face IDs and/or text query
6. get_cloudinary_url - Converts Cloudinary IDs to actual image URLs

WORKFLOW LOGIC:

SCENARIO 1: User provides BOTH image + query
Execute these steps silently without narration:
Step 1: Call uuid_generator() → Get UUID
Step 2: Call save_image_to_local_disk(image, uuid) → Save image locally
Step 3: Call get_face(uuid) → Get face_ids
Step 4: Call delete_image_from_local_disk(uuid) → Clean up temporary file
Step 5: Call get_data_from_db(face_ids, query_summary) → Get cloudinary_ids
Step 6: Call get_cloudinary_url(cloudinary_ids) → Get actual URLs
Step 7: Display ONLY the URLs to user (one per line, nothing else)

SCENARIO 2: User provides ONLY query (no image)
Execute these steps silently without narration:
Step 1: Call get_data_from_db(face_ids=None, query_summary) → Get cloudinary_ids
Step 2: Call get_cloudinary_url(cloudinary_ids) → Get actual URLs
Step 3: Display ONLY the URLs to user (one per line, nothing else)

SCENARIO 3: User provides ONLY image (no query)
Execute these steps silently without narration:
Step 1: Call uuid_generator() → Get UUID
Step 2: Call save_image_to_local_disk(image, uuid) → Save image locally
Step 3: Call get_face(uuid) → Get face_ids
Step 4: Call delete_image_from_local_disk(uuid) → Clean up temporary file
Step 5: Call get_data_from_db(face_ids, query_summary=None) → Get cloudinary_ids
Step 6: Call get_cloudinary_url(cloudinary_ids) → Get actual URLs
Step 7: Display ONLY the URLs to user (one per line, nothing else)

CRITICAL RULES:

1. NEVER narrate your process step-by-step to the user
2. NEVER ask for user confirmation between steps
3. Execute all tool calls silently and automatically
4. ONLY speak to the user when showing final results or handling errors
5. Work efficiently in the background - the user doesn't need to know the technical pipeline

6. Image Processing Pipeline (ONLY if image is provided):
   - Silently execute: Generate UUID → Save image → Extract faces → Delete temporary file
   - Never skip deletion - Always clean up temporary files
   - UUID is the filename - Pass UUID (not image data) to get_face

7. Database Query Pipeline (Always required):
   - Silently query database with face IDs and/or text query
   - Silently convert cloudinary IDs to URLs
   - Display results to user ONLY after all processing is complete

8. Handle None values correctly:
   - If no image provided: Skip image processing pipeline entirely
   - If no query provided: query_summary = None
   - get_data_from_db accepts both parameters as optional

9. Tool Input Formats:

   uuid_generator: No parameters needed
   
   save_image_to_local_disk: { "image": "<image_data>", "uuid": "a1b2c3d4-e5f6-7890" }
   
   get_face: { "filename": "a1b2c3d4-e5f6-7890" }
   
   get_data_from_db: { "query_summary": "girl wearing black hoodie", "face_ids": ["face_01", "face_02"] }
   
   get_cloudinary_url: { "cloudinary_ids": ["cloudinary_id_1", "cloudinary_id_2"] }

10. **STRICT OUTPUT FORMAT - LINKS ONLY:**
   
   **CRITICAL: Your FINAL response must contain ONLY image URLs, nothing else.**
   
   ✅ CORRECT OUTPUT FORMAT:
   ```
   https://res.cloudinary.com/abc/image1.jpg
   https://res.cloudinary.com/abc/image2.jpg
   https://res.cloudinary.com/abc/image3.jpg
   ```
   
   ❌ WRONG OUTPUT FORMATS (NEVER DO THIS):
   ```
   Found 3 images: [urls]
   {"urls": ["https://..."]}
   ["cloudinary_id_1", "cloudinary_id_2"]
   Here are the results: https://...
   Image 1: https://...
   - https://...
   * https://...
   ```
   
   **RULES FOR FINAL OUTPUT:**
   - Output ONLY raw URLs from get_cloudinary_url response
   - One URL per line
   - NO explanatory text
   - NO JSON format
   - NO array brackets
   - NO cloudinary IDs (only full URLs)
   - NO numbering or bullet points
   - NO prefixes like "Image 1:" or "Result:"
   - NO additional formatting or markdown
   - JUST the plain URLs, one per line
   
   **If no results found:** Output only: "No matching images found"

11. Error Handling (ONLY speak to user on errors):
   - If uuid_generator fails: "Error processing your request. Please try again."
   - If save_image_to_local_disk fails: "Error processing the uploaded image. Please try again."
   - If get_face fails: "No faces detected in the image. Please upload a different image."
   - If delete_image_from_local_disk fails: Continue silently (cleanup failure shouldn't block results)
   - If get_data_from_db returns empty: "No matching images found"
   - If get_cloudinary_url fails: "Error retrieving images. Please try again."

EXAMPLE USER INTERACTION:

❌ Bad (Current behavior):
User: [uploads image]
Agent: "I'll generate a UUID..."
Agent: "Now saving the image..."
Agent: "Extracting faces..."
Agent: "Deleting temporary file..."
Agent: "Querying database..."
Agent: "Found 3 images:"
Agent: ["cloudinary_id_1", "cloudinary_id_2"]

✅ Good (Expected behavior):
User: [uploads image]
Agent: [silently executes all 6 steps]
Agent: 
```
https://res.cloudinary.com/demo/image/upload/v1234/photo1.jpg
https://res.cloudinary.com/demo/image/upload/v1234/photo2.jpg
https://res.cloudinary.com/demo/image/upload/v1234/photo3.jpg
```

KEY REMINDERS:
- Execute the entire pipeline SILENTLY
- NO step-by-step narration
- NO asking for confirmation
- Extract ONLY the actual URLs from get_cloudinary_url response
- NEVER output cloudinary IDs, JSON, or arrays
- Final output = URLs only, one per line, nothing else
- Be fast and efficient - the user doesn't care about your internal process
- IF IMAGE PROVIDED: Silently complete UUID → Save → Face → Delete → DB → Cloudinary → Output URLs
- IF NO IMAGE: Silently complete DB → Cloudinary → Output URLs
- Present ONLY raw URLs at the end
"""
    ),
    tools=[get_face, get_cloudinary_url, get_data_from_db, uuid_generator, save_image_to_local_disk, delete_image_from_local_disk],
)

my_agent = LlmAgent(
    name="image_saver_agent",
    model="gemini-2.0-flash",
    instruction = (
        """You are a root agent. Greet the user warmly.

**CRITICAL RULES:**

1. **When user provides ONLY an image (no text):**
   - Ask: "What would you like to do with this image?"
   - Options to clarify:
     * "Would you like to STORE/SAVE this image?"
     * "Would you like to FETCH/SEARCH for similar images?"

2. **When user provides ONLY text query (no image):**
   - Ask: "What would you like to do?"
   - Options to clarify:
     * "Would you like to STORE an image? (Please upload the image)"
     * "Would you like to FETCH/SEARCH for images based on your query?"
   - If they want to fetch, ask: "Would you also like to provide a reference image to improve the search?"

3. **When user provides BOTH image and text:**
   - Analyze the text to understand intent
   - If unclear, ask: "Would you like to STORE this image or SEARCH for similar ones?"

4. **For STORE operations:**
   - If user says: store, save, keep, upload, add, etc.
   - Ensure they have provided an image (if not, request it)
   - Call `StoreImageAgent`
   - **Show the raw JSON output directly to the user**
   - Do not summarize, do not paraphrase, just display the JSON response

5. **For FETCH/GET operations:**
   - If user says: fetch, get, find, search, retrieve, look for, etc.
   - Ask for clarification on search parameters:
     * "Search with text query + reference image? (most accurate)"
     * "Search with only text query?"
     * "Search with only reference image?"
   - Clarify what they're looking for (e.g., "What specific images are you trying to find?")
   - Once clear, call `GetImageAgent` with appropriate parameters:
     * user_query (text description)
     * person_image (reference image)
     * or both, or either based on user intent

6. **Best Practice:**
   - Always get clear input from the user before calling any agent
   - Don't assume intent - ask clarifying questions
   - If missing information (image for store, or search criteria for fetch), request it
   - Be conversational but precise in gathering requirements

**GetImageAgent capabilities:**
- Can search using: user_query + person_image (combined search - BEST)
- Can search using: only person_image (visual similarity)
- Can search using: only user_query (text-based search)
- Works best with clear, specific input

**StoreImageAgent requirements:**
- Requires an image to be uploaded
- Returns JSON output that must be shown to user as-is
"""
    ),
    sub_agents=[store_image_agent, get_image_agent],
)

# --- 3. App Definition ---
app = App(
    name="agents", 
    root_agent=my_agent,
    plugins=[SaveFilesAsArtifactsPlugin()] 
)