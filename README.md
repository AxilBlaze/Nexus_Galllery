# 📸 Nexus Smart AI Gallery

### *Stop Scrolling. Start Finding.*

[![Watch the Demo](https://img.youtube.com/vi/efWvyoxhFoo/0.jpg)](https://youtu.be/efWvyoxhFoo)

> **Click above to watch the 3-minute project walkthrough and demo.**

---

## 🔴 The Problem — The “Digital Landfill”

We capture thousands of photos.
But when we want **that one photo**, we’re stuck scrolling endlessly.

✔ Search engines don’t understand what’s *inside* images
✔ No context like *“me in the blue dress”* or *“Sandeep eating pizza”*
✔ No organization unless we manually tag everything

We hoard memories we can’t even find anymore.

---

## 🟢 Solution — Nexus Smart AI Gallery

An **AI-powered intelligent gallery** that understands images like humans do.

By combining:

| Capability                   | Powered By         |
| ---------------------------- | ------------------ |
| Semantic scene understanding | Gemini             |
| Face identification          | FaceNet512         |
| Fast vector search           | Qdrant             |
| Cloud storage                | Cloudinary         |
| Full automation              | Multi-Agent System |

You can search with meaning:

> “This person 🧑 eating pizza 🍕 at night 🌙”

and it appears — instantly.

---

## 💡 Why Agents?

Static scripts cannot manage visual understanding and identity resolution together.
Our **multi-agent architecture** allows reasoning, decision making, and tool delegation.

---

## 🛠 Tools & Responsibilities

| Tool Name               | Purpose                                |
| ----------------------- | -------------------------------------- |
| Generate UUID Tool      | Generate unique IDs for each new image |
| Save in Local Tool      | Temporarily store original image       |
| Get Summary Tool        | Gemini tool for semantic understanding |
| Get Face ID Tool        | FaceNet512 identity vector extraction  |
| Save in Cloudinary Tool | Upload permanent storage               |
| Delete from Local Tool  | Cleanup local temp copies              |
| Save in Qdrant Tool     | Store vectors + metadata for retrieval |
| Search Tool             | Text/face/hybrid vector search         |

---

Absolutely — here is a **clean, professional, and structured point-wise workflow explanation** perfectly matching your diagram and readable for the README:

---

## 🔄 Multi-Agent Workflow (Step-by-Step)

Our system uses a **Root Agent** that intelligently decides which sub-agent to activate — based on whether the user wants to **store** an image or **search** for one.

---

### 📥 A) Image Ingestion Flow (When user uploads a new image)

1️⃣ **Root Agent** receives user request to store image </br>

2️⃣ It delegates the task to the **Save Image Sub-Agent** </br>

3️⃣ Save Image Sub-Agent orchestrates multiple tools:</br>

* 🟠 **Generate UUID Tool** → creates a unique ID
* 🟠 **Save Image in Local Tool** → temporarily stores file 
* 🟠 **Get Summary Tool (Gemini)** → semantic scene understanding
* 🟠 **Get Face ID Tool (FaceNet512)** → extract identity embeddings
* 🟠 **Save in Cloudinary Tool** → uploads final image
* 🟠 **Delete From Local Tool** → cleanup to reduce storage use

4️⃣ When all metadata is ready
→ Save Image Sub-Agent passes to **Save in DB Sub-Agent**

5️⃣ Save in DB Sub-Agent calls:

* 🟠 **Save in Qdrant DB Tool** → store:

  ```json
  {
    "uuid": "...",
    "summary": "...",
    "summary_vec": [...],
    "face_ids": [...],
    "cloudinary_id": "..."
  }
  ```

📌 Result:
The photo is now fully searchable using text or identity.

---


### 🔍 B) Image Search Flow (Text / Face / Hybrid Search)


1️⃣ **Root Agent** detects search request </br>

2️⃣ Delegates to **Search Image Sub-Agent**

3️⃣ When search includes a reference face:

* 🟠 Save Image in Local Tool (temp import)
* 🟠 Get Face ID Tool → extract embeddings
* 🟠 Delete From Local Tool → cleanup

4️⃣ Search Image Sub-Agent calls:

* 🟠 **Search DB Tool (Qdrant hybrid search)**
  with:

  ```json
  {
    "query": "text query",
    "face_ids": [...]
  }
  ```

5️⃣ Qdrant returns:

*  **Cloudinary IDs** of matching images
* Results displayed instantly to user 🎯


Supports:

| Type      | Example                         |
| --------- | ------------------------------- |
| Text-only | “beach sunset”                  |
| Face-only | Upload a photo of a person      |
| Hybrid    | Upload a photo of a person + “eating pizza” |

Hybrid = **Face match** ∩ **Semantic match**
(A truly human-like “memory recall”)

---

📌 Architecture Diagram
![Workflow Diagram](https://res.cloudinary.com/dqcgbfxki/image/upload/v1764365450/Gemini_Generated_Image_1zjiq81zjiq81zji_q4smnx.png)

---

## 4️⃣ Setup & Usage

### 🔧 Prerequisites

Install & configure:

* Python **3.10+**
* Git
* Google Gemini API Key → Google AI Studio
* Cloudinary Account → for image storage
* Qdrant Cloud or Local Docker instance

---

### 📥 Installation Guide

#### Step 1 — Clone Repository

```bash
git clone https://github.com/meetbikhani/Nexus_Galllery.git .
```

#### Step 2 — Create Virtual Environment

```bash
python -m venv venv
```

#### Step 3 — Activate It

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

#### Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 5 — Configure Environment Variables

```bash
cd agents
```

Create `.env` inside **agents** folder:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_cloudinary_api_key
CLOUDINARY_API_SECRET=your_cloudinary_api_secret
GEMINI_API_KEY=your_gemini_api_key_here
QDRANT_URL=your_qdrant_instance_url
QDRANT_API_KEY=your_qdrant_api_key
```

👉 Notes:

* `GOOGLE_API_KEY` & `GEMINI_API_KEY` → same value
* No spaces in `.env` formatting

```bash
cd ..
```

#### Step 7 — Run the Web App

```bash
adk web
```


---

## 🔍 Usage Examples

### 1️⃣ Store Images (Auto Analyze Gallery)

Upload multiple files →
AI will automatically:

✔ Identify faces
✔ Summarize scenes
✔ Upload to Cloudinary
✔ Store embeddings in Qdrant

---

### 2️⃣ Text Search

Just describe the memory:

> “birthday celebration with cake”

Returns that exact moment 🎉

---

### 3️⃣ Face Search

Upload a reference face:

> “Show all photos of this person”

Finds every image of them — even years apart!

---

### 4️⃣ Hybrid Search

Upload face + add context:

> “This person on beach”
> “This person eating pizza”

→ AI intersects identity & meaning
→ Pinpoint-accurate recall 🔍

---

## 🧪 Troubleshooting

| Issue             | Fix                                    |
| ----------------- | -------------------------------------- |
| Face not detected | Use clearer front-facing image         |
| Qdrant errors     | Ensure cluster running / Docker active |
| Invalid API key   | Check `.env` config correctness        |

---

## 🤝 Contributors

Built with ❤️ by the **Nexus Gallery Team**

---

