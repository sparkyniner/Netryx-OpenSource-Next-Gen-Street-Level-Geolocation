<p align="center">
  <h1 align="center">NETRYX</h1>
  <p align="center"><strong>Open-Source Street-Level Geolocation Engine</strong></p>
  <p align="center">
    Upload any street photo. Get precise GPS coordinates.<br>
    Sub-50m accuracy. No landmarks needed. Runs entirely on your hardware.
  </p>
  <p align="center">
    <a href="#demos">Demos</a> · <a href="#how-it-works">How It Works</a> · <a href="#getting-started">Getting Started</a> · <a href="#usage">Usage</a> · <a href="#faq">FAQ</a>
  </p>
</p>

---

## What is Netryx?

Netryx is a local-first geolocation tool that identifies the exact GPS coordinates of any street-level photograph. Unlike reverse image search (which matches against uploaded web images), Netryx matches against **systematically crawled street-view panoramas** — meaning it works on any random street corner with zero internet presence.

The core pipeline combines three state-of-the-art computer vision models:

- **CosPlace** for global visual place recognition (retrieval)
- **ALIKED/DISK** for local feature extraction (keypoints)  
- **LightGlue** for deep feature matching (verification)

> **Google Lens searches the internet. Netryx searches the physical world.**

---

## Demos

| Use Case | Link |
|----------|------|
| Missile strike geolocation — Qatar (Feb 2026) | [Watch on YouTube](https://www.youtube.com/watch?v=Y_eC5VPypPU) |
| Conflict monitoring — Paris protests | [Watch on YouTube](https://www.youtube.com/watch?v=DV8vsoa5sLU) |
| Blind geolocation from a random photo — Paris | [Watch on YouTube](https://www.youtube.com/watch?v=N5Cx7j6qA7I) |
| Technical deep-dive: how the pipeline works | [Watch on YouTube](https://www.youtube.com/watch?v=KMbeABzG6IQ) |

---

## How It Works

Netryx uses a three-stage pipeline that progressively narrows from millions of candidates to a single precise match.

### Stage 1 — Global Retrieval (CosPlace)

Every street-view panorama in the index has been pre-processed into a 512-dimensional "fingerprint" using [CosPlace](https://github.com/gmberton/cosplace), a visual place recognition model trained on millions of geo-tagged images.

When you upload a query photo:

1. The system extracts a CosPlace descriptor from your image
2. It also extracts a descriptor from a horizontally-flipped version (catches reversed perspectives)
3. Both descriptors are compared against every entry in the index using cosine similarity
4. A radius filter (haversine distance) narrows candidates to your specified search area
5. The top 500–1000 most visually similar panorama views are returned as candidates

This stage runs in **under 1 second** regardless of index size — it's a single matrix multiplication.

### Stage 2 — Local Geometric Verification (ALIKED/DISK + LightGlue)

CosPlace finds places that *look similar*. Stage 2 proves they're the *same place* using geometric verification.

For each candidate:

1. The original panorama is downloaded from Google Street View (8 tiles, stitched)
2. A rectilinear crop is extracted at the indexed heading angle
3. **Multi-FOV crops** are generated at three fields of view (70°, 90°, 110°) to handle zoom mismatches between the query photo and the indexed view
4. [ALIKED](https://github.com/naver/alike) (on CUDA) or [DISK](https://github.com/cvlab-epfl/disk) (on MPS/CPU) extracts local keypoints and descriptors
5. [LightGlue](https://github.com/cvg/LightGlue) performs deep feature matching between query and candidate keypoints
6. **RANSAC** filters matches to keep only geometrically consistent correspondences (rejects false matches)
7. The candidate with the most verified inliers is the best match

This stage processes 300–500 candidates in **2–5 minutes** depending on your hardware.

### Stage 3 — Refinement

The initial match is good but not always optimal. Refinement improves it:

- **Heading refinement**: For the top 15 candidates, the system tests ±45° heading offsets at 15° steps across 3 FOVs. This catches cases where the indexed heading doesn't exactly match the query's viewing direction.
- **Spatial consensus**: Matches are clustered into 50m grid cells. If multiple candidates cluster in one area, that cluster is preferred over a single high-inlier outlier — reducing false positives.
- **Confidence scoring**: The system evaluates geographic clustering of top matches and uniqueness ratio (how much better the best match is vs. the runner-up at a different location).

### Ultra Mode (Optional)

For difficult images (night, blur, low texture), Ultra Mode adds:

- **LoFTR**: A detector-free dense matcher that finds correspondences without relying on keypoint detection. Handles blur and low-contrast scenes where ALIKED/DISK struggle.
- **Descriptor hopping**: If the initial match is weak (<50 inliers), the system extracts a CosPlace descriptor from the *matched panorama* (which is clean/high-quality) and re-searches the index. This often finds the exact right panorama that the degraded query missed.
- **Neighborhood expansion**: Searches all panoramas within 100m of the best match. The correct location is often one street node away from the CosPlace top match.

---

## Architecture

```
Query Image
    │
    ├── CosPlace descriptor extraction (512-dim fingerprint)
    ├── Flipped descriptor extraction
    │
    ▼
Index Search (cosine similarity, radius-filtered)
    │
    ├── Top 500 candidates ranked by visual similarity
    │
    ▼
Download Panoramas → Crop at 3 FOVs → Extract ALIKED/DISK keypoints
    │
    ├── LightGlue matching against query keypoints
    ├── RANSAC geometric verification
    │
    ▼
Heading Refinement (±45°, 3 FOVs, top 15 candidates)
    │
    ├── Spatial consensus clustering
    ├── Confidence scoring
    │
    ▼
📍 GPS Coordinates + Confidence Score
```

---

## Getting Started

### Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | macOS / Linux / Windows | macOS (M1+) or Linux with NVIDIA GPU |
| GPU VRAM | 4GB | 8GB+ |
| RAM | 8GB | 16GB+ |
| Storage | 10GB | 50GB+ (depends on indexed area size) |
| Internet | Required for indexing and searching | Broadband recommended |
| Python | 3.9+ | 3.10+ |

**GPU support:**
- **Mac**: MPS (Metal Performance Shaders) — M1/M2/M3/M4
- **NVIDIA**: CUDA — any GPU with 4GB+ VRAM
- **CPU**: Works but significantly slower

### Installation

```bash
# Clone the repository
git clone https://github.com/sparkyniner/netryx.git
cd netryx

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install kornia for Ultra Mode (LoFTR)
pip install kornia
```

### Gemini API Key (Optional — for AI Coarse mode)

If you want to use the AI-assisted blind geolocation feature:

1. Get a free API key from [Google AI Studio](https://aistudio.google.com)
2. Set it as an environment variable:
```bash
export GEMINI_API_KEY="your_key_here"
```

---

## Usage

### Launch the GUI

```bash
python test_super.py
```

### Step 1: Create an Index

Before searching, you need to index an area. This crawls Google Street View panoramas and extracts CosPlace fingerprints.

1. Select **Create** mode
2. Enter the center coordinates (latitude, longitude) of the area you want to index
3. Set the search radius (start with 0.5–1km for testing, 5–10km for production)
4. Set grid resolution (8 is a good default — higher = denser coverage)
5. Click **Create Index**

**Indexing time estimates:**

| Radius | Approx. Panoramas | Time (M2 Max) | Index Size |
|--------|-------------------|---------------|------------|
| 0.5 km | ~500 | 30 min | ~60 MB |
| 1 km | ~2,000 | 1–2 hours | ~250 MB |
| 5 km | ~30,000 | 8–12 hours | ~3 GB |
| 10 km | ~100,000 | 24–48 hours | ~7 GB |

The index is saved incrementally — if the process is interrupted, it resumes from where it left off on the next run.

### Step 2: Search

1. Select **Search** mode
2. Upload a street-level photo
3. Choose search method:
   - **Manual**: Enter coordinates + radius if you know approximately where the photo is from
   - **AI Coarse**: Let Gemini analyze visual clues (signs, architecture, vegetation) to guess the region — no prior knowledge needed
4. Click **Run Search**, then **Start Full Search**
5. Watch the real-time scanning visualization as candidates are evaluated
6. Result appears on the map with confidence score

### Ultra Mode

Enable the **Ultra Mode** checkbox for difficult images. This adds LoFTR dense matching, descriptor hopping, and neighborhood expansion. Significantly slower but catches matches that the standard pipeline misses.

---

## How the Index Works

All embeddings are stored in a single unified index. When you search, the radius filter automatically restricts results to the area you specify. This means:

- You can index Paris, then Tel Aviv, then London — all into the same index
- Searching with center=Paris, radius=5km only returns Paris results
- Searching with center=London, radius=10km only returns London results
- No city selection needed — coordinates + radius handle everything

**Data flow:**

```
Create Mode:
  Grid points → Google Street View API → Panoramas → Crops → CosPlace → cosplace_parts/*.npz

Auto-build:
  cosplace_parts/*.npz → cosplace_descriptors.npy + metadata.npz (searchable index)

Search Mode:
  Query image → CosPlace → Index search (radius-filtered) → Download candidates → ALIKED/DISK + LightGlue → Result
```

---

## Project Structure

```
netryx/
├── test_super.py          # Main application (GUI + indexing + search)
├── cosplace_utils.py      # CosPlace model loading and descriptor extraction
├── build_index.py         # Standalone high-performance index builder (for large datasets)
├── requirements.txt       # Python dependencies
├── cosplace_parts/        # Raw embedding chunks (created during indexing)
├── index/                 # Compiled searchable index
│   ├── cosplace_descriptors.npy   # All 512-dim descriptors
│   └── metadata.npz               # Coordinates, headings, panoid IDs
└── README.md
```

---

## Technical Details

### Models Used

| Model | Purpose | Paper |
|-------|---------|-------|
| [CosPlace](https://github.com/gmberton/cosplace) | Visual place recognition (global descriptor) | [CVPR 2022](https://arxiv.org/abs/2204.02287) |
| [ALIKED](https://github.com/naver/alike) | Local feature extraction (used on CUDA) | [IEEE TIP 2023](https://arxiv.org/abs/2304.03608) |
| [DISK](https://github.com/cvlab-epfl/disk) | Local feature extraction (used on MPS/CPU) | [NeurIPS 2020](https://arxiv.org/abs/2006.13566) |
| [LightGlue](https://github.com/cvg/LightGlue) | Deep feature matching | [ICCV 2023](https://arxiv.org/abs/2306.13643) |
| [LoFTR](https://github.com/zju3dv/LoFTR) | Detector-free dense matching (Ultra Mode) | [CVPR 2021](https://arxiv.org/abs/2104.00680) |

### Platform-Specific Behavior

| Feature | CUDA (NVIDIA) | MPS (Mac) | CPU |
|---------|--------------|-----------|-----|
| Feature extractor | ALIKED (1024 kp) | DISK (768 kp) | DISK (768 kp) |
| LightGlue flash attention | Enabled | Disabled | Disabled |
| LoFTR (Ultra Mode) | Full speed | CPU fallback for some ops | Full CPU |
| Indexing speed | Fastest | Good | Slow |

---

## FAQ

**Does accuracy decrease with larger search radius?**

No. CosPlace ranks by visual similarity across the entire index. Whether your index covers 1km or 50km, the top candidates are always the most visually similar. The radius filter simply excludes geographically irrelevant entries before ranking.

**How long does a search take?**

Typically 2–5 minutes on Apple Silicon, 1–3 minutes on NVIDIA GPU. The bottleneck is downloading panorama tiles from Google (network-bound), not the matching itself.

**Can I index an entire country?**

Not practical. A 10km radius produces ~100K panoramas and a ~7GB index. An entire country would require petabytes. Index specific cities or regions of interest (5–20km).

**My search found 0 candidates.**

Your search coordinates don't overlap with any indexed area. Either index the area first, or adjust your search center/radius to overlap with existing indexed data. Use "Show Coverage Map" to visualize what's been indexed.

**The match is slightly wrong (off by one block).**

Enable Ultra Mode. Neighborhood expansion searches all panoramas within 100m of the initial match, often finding the exact correct street node.

**Does this work with indoor photos?**

No. Netryx matches against outdoor street-view panoramas only. Indoor photos, aerial/satellite imagery, and close-up object photos will not produce meaningful results.

**Does this cost money?**

Netryx is free and open source. The optional AI Coarse feature requires a Gemini API key (free tier available from Google). Compute costs are your own hardware — no cloud services required.

---

## Built By

**Sairaj Balaji** — AI researcher and founder of PrismX. Grants from Microsoft and ElevenLabs. Featured in Fast Company and Deutsche Welle.

- [LinkedIn](https://www.linkedin.com/in/sairaj-balaji-7295b2246/)
  

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Legal Disclaimer

Netryx is designed for legitimate OSINT research, investigative journalism, human rights monitoring, disaster response, and academic research.

**User responsibility**: You are solely responsible for ensuring compliance with applicable laws, including the Google Maps/Street View Terms of Service and local privacy regulations.

**Do not use for**: Stalking, harassment, unauthorized surveillance, or any activity that violates the privacy or safety of individuals.

The developers of Netryx disclaim all liability for misuse.
