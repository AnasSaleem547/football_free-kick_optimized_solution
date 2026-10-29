# Football Free-Kick Detector

Detects stationary-ball kick attempts in videos, creates annotated kick clips, and uploads them to Cloudinary. The API returns the confirmed kick count and clip URLs.

[Postman API documentation](https://documenter.getpostman.com/view/53820649/2sBYHVwevF)

## Requirements

- Git and Conda.
- Cloudinary cloud name, API key and API secret.
- Internet access for installation, model downloads and clip uploads.

Runs on CPU or a supported NVIDIA GPU. GPU execution requires a compatible driver and CUDA-enabled PyTorch. CPU processing can be slow.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/AnasSaleem547/football_free-kick_optimized_solution.git
cd football_free-kick_optimized_solution
```

### 2. Install dependencies

```bash
conda env create -f environment.yml
conda activate football-kick
```

The environment includes Python 3.11, FFmpeg and the Python dependencies. FFmpeg must provide FFprobe and the `libx264` encoder.

### 3. Configure Cloudinary

Create a `.env` file in the repository root and replace these placeholders:

```dotenv
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
FOOTBALL_DEVICE=auto
```

Do not commit your `.env` file. `auto` selects an available CUDA GPU or falls back to CPU. Set `FOOTBALL_DEVICE=cpu` to require CPU execution.

### 4. Start the API

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Open **http://127.0.0.1:8000/docs** to try the API interactively. Keep one worker because job coordination runs within a single process.

The default models download when the first video is processed and are cached for reuse. The first job can therefore take longer.

## Usage

### 1. Upload a video

```bash
curl -X POST http://127.0.0.1:8000/video/upload \
  -F "file=@/absolute/path/to/video.mp4"
```

Supported formats: **MP4, MOV, AVI, WebM and MKV**, provided the server can validate and decode the file. The default upload limit is **2 GiB**.

POST returns **HTTP 202** with a session ID:

```json
{"session_id":"session_0123456789abcdef0123456789abcdef"}
```

### 2. Get progress and results

Replace `YOUR_SESSION_ID` with the ID returned by POST:

```bash
curl http://127.0.0.1:8000/video/status/YOUR_SESSION_ID
```

Repeat GET every few seconds. While processing:

```json
{
  "stage": "detecting",
  "current_frame": 120,
  "total_frames": 600,
  "progress_pct": 20.0
}
```

Percentage applies to the current stage. Finishing detection does not mean rendering and uploads have finished.

When complete:

```json
{
  "total_kicks": 2,
  "cloudinary_urls": [
    "https://res.cloudinary.com/YOUR_CLOUD/video/upload/kick_1.mp4",
    "https://res.cloudinary.com/YOUR_CLOUD/video/upload/kick_2.mp4"
  ]
}
```

These URLs are examples; actual links are returned in kick order. No confirmed kicks returns `{"total_kicks":0,"cloudinary_urls":[]}`. Failed processing or incomplete uploads return an HTTP error instead of a successful completed result.

For **Postman**, send POST to `/video/upload` with **Body → form-data → file**, using type **File**. Then poll GET `/video/status/{session_id}`. Let Postman set the multipart Content-Type header.

## Notes

- Counts confirmed kick attempts, including shots that miss the goal. Goal success is not classified, and strong passes can resemble shots.
- Creates kick clips rather than a full annotated source video. Generated clips have no audio.
- Stores job files under `videosofkicks/`. Clips and results remain locally; source videos are deleted after local completion by default. There is no automatic cleanup of retained results.
- No built-in authentication. Add access control before exposing the API publicly.
