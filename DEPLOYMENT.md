# How to Deploy CRIMSON-India

This guide covers deploying the **Streamlit dashboard** (`app.py`), which is the main user-facing part of the project. The CLI pipeline (`main.py`) is typically run locally or on a scheduler; the dashboard can be deployed to the cloud.

---

## What You’re Deploying

- **App**: Streamlit dashboard at `app.py` (runs with `streamlit run app.py`).
- **Dependencies**: In `requirements.txt` (Python 3.11+, PyTorch, transformers, Streamlit, etc.).
- **Local paths used by the app**:
  - `data/` — `labeled_news.csv`, `raw_news.csv`, `ncrb_stats.csv`, `.last_updated`
  - `plots/` — `crime_trends.png`
  - `models/` — `baseline_svm.pkl` (used for “Fetch Live News” classification)

The app works without pre-trained models (it will show “no labels” until you run the pipeline or use “Fetch Live News,” which needs the SVM model). For a full experience, either run `main.py --use-synthetic` once to generate data + SVM model, or deploy with pre-built artifacts (see below).

---

## Option 1: Streamlit Community Cloud (easiest)

Best for: quick public demo, no server management.

1. **Push your code to GitHub** (no `venv/`, no `__pycache__/`; add them to `.gitignore`).

2. **Go to [share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.

3. **New app** → choose repo `crimson`, branch `main`, **Main file path**: `app.py`.

4. **Advanced settings** (optional):
   - Python version: **3.11**.
   - Add **packages** if needed (Streamlit usually picks up `requirements.txt`).

5. **Deploy.**  
   First build can take 10–15 minutes (PyTorch/transformers are large).

**Limitations:**

- **Ephemeral filesystem**: Anything written to `data/` or `models/` is lost on restart. So:
  - “Fetch Live News” will work for that session but data won’t persist.
  - To have pre-loaded data or the SVM model, you’d need to either:
    - Check in minimal `data/` and `models/` (e.g. `labeled_news.csv`, `baseline_svm.pkl`) in the repo, or
    - Use Streamlit’s “Secrets” + an external store (e.g. S3) and load/save there (requires code changes).
- Free tier has memory limits; heavy transformer inference may need a paid tier or another option.

---

## Option 2: Docker (any VPS or cloud)

Use this for full control, persistent disk, or when you want to run both the dashboard and the pipeline on the same machine.

### 2.1 Create a Dockerfile

Create `Dockerfile` in the project root:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system deps if needed (e.g. for some ML libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Run from project root so paths data/, plots/, models/ resolve
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 2.2 Build and run locally

```powershell
docker build -t crimson-dashboard .
docker run -p 8501:8501 crimson-dashboard
```

Open `http://localhost:8501`.

### 2.3 Deploy to a server

- Copy the image to your server (or use a registry) and run the same `docker run` (or use `docker-compose`).
- To **persist data** between restarts, mount volumes:

  ```powershell
  docker run -p 8501:8501 -v crimson-data:/app/data -v crimson-models:/app/models -v crimson-plots:/app/plots crimson-dashboard
  ```

- Optionally run the pipeline once to populate data/models:

  ```powershell
  docker run --rm -v crimson-data:/app/data -v crimson-models:/app/models -v crimson-plots:/app/plots crimson-dashboard python main.py --use-synthetic
  ```

  Then start the dashboard with the same volumes so it sees the same `data/`, `models/`, `plots/`.

---

## Option 3: Railway / Render / Fly.io

These platforms can run a Dockerfile or a “Python + start command” setup.

- **Start command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`  
  (Use `$PORT` if the platform sets it; otherwise use `8501` and expose that port.)

- **Python version**: 3.11.

- **Persistent storage**:  
  If the platform offers volumes, mount them at `./data`, `./models`, `./plots` so the app and pipeline share the same paths. Otherwise you get the same ephemeral behavior as Streamlit Cloud.

- **Resource**: Prefer at least 2 GB RAM (and more if you ever add transformer inference in the app).

---

## Pre-deployment checklist

1. **`.gitignore`**  
   Ignore `venv/`, `__pycache__/`, `.env`, and optionally large artifacts (e.g. `models/saved_muril/`, `models/saved_xlmroberta/`) if you don’t want them in the repo.

2. **Secrets**  
   The app doesn’t use API keys today; if you add any, use env vars or Streamlit Secrets and never commit them.

3. **Run from project root**  
   All paths are relative to the current working directory. Ensure the process starts with CWD = project root (e.g. `WORKDIR /app` in Docker and `CMD` from there).

4. **Optional: base directory config**  
   For flexibility you could introduce a single env var (e.g. `CRIMSON_BASE_DIR`) and resolve `data/`, `plots/`, `models/` relative to it; default to `.` so local runs stay unchanged.

---

## Summary

| Method                    | Best for              | Data/model persistence      |
|---------------------------|------------------------|-----------------------------|
| Streamlit Community Cloud | Easiest public demo   | No (ephemeral)              |
| Docker on VPS/cloud       | Full control, staging | Yes (with volumes)          |
| Railway / Render / Fly    | Managed app hosting   | Yes if they offer volumes   |

For a quick public link, use **Streamlit Community Cloud**. For persistent data and models and running the pipeline on the same box, use **Docker** (or a similar stack) with mounted volumes and run `main.py` once to populate `data/` and `models/`.
