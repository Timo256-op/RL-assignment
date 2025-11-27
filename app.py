# RL/api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from stable_baselines3 import PPO
import os
import uvicorn
import sys
import warnings

# --- Fix Gym warning ---
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Ensure correct module path ---
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from env.teacher_alloc_env import TeacherAllocEnv

# --- Load model path ---
DEFAULT_LOCAL_PATH = os.path.join("saved_models", "ppo_teacher_alloc.zip")
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_LOCAL_PATH)

app = FastAPI(title="Teacher Allocation RL API")

# Load environment and model once
env = TeacherAllocEnv()
model = None

try:
    print(f"Loading model from: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print("Model load failed:", e)


# ---------- Pydantic Request Model ----------
class AllocationRequest(BaseModel):
    students: list[int]             # length must match env.n_classes
    prev_coverage: list[float]      # must be between 0 and 1
    available_teachers: int


# ---------- Prediction Endpoint ----------
@app.post("/predict")
def predict(req: AllocationRequest):

    if model is None:
        raise HTTPException(status_code=503, detail="Model failed to load.")

    # --- Validate lengths ---
    n = env.n_classes
    if len(req.students) != n:
        raise HTTPException(
            status_code=422,
            detail=f"'students' must contain {n} values, got {len(req.students)}"
        )
    if len(req.prev_coverage) != n:
        raise HTTPException(
            status_code=422,
            detail=f"'prev_coverage' must contain {n} values, got {len(req.prev_coverage)}"
        )

    # --- Validate coverage range ---
    if any(c < 0 or c > 1 for c in req.prev_coverage):
        raise HTTPException(
            status_code=422,
            detail=f"'prev_coverage' values must be between 0 and 1."
        )

    # --- Build observation correctly ---
    obs_students = np.array(req.students) / max(1, env.demand_scale)
    obs_prev_cov = np.array(req.prev_coverage, dtype=np.float32)
    obs_teachers = np.array([req.available_teachers], dtype=np.float32)

    obs = np.concatenate([obs_students, obs_prev_cov, obs_teachers]).astype(np.float32)

    # PPO expects obs shape (13,)
    if obs.shape[0] != env.observation_space.shape[0]:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid observation dimension {obs.shape[0]}, expected {env.observation_space.shape[0]}"
        )

    # --- Get RL action ---
    action_idx, _ = model.predict(obs, deterministic=True)
    allocation = env.all_patterns[int(action_idx)].tolist()

    return {
        "action_index": int(action_idx),
        "allocation": allocation,
        "message": "Teacher allocation per class"
    }


# ---------- Run App ----------
if __name__ == "__main__":
    print(f"API ready at: http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
