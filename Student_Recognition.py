"""
live_student_recognition.py
========================================================================
My Eye Pro \u2014 Folder Enrollment + Live Video Recognition

WHAT THIS DOES
--------------
1. Scans STUDENTS_DIR (one subfolder per student) and enrolls every
   student it finds \u2014 no manual input. Each student is assigned a
   random, unique ARID number in the format "2023-ARID-XXXX".
2. Trains a classifier on the enrolled students' face embeddings.
3. Runs VIDEO_PATH through face detection + recognition and opens a
   LIVE preview window: every face is boxed and labeled with the
   matched student's name, ARID and confidence, and a details panel
   in the corner lists everyone currently recognized on screen.

FOLDER LAYOUT EXPECTED
-----------------------
students/
    Ali Raza/
        img1.jpg
        img2.jpg
    Ahmed Khan/
        photo1.png
        photo2.png
        photo3.png

Folder name = student name. Any number of photos per student (more
angles = a more robust match). A photo with zero or more than one
detected face is skipped with a warning, not a hard failure.

MODEL CHOICE (best accuracy)
-----------------------------
- Face detection + embedding: InsightFace's "buffalo_l" pack \u2014 SCRFD
  detector + ArcFace (ResNet-100) recognizer. ArcFace's angular-margin
  embeddings are what make multiple photos of the same person cluster
  tightly, which is what the classifier below needs.
- Classifier: a linear SVM (scikit-learn) trained on every enrolled
  student's embeddings \u2014 the actual "model training" step. An SVM is
  a *closed-set* classifier (it always picks its closest known class),
  so a face is only accepted as recognized when BOTH the SVM's
  probability AND the cosine similarity to that student's own centroid
  clear their thresholds. That second check is what lets the system
  say "Unknown" for someone unenrolled instead of confidently
  misnaming them.
- Tracking: a small IoU tracker, used only to keep a stable on-screen
  ID per face across frames.

INSTALL
-------
pip install insightface onnxruntime opencv-python numpy scikit-learn
(use onnxruntime-gpu instead of onnxruntime if you have a CUDA GPU)

CONFIGURE & RUN
----------------
Edit STUDENTS_DIR and VIDEO_PATH below, then:
    python live_student_recognition.py
Press 'q' in the preview window to stop early.
========================================================================
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.svm import SVC

# ------------------------------------------------------------------ #
# CONFIG — edit these for your setup
# ------------------------------------------------------------------ #

STUDENTS_DIR = "students"          # folder of student photos, one subfolder per student
VIDEO_PATH = "lecture.mp4"         # input video to recognize (use 0 for a live webcam instead)
LIVE_LOG_PATH = "live_attendance_log.jsonl"  # every recognized face gets one line logged here

DET_SIZE = (640, 640)              # detector input size — raise for small/far faces
ARID_YEAR = datetime.now().year

SVM_PROB_THRESHOLD = 0.55          # classifier confidence needed to accept a match
COSINE_THRESHOLD = 0.42            # cosine similarity to that student's own centroid
                                    # (both must pass — see module docstring)

IOU_MATCH_THRESHOLD = 0.3
TRACK_MAX_AGE = 15
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("live_student_recognition")


def build_face_model(use_gpu: bool = True) -> FaceAnalysis:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    # allowed_modules skips 3D landmarks, 2D landmarks, and gender/age estimators
    model = FaceAnalysis(
        name="buffalo_l",
        providers=providers,
        allowed_modules=["detection", "recognition"]
    )
    model.prepare(ctx_id=0 if use_gpu else -1, det_size=DET_SIZE)
    logger.info("Loaded InsightFace buffalo_l (detection + recognition only)")
    return model


# ------------------------------------------------------------------ #
# Enrolled student + in-memory store
# ------------------------------------------------------------------ #

@dataclass
class EnrolledStudent:
    identity_id: str
    name: str
    arid_no: str
    angle_embeddings: List[np.ndarray]   # one per usable photo found for this student
    centroid: np.ndarray                 # mean, re-normalised — used for the cosine check


def generate_random_arid(existing: set, year: int = ARID_YEAR) -> str:
    """Random (not sequential) ARID in the form '2023-ARID-0000'..'2023-ARID-9999',
    retrying on the rare collision with an already-used number."""
    while True:
        candidate = f"{year}-ARID-{random.randint(0, 9999):04d}"
        if candidate not in existing:
            return candidate


class FaceMemoryStore:
    def __init__(self):
        self.students: Dict[str, EnrolledStudent] = {}
        self._next_index = 1

    def used_arids(self) -> set:
        return {s.arid_no for s in self.students.values()}

    def add_from_embeddings(self, name: str, embeddings: List[np.ndarray]) -> EnrolledStudent:
        centroid = np.mean(embeddings, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        student = EnrolledStudent(
            identity_id=f"student_{self._next_index:04d}",
            name=name,
            arid_no=generate_random_arid(self.used_arids()),
            angle_embeddings=[e.astype(np.float32) for e in embeddings],
            centroid=centroid.astype(np.float32),
        )
        self._next_index += 1
        self.students[student.identity_id] = student
        logger.info("Enrolled %-20s -> %s  (%d photo[s])", student.name, student.arid_no, len(embeddings))
        return student

    def match_by_cosine(self, embedding: np.ndarray) -> Tuple[Optional[EnrolledStudent], float]:
        """Fallback matcher used when there are fewer than 2 students (too few to train an SVM)."""
        if not self.students:
            return None, 0.0
        best_student, best_sim = None, -1.0
        for student in self.students.values():
            sim = float(np.dot(embedding, student.centroid))
            if sim > best_sim:
                best_student, best_sim = student, sim
        return best_student, best_sim


# ------------------------------------------------------------------ #
# Batch enrollment from a folder of photos
# ------------------------------------------------------------------ #

def enroll_from_folder(store: FaceMemoryStore, model: FaceAnalysis, students_dir: str) -> None:
    root = Path(students_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"'{students_dir}' not found. Expected one subfolder per student, e.g. "
            f"{students_dir}/Ali Raza/photo1.jpg"
        )

    student_folders = sorted(p for p in root.iterdir() if p.is_dir())
    if not student_folders:
        raise ValueError(f"No student subfolders found inside '{students_dir}'.")

    for folder in student_folders:
        name = folder.name
        image_paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not image_paths:
            logger.warning("No photos found for '%s' — skipped.", name)
            continue

        # ---> PASTE STARTS HERE <---
        embeddings = []
        for path in image_paths:
            img = cv2.imread(str(path))
            if img is None:
                logger.warning("Could not read %s — skipped.", path)
                continue

            faces = model.get(img)
            if not faces:
                logger.warning("%s: no face detected — skipped.", path.name)
                continue

            # If multiple faces appear (e.g. background people), pick the largest face
            target_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            embeddings.append(target_face.normed_embedding)

        if not embeddings:
            logger.warning("No usable photos for '%s' — student not enrolled.", name)
            continue
        # ---> PASTE ENDS HERE <---

        store.add_from_embeddings(name, embeddings)

    if not store.students:
        raise RuntimeError("No students were successfully enrolled — check your photos and folder layout.")


# ------------------------------------------------------------------ #
# Trained classifier (this is the "model training" step)
# ------------------------------------------------------------------ #

class FaceClassifier:
    def __init__(self):
        self.svm: Optional[SVC] = None

    def train(self, store: FaceMemoryStore) -> None:
        X, y = [], []
        for identity_id, student in store.students.items():
            for emb in student.angle_embeddings:
                X.append(emb)
                y.append(identity_id)

        if len(store.students) < 2:
            self.svm = None
            logger.warning(
                "Only %d student enrolled — need at least 2 to train an SVM. "
                "Using cosine-similarity matching instead.", len(store.students)
            )
            return

        self.svm = SVC(kernel="linear", C=1.0, probability=True)
        self.svm.fit(np.array(X), y)
        logger.info("Trained SVM on %d embeddings across %d students", len(X), len(store.students))

    def predict(self, embedding: np.ndarray, store: FaceMemoryStore) -> Tuple[Optional[EnrolledStudent], float]:
        """Returns (matched_student_or_None, confidence). Requires BOTH the SVM
        probability and the cosine similarity to the predicted student's centroid
        to clear their thresholds — see module docstring for why."""
        if self.svm is not None:
            probs = self.svm.predict_proba([embedding])[0]
            best_idx = int(np.argmax(probs))
            identity_id = self.svm.classes_[best_idx]
            prob = float(probs[best_idx])
            student = store.students[identity_id]
            cosine_sim = float(np.dot(embedding, student.centroid))
            if prob >= SVM_PROB_THRESHOLD and cosine_sim >= COSINE_THRESHOLD:
                return student, prob
            return None, prob
        else:
            student, sim = store.match_by_cosine(embedding)
            if student is not None and sim >= COSINE_THRESHOLD:
                return student, sim
            return None, sim


# ------------------------------------------------------------------ #
# Lightweight IoU tracker (stable on-screen IDs across frames)
# ------------------------------------------------------------------ #

def _iou(box_a, box_b) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class SimpleIOUTracker:
    def __init__(self, iou_threshold: float = IOU_MATCH_THRESHOLD, max_age: int = TRACK_MAX_AGE):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._tracks: Dict[int, Dict] = {}
        self._next_id = 1

    def update(self, boxes: List[Tuple[float, float, float, float]]) -> List[int]:
        assigned = [None] * len(boxes)
        candidates = []
        for i, box in enumerate(boxes):
            for tid, info in self._tracks.items():
                score = _iou(box, info["bbox"])
                if score >= self.iou_threshold:
                    candidates.append((score, i, tid))
        candidates.sort(key=lambda c: c[0], reverse=True)

        used = set()
        for score, i, tid in candidates:
            if assigned[i] is None and tid not in used:
                assigned[i] = tid
                used.add(tid)
        for i in range(len(boxes)):
            if assigned[i] is None:
                assigned[i] = self._next_id
                self._next_id += 1

        seen = set(assigned)
        for tid in list(self._tracks.keys()):
            if tid not in seen:
                self._tracks[tid]["age"] += 1
                if self._tracks[tid]["age"] > self.max_age:
                    del self._tracks[tid]
        for i, box in enumerate(boxes):
            self._tracks[assigned[i]] = {"bbox": box, "age": 0}
        return assigned


# ------------------------------------------------------------------ #
# Live preview — box + label on every face, plus a details panel
# ------------------------------------------------------------------ #

COLOR_KNOWN = (79, 158, 46)      # BGR — green, an enrolled student
COLOR_UNKNOWN = (60, 60, 200)    # BGR — red, unmatched face
PANEL_BG = (24, 24, 24)


def _draw_face_label(frame, bbox, text: str, color) -> None:
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    label_y1 = max(0, y1 - th - 12)
    cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 10, y1), color, -1)
    cv2.putText(frame, text, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def _draw_details_panel(frame, current_matches: List[Tuple[EnrolledStudent, float]]) -> None:
    """Lists every student recognized in THIS frame, top-left corner."""
    x0, y0 = 12, 12
    line_h = 22
    rows = max(1, len(current_matches))
    panel_w, panel_h = 360, 34 + rows * line_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), PANEL_BG, -1)
    frame[:] = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    cv2.putText(frame, "DETECTED STUDENTS", (x0 + 10, y0 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if not current_matches:
        cv2.putText(frame, "— none in frame —", (x0 + 10, y0 + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
        return

    for i, (student, confidence) in enumerate(current_matches):
        y = y0 + 44 + i * line_h
        text = f"{student.name}  |  {student.arid_no}  |  {confidence*100:.0f}%"
        cv2.putText(frame, text, (x0 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1, cv2.LINE_AA)


def run_live_recognition(store: FaceMemoryStore, classifier: FaceClassifier, model: FaceAnalysis, video_source) -> None:
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_source}")

    session_id = f"live_{int(time.time())}"
    tracker = SimpleIOUTracker()
    seen_tracks: set = set()
    window = "My Eye Pro — Live Student Recognition (press 'q' to quit)"

    with open(LIVE_LOG_PATH, "a") as log_file:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            faces = model.get(frame)
            boxes = [tuple(f.bbox.tolist()) for f in faces]
            track_ids = tracker.update(boxes)

            current_matches = []
            for face, track_id in zip(faces, track_ids):
                student, confidence = classifier.predict(face.normed_embedding, store)
                if student is not None:
                    label = f"{student.name}  {student.arid_no}  {confidence*100:.0f}%"
                    _draw_face_label(frame, face.bbox, label, COLOR_KNOWN)
                    current_matches.append((student, confidence))
                    if track_id not in seen_tracks:
                        seen_tracks.add(track_id)
                        event = {
                            "session_id": session_id,
                            "track_id": track_id,
                            "identity_id": student.identity_id,
                            "identity_name": student.name,
                            "arid_no": student.arid_no,
                            "role": "student",
                            "match_confidence": round(confidence, 4),
                            "resolved_ts": round(time.time(), 3),
                        }
                        log_file.write(json.dumps(event) + "\n")
                        log_file.flush()
                else:
                    _draw_face_label(frame, face.bbox, "Unknown", COLOR_UNKNOWN)

            _draw_details_panel(frame, current_matches)
            cv2.imshow(window, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyWindow(window)


# ------------------------------------------------------------------ #
# Run
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    model = build_face_model(use_gpu=True)

    store = FaceMemoryStore()
    enroll_from_folder(store, model, STUDENTS_DIR)

    classifier = FaceClassifier()
    classifier.train(store)

    run_live_recognition(store, classifier, model, VIDEO_PATH)
