import base64
import io
import json
import os
import pickle
import threading
import secrets
import shutil
from datetime import datetime
from functools import wraps
from pathlib import Path

import cv2
import face_recognition
import numpy as np
import pandas as pd
from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session as flask_session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = Path(__file__).resolve().parent
INSTANCE_DIR = BASE_DIR / "instance"
DATABASE_PATH = INSTANCE_DIR / "attendance.db"
FACES_DIR = INSTANCE_DIR / "faces"
ENCODINGS_PATH = INSTANCE_DIR / "encodings.pkl"
LABELS_PATH = INSTANCE_DIR / "labels.json"

# Ngưỡng khoảng cách face_recognition
RECOGNITION_THRESHOLD = 0.45

# Liveness
LIVENESS_REQUIRED_DISTANCE = 45      
LIVENESS_REQUIRED_BLINKS = 1         
LIVENESS_EXPIRE_SECONDS = 30
EAR_THRESHOLD = 0.22                
EAR_CONSEC_FRAMES = 2               

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "demo-secret-key-change-in-production")
_YUNET_MODEL_PATH = cv2.data.haarcascades.replace(
    "haarcascades/", ""
) + "face_detection_yunet_2023mar.xml"

def _build_yunet(input_size=(320, 320)):
    """Tạo YuNet detector. Fallback về Haar nếu file model không có."""
    if os.path.exists(_YUNET_MODEL_PATH):
        detector = cv2.FaceDetectorYN.create(
            _YUNET_MODEL_PATH,
            "",
            input_size,
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5,
        )
        return detector
    return None

YUNET_DETECTOR = _build_yunet()
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
try:
    import dlib
    _DLIB_PREDICTOR_PATH = str(BASE_DIR / "shape_predictor_68_face_landmarks.dat")
    if os.path.exists(_DLIB_PREDICTOR_PATH):
        DLIB_DETECTOR = dlib.get_frontal_face_detector()
        DLIB_PREDICTOR = dlib.shape_predictor(_DLIB_PREDICTOR_PATH)
    else:
        DLIB_DETECTOR = None
        DLIB_PREDICTOR = None
except ImportError:
    DLIB_DETECTOR = None
    DLIB_PREDICTOR = None

LIVENESS_STATES = {}
ENCODINGS_CACHE = {"mtime": None, "data": None}
TRAIN_LOCK = threading.Lock()
TRAIN_STATUS = {"running": False, "message": "", "error": ""}


def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def today_text():
    return datetime.now().strftime("%Y-%m-%d")


def ensure_dirs():
    INSTANCE_DIR.mkdir(exist_ok=True)
    FACES_DIR.mkdir(exist_ok=True)


def get_db():
    import sqlite3

    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    ensure_dirs()
    db = get_db()

    db.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin', 'teacher', 'student')),
            created_at TEXT NOT NULL
        )
        """
    )

    db.execute(
        """
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_code TEXT NOT NULL UNIQUE,
            full_name TEXT NOT NULL,
            class_name TEXT NOT NULL,
            user_id INTEGER,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )

    db.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_name TEXT NOT NULL,
            subject_name TEXT NOT NULL,
            date TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'closed')),
            created_by INTEGER,
            created_at TEXT NOT NULL,
            FOREIGN KEY(created_by) REFERENCES users(id)
        )
        """
    )

    db.execute(
        """
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            session_id INTEGER NOT NULL,
            checkin_time TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'present',
            created_at TEXT NOT NULL,
            FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
            UNIQUE(student_id, session_id)
        )
        """
    )

    create_default_user(db, "admin", "admin123", "Quản trị viên", "admin")
    create_default_user(db, "teacher", "teacher123", "Giáo viên demo", "teacher")
    student_user_id = create_default_user(
        db, "B23DCCN455", "student123", "Cao Trung Kiên", "student"
    )

    existing_student = db.execute(
        "SELECT id FROM students WHERE student_code = ?", ("B23DCCN455",)
    ).fetchone()
    if not existing_student:
        db.execute(
            """
            INSERT INTO students(student_code, full_name, class_name, user_id, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("B23DCCN455", "Cao Trung Kiên", "D23CQCN07-B", student_user_id, now_text()),
        )

    existing_session = db.execute("SELECT id FROM sessions LIMIT 1").fetchone()
    if not existing_session:
        db.execute(
            """
            INSERT INTO sessions(class_name, subject_name, date, start_time, end_time, status, created_by, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "D23CQCN07-B",
                "Lập trình Web",
                today_text(),
                "08:00",
                "10:00",
                "active",
                1,
                now_text(),
            ),
        )

    db.commit()
    db.close()


def create_default_user(db, username, password, full_name, role):
    user = db.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    if user:
        return user["id"]

    cursor = db.execute(
        """
        INSERT INTO users(username, password_hash, full_name, role, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (username, generate_password_hash(password), full_name, role, now_text()),
    )
    return cursor.lastrowid


def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if "user_id" not in flask_session:
            flash("Vui lòng đăng nhập để tiếp tục.", "warning")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapper


def roles_required(*roles):
    def decorator(view_func):
        @wraps(view_func)
        def wrapper(*args, **kwargs):
            if "user_id" not in flask_session:
                flash("Vui lòng đăng nhập để tiếp tục.", "warning")
                return redirect(url_for("login"))
            if flask_session.get("role") not in roles:
                flash("Bạn không có quyền truy cập chức năng này.", "danger")
                return redirect(url_for("dashboard"))
            return view_func(*args, **kwargs)

        return wrapper

    return decorator


@app.context_processor
def inject_globals():
    return {"current_year": datetime.now().year}


def decode_image_from_base64(data_url):
    if not data_url or "," not in data_url:
        raise ValueError("Dữ liệu ảnh không hợp lệ")
    _, encoded = data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Không đọc được ảnh")
    return frame


def detect_faces(frame):
    """
    Phát hiện khuôn mặt bằng YuNet (ưu tiên) hoặc Haar Cascade (fallback).
    Trả về: (frame_rgb, [(x, y, w, h), ...]) — frame_rgb để face_recognition dùng.
    """
    h_img, w_img = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    valid_faces = []

    if YUNET_DETECTOR is not None:
        YUNET_DETECTOR.setInputSize((w_img, h_img))
        _, detections = YUNET_DETECTOR.detect(frame)
        if detections is not None:
            for det in detections:
                x, y, w, h = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                if w < 60 or h < 60:
                    continue
                aspect = w / h if h else 0
                if aspect < 0.5 or aspect > 2.0:
                    continue
                valid_faces.append((x, y, w, h))
    else:
        # Fallback: Haar Cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        for face in faces:
            x, y, w, h = [int(v) for v in face]
            if w < 60 or h < 60:
                continue
            aspect = w / h if h else 0
            if aspect < 0.5 or aspect > 2.0:
                continue
            valid_faces.append((x, y, w, h))

    valid_faces = sorted(valid_faces, key=lambda f: f[2] * f[3], reverse=True)
    return frame_rgb, valid_faces


def compute_ear(eye_points):
    """Tính Eye Aspect Ratio từ 6 điểm landmark của một mắt."""
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.3


def detect_blink(frame_rgb, face_box):
    """
    Phát hiện nháy mắt qua EAR.
    Trả về (ear_value, is_eye_closed).
    Yêu cầu dlib predictor; nếu không có thì trả None.
    """
    if DLIB_DETECTOR is None or DLIB_PREDICTOR is None:
        return None, False

    x, y, w, h = face_box
    frame_gray = cv2.cvtColor(
        cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY
    )
    rect = dlib.rectangle(x, y, x + w, y + h)
    shape = DLIB_PREDICTOR(frame_gray, rect)
    coords = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])

    left_eye = coords[36:42]
    right_eye = coords[42:48]
    ear = (compute_ear(left_eye) + compute_ear(right_eye)) / 2.0
    return ear, ear < EAR_THRESHOLD


def count_student_face_images(student_id):
    folder = FACES_DIR / str(student_id)
    if not folder.exists():
        return 0
    return len([p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])


ENCODED_INDEX_PATH = INSTANCE_DIR / "encoded_index.json"


def _load_encoded_index():
    """Load danh sách ảnh đã encode: {path: True}"""
    if ENCODED_INDEX_PATH.exists():
        try:
            return json.loads(ENCODED_INDEX_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def train_model():
    """
    Incremental encoding: chỉ encode ảnh mới, giữ lại encoding cũ.
    Nhanh hơn 5-10x khi đã có dữ liệu từ trước.
    """
    db = get_db()
    students = db.execute("SELECT id, student_code, full_name FROM students").fetchall()
    student_map = {str(row["id"]): dict(row) for row in students}
    db.close()

    # Load encodings cũ
    encoded_index = _load_encoded_index()
    if ENCODINGS_PATH.exists():
        with open(ENCODINGS_PATH, "rb") as f:
            old_data = pickle.load(f)
        encodings_list = list(old_data.get("encodings", []))
        labels = list(old_data.get("labels", []))
    else:
        encodings_list = []
        labels = []

    label_info = {}
    new_count = 0

    for student_folder in (FACES_DIR.iterdir() if FACES_DIR.exists() else []):
        if not student_folder.is_dir():
            continue
        try:
            student_id = int(student_folder.name)
        except ValueError:
            continue

        for image_path in student_folder.glob("*.jpg"):
            path_key = str(image_path)
            label_info[str(student_id)] = student_map.get(str(student_id), {})

            # Bỏ qua ảnh đã encode rồi
            if path_key in encoded_index:
                continue

            img_bgr = cv2.imread(path_key)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            boxes = face_recognition.face_locations(img_rgb, model="hog")
            if not boxes:
                continue
            encs = face_recognition.face_encodings(img_rgb, known_face_locations=boxes)
            if not encs:
                continue

            encodings_list.append(encs[0])
            labels.append(student_id)
            encoded_index[path_key] = True
            new_count += 1

    total_images = len(encodings_list)
    if total_images < 2:
        raise RuntimeError("Cần ít nhất 2 ảnh khuôn mặt để tạo encodings.")

    data = {"encodings": encodings_list, "labels": labels}
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(data, f)

    ENCODED_INDEX_PATH.write_text(
        json.dumps(encoded_index, ensure_ascii=False), encoding="utf-8"
    )

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(label_info, f, ensure_ascii=False, indent=2)

    ENCODINGS_CACHE["mtime"] = None
    ENCODINGS_CACHE["data"] = None
    return total_images, len(label_info), new_count


def train_model_background():
    """Chạy train_model trong background thread, không block server."""
    if TRAIN_STATUS["running"]:
        return False  # Đang train rồi
    def _run():
        with TRAIN_LOCK:
            TRAIN_STATUS["running"] = True
            TRAIN_STATUS["error"] = ""
            TRAIN_STATUS["message"] = "Đang huấn luyện..."
            try:
                total, students, new_count = train_model()
                TRAIN_STATUS["message"] = (
                    f"Hoàn tất! {new_count} ảnh mới, tổng {total} ảnh / {students} sinh viên."
                )
            except Exception as e:
                TRAIN_STATUS["error"] = str(e)
                TRAIN_STATUS["message"] = ""
            finally:
                TRAIN_STATUS["running"] = False
    threading.Thread(target=_run, daemon=True).start()
    return True


def load_recognizer():
    """Load encodings từ file .pkl, có cache để tránh đọc lại mỗi request."""
    if not ENCODINGS_PATH.exists():
        return None
    mtime = ENCODINGS_PATH.stat().st_mtime
    if ENCODINGS_CACHE["data"] is not None and ENCODINGS_CACHE["mtime"] == mtime:
        return ENCODINGS_CACHE["data"]

    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)
    ENCODINGS_CACHE["mtime"] = mtime
    ENCODINGS_CACHE["data"] = data
    return data


def liveness_key(lesson_id):
    if "client_id" not in flask_session:
        flask_session["client_id"] = secrets.token_hex(16)
    return f"{flask_session['client_id']}:{lesson_id}"


def update_liveness(lesson_id, face, frame_rgb=None):
    """
    Liveness 2 lớp:
      1. Head-turn: khuôn mặt phải di chuyển ngang >= LIVENESS_REQUIRED_DISTANCE px
      2. Blink: phát hiện >= LIVENESS_REQUIRED_BLINKS lần nháy mắt (nếu có dlib)
    Trả về (verified, progress_0_100, hint_message)
    """
    key = liveness_key(lesson_id)
    now = datetime.now().timestamp()
    x, y, w, h = [int(v) for v in face]
    center_x = x + w / 2

    state = LIVENESS_STATES.get(key)
    if not state or now - state.get("updated_at", 0) > LIVENESS_EXPIRE_SECONDS:
        state = {
            "min_x": center_x,
            "max_x": center_x,
            "frames": 0,
            "verified": False,
            "updated_at": now,
            # blink tracking
            "blink_count": 0,
            "eye_closed_frames": 0,
            "eye_was_closed": False,
        }

    state["min_x"] = min(state["min_x"], center_x)
    state["max_x"] = max(state["max_x"], center_x)
    state["frames"] += 1
    state["updated_at"] = now

    # --- Blink detection ---
    blink_available = False
    if frame_rgb is not None and DLIB_PREDICTOR is not None:
        blink_available = True
        _, is_closed = detect_blink(frame_rgb, face)
        if is_closed:
            state["eye_closed_frames"] += 1
        else:
            if state["eye_closed_frames"] >= EAR_CONSEC_FRAMES:
                state["blink_count"] += 1
            state["eye_closed_frames"] = 0

    # --- Tính progress tổng hợp ---
    head_turn_distance = state["max_x"] - state["min_x"]
    head_progress = min(100, int((head_turn_distance / LIVENESS_REQUIRED_DISTANCE) * 100))

    if blink_available:
        blink_progress = min(100, int((state["blink_count"] / LIVENESS_REQUIRED_BLINKS) * 100))
        progress = int((head_progress + blink_progress) / 2)
        head_ok = head_turn_distance >= LIVENESS_REQUIRED_DISTANCE
        blink_ok = state["blink_count"] >= LIVENESS_REQUIRED_BLINKS
        verified = head_ok and blink_ok and state["frames"] >= 4

        if not head_ok:
            hint = "Vui lòng lắc nhẹ đầu sang trái/phải."
        elif not blink_ok:
            hint = "Vui lòng nháy mắt một lần."
        else:
            hint = "Xác minh hoàn tất!"
    else:
        # Chỉ head-turn nếu không có dlib
        progress = head_progress
        verified = head_turn_distance >= LIVENESS_REQUIRED_DISTANCE and state["frames"] >= 4
        hint = (
            "Xác minh hoàn tất!"
            if verified
            else "Vui lòng lắc nhẹ đầu sang trái/phải."
        )

    if verified:
        state["verified"] = True
        progress = 100

    LIVENESS_STATES[key] = state
    return state["verified"], progress, hint


def reset_liveness_state(lesson_id):
    key = liveness_key(lesson_id)
    LIVENESS_STATES.pop(key, None)


def build_report_query(filters):
    sql = """
        SELECT
            a.id AS attendance_id,
            st.student_code,
            st.full_name,
            st.class_name,
            se.subject_name,
            se.date,
            se.start_time,
            se.end_time,
            a.checkin_time,
            CASE
                WHEN a.status = 'present' THEN 'Có mặt'
                ELSE 'Vắng mặt'
            END AS status
        FROM sessions se
        JOIN students st 
            ON st.class_name = se.class_name
        LEFT JOIN attendance a
            ON a.student_id = st.id
            AND a.session_id = se.id
        WHERE 1 = 1
    """

    params = []

    if filters.get("date"):
        sql += " AND se.date = ?"
        params.append(filters["date"])

    if filters.get("class_name"):
        sql += " AND st.class_name = ?"
        params.append(filters["class_name"])

    if filters.get("session_id"):
        sql += " AND se.id = ?"
        params.append(filters["session_id"])

    sql += """
        ORDER BY 
            se.date DESC,
            se.start_time DESC,
            st.id ASC
    """

    return sql, params

@app.route("/")
def index():
    if "user_id" in flask_session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        db = get_db()
        user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        db.close()

        if user and check_password_hash(user["password_hash"], password):
            flask_session.clear()
            flask_session["user_id"] = user["id"]
            flask_session["username"] = user["username"]
            flask_session["full_name"] = user["full_name"]
            flask_session["role"] = user["role"]
            flash("Đăng nhập thành công.", "success")
            return redirect(url_for("dashboard"))

        flash("Sai tài khoản hoặc mật khẩu.", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    flask_session.clear()
    flash("Đã đăng xuất.", "info")
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    db = get_db()
    
    # Nếu là sinh viên, chỉ hiển thị dữ liệu của chính họ
    if flask_session.get("role") == "student":
        student = db.execute(
            "SELECT id FROM students WHERE user_id = ?", (flask_session["user_id"],)
        ).fetchone()
        
        if student:
            total_students = 1
            active_sessions = db.execute("SELECT COUNT(*) AS c FROM sessions WHERE status='active'").fetchone()["c"]
            today_attendance = db.execute(
                """
                SELECT COUNT(*) AS c
                FROM attendance a
                JOIN sessions s ON s.id = a.session_id
                WHERE s.date = ? AND a.student_id = ? AND a.status = 'present'
                """,
                (today_text(), student["id"]),
            ).fetchone()["c"]
            recent = db.execute(
                """
                SELECT st.student_code, st.full_name, se.subject_name, se.date, a.checkin_time, a.status
                FROM attendance a
                JOIN students st ON st.id = a.student_id
                JOIN sessions se ON se.id = a.session_id
                WHERE a.student_id = ?
                ORDER BY a.id DESC
                LIMIT 8
                """,
                (student["id"],),
            ).fetchall()
            # Thống kê tổng có mặt / vắng mặt
            attendance_stats = db.execute(
                """
                SELECT a.status, COUNT(*) as count
                FROM attendance a
                WHERE a.student_id = ?
                GROUP BY a.status
                """,
                (student["id"],),
            ).fetchall()
            present_count = sum(r["count"] for r in attendance_stats if r["status"] == "present")
            absent_count  = sum(r["count"] for r in attendance_stats if r["status"] == "absent")

            # Thống kê điểm danh theo môn học
            subject_stats = db.execute(
                """
                SELECT se.subject_name,
                       SUM(CASE WHEN a.status='present' THEN 1 ELSE 0 END) AS present,
                       SUM(CASE WHEN a.status='absent'  THEN 1 ELSE 0 END) AS absent
                FROM attendance a
                JOIN sessions se ON se.id = a.session_id
                WHERE a.student_id = ?
                GROUP BY se.subject_name
                ORDER BY se.subject_name
                """,
                (student["id"],),
            ).fetchall()

            # Thống kê điểm danh theo 7 ngày gần nhất
            weekly_stats = db.execute(
                """
                SELECT se.date,
                       SUM(CASE WHEN a.status='present' THEN 1 ELSE 0 END) AS present,
                       SUM(CASE WHEN a.status='absent'  THEN 1 ELSE 0 END) AS absent
                FROM attendance a
                JOIN sessions se ON se.id = a.session_id
                WHERE a.student_id = ?
                ORDER BY se.date DESC
                LIMIT 7
                """,
                (student["id"],),
            ).fetchall()
        else:
            total_students = 0
            active_sessions = 0
            today_attendance = 0
            recent = []
            present_count = 0
            absent_count = 0
            subject_stats = []
            weekly_stats = []
    else:
        # Admin và giáo viên xem toàn bộ dữ liệu
        total_students = db.execute("SELECT COUNT(*) AS c FROM students").fetchone()["c"]
        active_sessions = db.execute("SELECT COUNT(*) AS c FROM sessions WHERE status='active'").fetchone()["c"]
        today_attendance = db.execute(
            """
            SELECT COUNT(*) AS c
            FROM attendance a
            JOIN sessions s ON s.id = a.session_id
            WHERE s.date = ? AND a.status = 'present'
            """,
            (today_text(),),
        ).fetchone()["c"]
        recent = db.execute(
            """
            SELECT st.student_code, st.full_name, se.subject_name, se.date, a.checkin_time
            FROM attendance a
            JOIN students st ON st.id = a.student_id
            JOIN sessions se ON se.id = a.session_id
            ORDER BY a.id DESC
            LIMIT 8
            """
        ).fetchall()
    
    db.close()

    is_student = flask_session.get("role") == "student"
    return render_template(
        "dashboard.html",
        total_students=total_students,
        active_sessions=active_sessions,
        today_attendance=today_attendance,
        recent=recent,
        is_student=is_student,
        present_count=present_count if is_student else 0,
        absent_count=absent_count if is_student else 0,
        subject_stats=[dict(r) for r in subject_stats] if is_student else [],
        weekly_stats=[dict(r) for r in weekly_stats] if is_student else [],
    )


@app.route("/users", methods=["GET", "POST"])
@roles_required("admin")
def users():
    db = get_db()
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        full_name = request.form.get("full_name", "").strip()
        role = request.form.get("role", "student").strip()

        if not username or not password or not full_name or role not in {"admin", "teacher", "student"}:
            flash("Vui lòng nhập đầy đủ và đúng thông tin tài khoản.", "danger")
        else:
            try:
                db.execute(
                    """
                    INSERT INTO users(username, password_hash, full_name, role, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (username, generate_password_hash(password), full_name, role, now_text()),
                )
                db.commit()
                flash("Đã tạo tài khoản mới.", "success")
            except Exception as exc:
                db.rollback()
                flash(f"Không tạo được tài khoản: {exc}", "danger")

    all_users = db.execute("SELECT * FROM users ORDER BY id DESC").fetchall()
    db.close()
    return render_template("users.html", users=all_users)


@app.route("/students")
@login_required
def students():
    db = get_db()
    if flask_session.get("role") == "student":
        rows = db.execute(
            "SELECT * FROM students WHERE user_id = ? ORDER BY id ASC",
            (flask_session["user_id"],),
        ).fetchall()
    else:
        rows = db.execute("SELECT * FROM students ORDER BY id ASC").fetchall()
    counts = {row["id"]: count_student_face_images(row["id"]) for row in rows}
    db.close()
    return render_template("students.html", students=rows, face_counts=counts)


@app.route("/students/add", methods=["GET", "POST"])
@roles_required("admin", "teacher")
def add_student():
    if request.method == "POST":
        student_code = request.form.get("student_code", "").strip()
        full_name = request.form.get("full_name", "").strip()
        class_name = request.form.get("class_name", "").strip()
        username = request.form.get("username", "").strip() or student_code
        password = request.form.get("password", "").strip() or "123456"

        if not student_code or not full_name or not class_name:
            flash("Vui lòng nhập mã sinh viên, họ tên và lớp.", "danger")
            return render_template("student_form.html")

        db = get_db()
        try:
            user = db.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
            if user:
                user_id = user["id"]
            else:
                cur = db.execute(
                    """
                    INSERT INTO users(username, password_hash, full_name, role, created_at)
                    VALUES (?, ?, ?, 'student', ?)
                    """,
                    (username, generate_password_hash(password), full_name, now_text()),
                )
                user_id = cur.lastrowid

            db.execute(
                """
                INSERT INTO students(student_code, full_name, class_name, user_id, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (student_code, full_name, class_name, user_id, now_text()),
            )
            db.commit()
            flash("Đã thêm sinh viên.", "success")
            return redirect(url_for("students"))
        except Exception as exc:
            db.rollback()
            flash(f"Không thêm được sinh viên: {exc}", "danger")
        finally:
            db.close()

    return render_template("student_form.html")


@app.route("/students/<int:student_id>/delete", methods=["POST"])
@roles_required("admin")
def delete_student(student_id):
    db = get_db()
    db.execute("DELETE FROM students WHERE id = ?", (student_id,))
    db.commit()
    db.close()
    folder = FACES_DIR / str(student_id)
    if folder.exists():
        shutil.rmtree(folder)
    try:
        train_model()
    except Exception:
        pass
    flash("Đã xóa sinh viên.", "info")
    return redirect(url_for("students"))


@app.route("/students/<int:student_id>/delete-faces", methods=["POST"])
@roles_required("admin", "teacher")
def delete_student_faces(student_id):
    folder = FACES_DIR / str(student_id)
    if folder.exists():
        shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)
    try:
        train_model()
    except Exception:
        pass
    flash("Đã xóa toàn bộ ảnh khuôn mặt của sinh viên.", "info")
    return redirect(url_for("students"))


@app.route("/students/<int:student_id>/enroll")
@roles_required("admin", "teacher")
def enroll(student_id):
    db = get_db()
    student = db.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
    db.close()
    if not student:
        flash("Không tìm thấy sinh viên.", "danger")
        return redirect(url_for("students"))
    return render_template(
        "enroll.html",
        student=student,
        face_count=count_student_face_images(student_id),
    )


@app.route("/api/enroll/<int:student_id>", methods=["POST"])
@roles_required("admin", "teacher")
def api_enroll(student_id):
    data = request.get_json(silent=True) or {}
    image_data = data.get("image")

    db = get_db()
    student = db.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
    db.close()
    if not student:
        return jsonify({"ok": False, "message": "Không tìm thấy sinh viên."}), 404

    try:
        frame = decode_image_from_base64(image_data)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(frame_rgb, model="hog")
        if len(boxes) == 0:
            return jsonify({"ok": False, "message": "Không phát hiện khuôn mặt. Hãy nhìn vào camera và đảm bảo đủ ánh sáng."})
        if len(boxes) > 1:
            return jsonify({"ok": False, "message": "Chỉ chấp nhận 1 khuôn mặt trong khung hình."})

        encs = face_recognition.face_encodings(frame_rgb, known_face_locations=boxes)
        if not encs:
            return jsonify({"ok": False, "message": "Không trích xuất được đặc trưng khuôn mặt. Hãy chụp lại."})

        # Lưu toàn bộ frame (BGR) để dùng khi retrain
        folder = FACES_DIR / str(student_id)
        folder.mkdir(parents=True, exist_ok=True)
        filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
        cv2.imwrite(str(folder / filename), frame)

        count = count_student_face_images(student_id)
        # Chạy train trong background — không block request
        if count >= 5:
            train_model_background()
            train_message = "Đang cập nhật model nhận diện trong nền..."
        else:
            train_message = f"Cần thêm {5 - count} ảnh nữa để bắt đầu huấn luyện."

        return jsonify(
            {
                "ok": True,
                "count": count,
                "trained": count >= 5,
                "message": f"Đã lưu ảnh khuôn mặt số {count}. {train_message}",
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)}), 400


@app.route("/api/train", methods=["POST"])
@roles_required("admin", "teacher")
def api_train():
    if TRAIN_STATUS["running"]:
        return jsonify({"ok": False, "message": "Đang huấn luyện, vui lòng chờ..."})
    started = train_model_background()
    return jsonify({
        "ok": True,
        "message": "Bắt đầu huấn luyện trong nền. Dùng /api/train/status để kiểm tra tiến trình.",
    })


@app.route("/api/train/status")
@roles_required("admin", "teacher")
def api_train_status():
    return jsonify({
        "running": TRAIN_STATUS["running"],
        "message": TRAIN_STATUS["message"],
        "error": TRAIN_STATUS["error"],
    })


@app.route("/sessions")
@login_required
def sessions_list():
    db = get_db()
    rows = db.execute("SELECT * FROM sessions ORDER BY date DESC, start_time DESC").fetchall()
    db.close()
    return render_template("sessions.html", sessions=rows)


@app.route("/sessions/add", methods=["GET", "POST"])
@roles_required("admin", "teacher")
def add_session():
    if request.method == "POST":
        class_name = request.form.get("class_name", "").strip()
        subject_name = request.form.get("subject_name", "").strip()
        date = request.form.get("date", "").strip()
        start_time = request.form.get("start_time", "").strip()
        end_time = request.form.get("end_time", "").strip()
        status = request.form.get("status", "active")

        if not class_name or not subject_name or not date or not start_time or not end_time:
            flash("Vui lòng nhập đầy đủ thông tin buổi học.", "danger")
            return render_template("session_form.html", today=today_text())

        db = get_db()
        db.execute(
            """
            INSERT INTO sessions(class_name, subject_name, date, start_time, end_time, status, created_by, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                class_name,
                subject_name,
                date,
                start_time,
                end_time,
                status,
                flask_session["user_id"],
                now_text(),
            ),
        )
        db.commit()
        db.close()
        flash("Đã tạo buổi học.", "success")
        return redirect(url_for("sessions_list"))

    return render_template("session_form.html", today=today_text())


@app.route("/sessions/<int:lesson_id>/camera")
@roles_required("admin", "teacher")
def attendance_camera(lesson_id):
    db = get_db()
    lesson = db.execute("SELECT * FROM sessions WHERE id = ?", (lesson_id,)).fetchone()
    db.close()
    if not lesson:
        flash("Không tìm thấy buổi học.", "danger")
        return redirect(url_for("sessions_list"))
    return render_template("attendance_camera.html", lesson=lesson)


@app.route("/api/recognize/<int:lesson_id>", methods=["POST"])
@roles_required("admin", "teacher")
def api_recognize(lesson_id):
    db = get_db()
    lesson = db.execute("SELECT * FROM sessions WHERE id = ?", (lesson_id,)).fetchone()
    if not lesson:
        db.close()
        return jsonify({"ok": False, "message": "Không tìm thấy buổi học."}), 404
    if lesson["status"] != "active":
        db.close()
        return jsonify({"ok": False, "message": "Buổi học đã đóng, không thể điểm danh."})

    data = request.get_json(silent=True) or {}
    image_data = data.get("image")

    try:
        frame = decode_image_from_base64(image_data)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Dùng face_recognition detect — chịu góc nghiêng tốt hơn YuNet/Haar
        boxes = face_recognition.face_locations(frame_rgb, model="hog")
        if len(boxes) == 0:
            db.close()
            return jsonify({"ok": False, "status": "no_face", "message": "Không phát hiện khuôn mặt."})
        if len(boxes) > 1:
            db.close()
            return jsonify({"ok": False, "status": "multiple_faces", "message": "Chỉ đưa 1 khuôn mặt vào camera."})

        # Chuyển box từ (top,right,bottom,left) sang (x,y,w,h) cho liveness
        top, right, bottom, left = boxes[0]
        face_xywh = (left, top, right - left, bottom - top)

        verified, progress, hint = update_liveness(lesson_id, face_xywh, frame_rgb)
        if not verified:
            db.close()
            return jsonify(
                {
                    "ok": True,
                    "status": "liveness_required",
                    "progress": progress,
                    "message": hint,
                }
            )

        enc_data = load_recognizer()
        if enc_data is None:
            db.close()
            return jsonify(
                {
                    "ok": False,
                    "status": "model_missing",
                    "message": "Chưa có encodings nhận diện. Hãy đăng ký khuôn mặt và huấn luyện model.",
                }
            )

        # face_recognition: so sánh encoding của frame với toàn bộ encodings đã lưu
        query_encs = face_recognition.face_encodings(frame_rgb, known_face_locations=boxes)
        if not query_encs:
            db.close()
            return jsonify({"ok": True, "status": "unknown", "message": "Không trích xuất được đặc trưng. Hãy thử lại."})

        query_enc = query_encs[0]
        distances = face_recognition.face_distance(enc_data["encodings"], query_enc)

        if len(distances) == 0:
            db.close()
            return jsonify({"ok": True, "status": "unknown", "message": "Không có dữ liệu khuôn mặt nào."})

        best_idx = int(np.argmin(distances))
        best_distance = float(distances[best_idx])
        confidence_pct = round((1 - best_distance) * 100, 2)  # % giống nhau

        if best_distance > RECOGNITION_THRESHOLD:
            db.close()
            return jsonify(
                {
                    "ok": True,
                    "status": "unknown",
                    "confidence": confidence_pct,
                    "message": "Không nhận diện được sinh viên. Hãy thử lại hoặc đăng ký thêm ảnh.",
                }
            )

        predicted_student_id = enc_data["labels"][best_idx]
        student = db.execute(
            "SELECT * FROM students WHERE id = ?", (int(predicted_student_id),)
        ).fetchone()
        if not student:
            db.close()
            return jsonify({"ok": False, "message": "Không tìm thấy sinh viên trong database."})

        if student["class_name"].strip() != lesson["class_name"].strip():
            db.close()
            return jsonify(
                {
                    "ok": True,
                    "status": "wrong_class",
                    "message": f"Sinh viên {student['full_name']} không thuộc lớp {lesson['class_name']}.",
                    "student": dict(student),
                    "confidence": confidence_pct,
                }
            )

        try:
            db.execute(
                """
                INSERT INTO attendance(student_id, session_id, checkin_time, status, created_at)
                VALUES (?, ?, ?, 'present', ?)
                """,
                (student["id"], lesson_id, now_text(), now_text()),
            )
            db.commit()
            reset_liveness_state(lesson_id)
            message = f"Điểm danh thành công: {student['student_code']} - {student['full_name']}"
            status = "marked"
        except Exception:
            reset_liveness_state(lesson_id)
            message = f"{student['student_code']} - {student['full_name']} đã điểm danh trước đó."
            status = "already"

        db.close()
        return jsonify(
            {
                "ok": True,
                "status": status,
                "message": message,
                "student": dict(student),
                "confidence": confidence_pct,
            }
        )
    except Exception as exc:
        db.close()
        return jsonify({"ok": False, "message": str(exc)}), 400


@app.route("/api/liveness/reset/<int:lesson_id>", methods=["POST"])
@roles_required("admin", "teacher")
def api_liveness_reset(lesson_id):
    reset_liveness_state(lesson_id)
    return jsonify({"ok": True, "message": "Đã reset bước xác minh người thật."})


@app.route("/reports")
@roles_required("admin", "teacher")
def reports():
    filters = {
        "date": request.args.get("date", "").strip(),
        "class_name": request.args.get("class_name", "").strip(),
        "session_id": request.args.get("session_id", "").strip(),
    }
    sql, params = build_report_query(filters)
    db = get_db()
    rows = db.execute(sql, params).fetchall()
    class_names = db.execute(
        "SELECT DISTINCT class_name FROM students ORDER BY class_name"
    ).fetchall()
    lesson_rows = db.execute(
        "SELECT id, class_name, subject_name, date, start_time FROM sessions ORDER BY date DESC"
    ).fetchall()
    db.close()
    return render_template(
        "reports.html",
        rows=rows,
        class_names=class_names,
        sessions=lesson_rows,
        filters=filters,
    )


@app.route("/reports/export")
@roles_required("admin", "teacher")
def export_report():
    filters = {
        "date": request.args.get("date", "").strip(),
        "class_name": request.args.get("class_name", "").strip(),
        "session_id": request.args.get("session_id", "").strip(),
    }
    sql, params = build_report_query(filters)
    db = get_db()
    rows = [dict(row) for row in db.execute(sql, params).fetchall()]
    db.close()

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "student_code",
                "full_name",
                "class_name",
                "subject_name",
                "date",
                "start_time",
                "end_time",
                "checkin_time",
                "status",
            ]
        )

    df = df.rename(
        columns={
            "student_code": "Mã sinh viên",
            "full_name": "Họ tên",
            "class_name": "Lớp",
            "subject_name": "Môn học",
            "date": "Ngày học",
            "start_time": "Giờ bắt đầu",
            "end_time": "Giờ kết thúc",
            "checkin_time": "Thời gian điểm danh",
            "status": "Trạng thái",
        }
    )

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="BaoCaoDiemDanh")
        worksheet = writer.sheets["BaoCaoDiemDanh"]
        for column_cells in worksheet.columns:
            max_length = 12
            column_letter = column_cells[0].column_letter
            for cell in column_cells:
                value = str(cell.value) if cell.value is not None else ""
                max_length = max(max_length, len(value) + 2)
            worksheet.column_dimensions[column_letter].width = min(max_length, 35)

    output.seek(0)
    filename = "bao_cao_diem_danh_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".xlsx"
    return send_file(
        output,
        as_attachment=True,
        download_name=filename,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.route("/api/close-session/<int:session_id>", methods=["POST"])
@roles_required("admin", "teacher")
def api_close_session(session_id):
    db = get_db()
    try:
        # Lấy thông tin buổi học
        session = db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if not session:
            db.close()
            return jsonify({"ok": False, "message": "Không tìm thấy buổi học."}), 404
        
        if session["status"] != "active":
            db.close()
            return jsonify({"ok": False, "message": "Buổi học đã đóng hoặc không hoạt động."}), 400
        
        # Lấy danh sách sinh viên của lớp
        students = db.execute(
            "SELECT id FROM students WHERE class_name = ?", (session["class_name"],)
        ).fetchall()
        
        # Lấy danh sách sinh viên đã điểm danh
        attended_students = db.execute(
            "SELECT student_id FROM attendance WHERE session_id = ?", (session_id,)
        ).fetchall()
        attended_ids = {row["student_id"] for row in attended_students}
        
        # Thêm attendance "absent" cho sinh viên chưa điểm danh
        absent_count = 0
        for student in students:
            student_id = student["id"]
            if student_id not in attended_ids:
                db.execute(
                    """
                    INSERT INTO attendance(student_id, session_id, checkin_time, status, created_at)
                    VALUES (?, ?, ?, 'absent', ?)
                    """,
                    # set checkin_time to now to satisfy NOT NULL constraint
                    (student_id, session_id, now_text(), now_text()),
                )
                absent_count += 1
        
        # Đóng buổi học
        db.execute("UPDATE sessions SET status = 'closed' WHERE id = ?", (session_id,))
        db.commit()
        
        message = f"Đã đóng buổi học. Thêm {absent_count} sinh viên vắng mặt."
        return jsonify({"ok": True, "message": message, "absent_count": absent_count})
    except Exception as exc:
        db.rollback()
        return jsonify({"ok": False, "message": str(exc)}), 400
    finally:
        db.close()


@app.route("/my-attendance")
@roles_required("student")
def my_attendance():
    db = get_db()
    student = db.execute(
        "SELECT * FROM students WHERE user_id = ?", (flask_session["user_id"],)
    ).fetchone()
    rows = []
    if student:
        rows = db.execute(
            """
            SELECT se.subject_name, se.date, se.start_time, se.end_time, a.checkin_time, a.status
            FROM attendance a
            JOIN sessions se ON se.id = a.session_id
            WHERE a.student_id = ?
            ORDER BY se.date DESC, a.checkin_time DESC
            """,
            (student["id"],),
        ).fetchall()
    db.close()
    return render_template("my_attendance.html", student=student, rows=rows)


if __name__ == "__main__":
    init_db()
    print("==============================================")
    print("Hệ thống điểm danh đang chạy tại:")
    print("http://127.0.0.1:5000")
    print("Tài khoản admin: admin / admin123")
    print("Tài khoản giáo viên: teacher / teacher123")
    print("==============================================")
    app.run(host="127.0.0.1", port=5000, debug=True)