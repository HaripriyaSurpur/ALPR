import os
import cv2
import re
import numpy as np
import threading
import webbrowser
import csv
import sqlite3

from datetime import datetime
from flask import Flask, render_template, request, send_from_directory, url_for, redirect, session, flash
from pathlib import Path
import easyocr

# ---------- Siren ----------
try:
    import winsound
    WINDOWS = True
except:
    WINDOWS = False
    import pygame

app = Flask(__name__)
app.secret_key = "alpr_secret"

UPLOAD = 'uploads'
RESULT = 'results'

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(RESULT, exist_ok=True)

# =========================
# DATABASE
# =========================
DB_NAME = "alpr.db"

def init_db():

    conn = sqlite3.connect(DB_NAME)

    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS scans(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        plate TEXT,
        state TEXT,
        threat TEXT,
        alert TEXT,
        image TEXT,
        date TEXT,
        time TEXT
    )
    """)

    # ---------- FIX OLD DATABASE ----------
    columns = [i[1] for i in cur.execute("PRAGMA table_info(scans)")]

    if "threat" not in columns:
        cur.execute("ALTER TABLE scans ADD COLUMN threat TEXT")

    if "alert" not in columns:
        cur.execute("ALTER TABLE scans ADD COLUMN alert TEXT")

    if "image" not in columns:
        cur.execute("ALTER TABLE scans ADD COLUMN image TEXT")

    if "date" not in columns:
        cur.execute("ALTER TABLE scans ADD COLUMN date TEXT")

    if "time" not in columns:
        cur.execute("ALTER TABLE scans ADD COLUMN time TEXT")

    conn.commit()
    conn.close()

init_db()

# ---------- YOLO load ----------
yolo = None

try:

    from ultralytics import YOLO

    for m in ["yolov8_lp.pt", "best.pt", "lp.pt", "yolov8n.pt"]:

        if Path(m).is_file():

            yolo = YOLO(m)

            print("Loaded YOLO:", m)

            break

except Exception as e:

    print("YOLO ERROR:", e)

# ---------- EAST load ----------
EAST_PATH = "frozen_east_text_detection.pb"

east = cv2.dnn.readNet(EAST_PATH) if Path(EAST_PATH).is_file() else None

# ---------- OCR ----------
reader = easyocr.Reader(['en'], gpu=False)

# ---------- State Codes ----------
STATE = {
    "KA":"Karnataka",
    "MH":"Maharashtra",
    "DL":"Delhi",
    "TN":"Tamil Nadu",
    "GJ":"Gujarat",
    "UP":"Uttar Pradesh",
    "RJ":"Rajasthan",
    "PB":"Punjab",
    "HR":"Haryana",
    "AP":"Andhra Pradesh",
    "TS":"Telangana",
    "KL":"Kerala",
    "BR":"Bihar",
    "WB":"West Bengal",
    "MP":"Madhya Pradesh",
    "JK":"Jammu & Kashmir"
}

# ---------- Load Blacklist ----------
BLACKLIST = set()

if Path("blacklist.csv").is_file():

    with open("blacklist.csv", newline='') as f:

        reader_csv = csv.DictReader(f)

        for row in reader_csv:

            BLACKLIST.add(
                row['plate'].strip().upper()
            )

# ---------- Utils ----------
def find_state(p):

    for i in range(len(p)-1):

        if p[i:i+2] in STATE:

            return STATE[p[i:i+2]]

    return "Unknown"

def east_detect(img):

    if east is None:
        return None

    H, W = img.shape[:2]

    blob = cv2.dnn.blobFromImage(
        img,
        1,
        (320,320),
        (123.68,116.78,103.94),
        swapRB=True,
        crop=False
    )

    east.setInput(blob)

    s, g = east.forward([
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ])

    rects = []
    conf = []

    for y in range(s.shape[2]):

        for x in range(s.shape[3]):

            if s[0,0,y,x] < 0.5:
                continue

            offX = x * 4.0
            offY = y * 4.0

            a = g[0,4,y,x]

            cy = np.cos(a)
            sy = np.sin(a)

            h = g[0,0,y,x]
            w = g[0,1,y,x]

            endX = int(offX + cy*w + sy*h)
            endY = int(offY - sy*w + cy*h)

            rects.append(
                (endX-w, endY-h, endX, endY)
            )

            conf.append(float(s[0,0,y,x]))

    if len(rects) == 0:
        return None

    boxes = cv2.dnn.NMSBoxes(
        rects,
        conf,
        0.5,
        0.4
    )

    if len(boxes) == 0:
        return None

    idx = boxes[0]

    if isinstance(idx, (list, np.ndarray)):
        idx = idx[0]

    x1,y1,x2,y2 = rects[idx]

    x1 = max(0, x1-5)
    y1 = max(0, y1-5)

    x2 = min(W-1, x2+5)
    y2 = min(H-1, y2+5)

    return img[y1:y2, x1:x2]

def contour_detect(img):

    g = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    e = cv2.Canny(g,30,200)

    cnt,_ = cv2.findContours(
        e,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for c in sorted(
        cnt,
        key=cv2.contourArea,
        reverse=True
    )[:40]:

        x,y,w,h = cv2.boundingRect(c)

        if 2 <= w/max(h,1) <= 6.5 and w > 60 and h > 15:

            return img[y:y+h, x:x+w]

    return img

def ocr(crop):

    if crop is None:
        return ""

    g = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )

    g = cv2.resize(
        g,
        None,
        fx=2,
        fy=2
    )

    _, t = cv2.threshold(
        g,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    text = []

    for im in (g,t):

        for r in reader.readtext(im):

            text.append(r[1])

    if not text:
        return ""

    best = max(
        text,
        key=lambda x: len(
            re.sub(
                r"[^A-Z0-9]",
                "",
                x.upper()
            )
        )
    )

    return re.sub(
        r"[^A-Z0-9]",
        "",
        best.upper()
    )

def is_red_plate(img):

    if img is None:
        return False

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    mask1 = cv2.inRange(
        hsv,
        np.array([0,70,50]),
        np.array([10,255,255])
    )

    mask2 = cv2.inRange(
        hsv,
        np.array([170,70,50]),
        np.array([180,255,255])
    )

    mask = mask1 + mask2

    return (
        cv2.countNonZero(mask) /
        (img.shape[0] * img.shape[1])
    ) > 0.25

# ---------- Siren ----------
siren_active = False

def play_siren():

    global siren_active

    if siren_active:
        return

    siren_active = True

    try:

        if WINDOWS:

            while siren_active:

                winsound.PlaySound(
                    "static/siren.wav",
                    winsound.SND_FILENAME
                )

        else:

            pygame.mixer.init()

            pygame.mixer.music.load(
                "static/siren.wav"
            )

            pygame.mixer.music.play(-1)

    except Exception as e:

        print("SIREN ERROR:", e)

def stop_siren():

    global siren_active

    siren_active = False

    if not WINDOWS:

        try:
            pygame.mixer.music.stop()
        except:
            pass

# ---------- Routes ----------
@app.route('/login', methods=['GET','POST'])
def login():

    if request.method == 'POST':

        u = request.form.get("username")
        p = request.form.get("password")

        if u == "admin" and p == "admin123":

            session['user'] = u

            return redirect('/')

        else:

            flash("Unauthorized User! Access Denied.")

            return redirect('/login')

    return render_template("login.html")

@app.route('/logout')
def logout():

    session.clear()

    stop_siren()

    return redirect('/login')

@app.route('/')
def index():

    if 'user' not in session:
        return redirect('/login')

    return render_template(
        "index.html",
        plate_text=None,
        threat=False
    )

@app.route('/upload',methods=['POST'])
def upload():

    if 'user' not in session:
        return redirect('/login')

    f = request.files.get('image')

    if not f:
        return redirect('/')

    path = os.path.join(
        UPLOAD,
        f.filename
    )

    f.save(path)

    img = cv2.imread(path)

    crop = None

    # ---------- YOLO ----------
    if yolo:

        try:

            r = yolo(
                img,
                conf=0.25,
                verbose=False
            )[0]

            if len(r.boxes):

                xy = r.boxes[0].xyxy[0].cpu().numpy().astype(int)

                x1,y1,x2,y2 = xy

                h,w = img.shape[:2]

                pad = 10

                x1 = max(0,x1-pad)
                y1 = max(0,y1-pad)

                x2 = min(w-1,x2+pad)
                y2 = min(h-1,y2+pad)

                crop = img[y1:y2,x1:x2]

        except:
            crop = None

    if crop is None:
        crop = east_detect(img)

    if crop is None:
        crop = contour_detect(img)

    crop_name = "crop_" + f.filename

    crop_path = os.path.join(
        RESULT,
        crop_name
    )

    cv2.imwrite(
        crop_path,
        crop if crop is not None else img
    )

    plate = ocr(crop) or ocr(img)

    red_plate = is_red_plate(crop)

    threat = False
    alert_message = ""

    if red_plate and not plate:

        plate = "OFFICIAL VEHICLE"

        alert_message = "VIP OFFICIAL VEHICLE DETECTED"

    elif plate in BLACKLIST:

        threat = True

        alert_message = f"BLACKLISTED VEHICLE DETECTED : {plate}"

    elif red_plate and plate:

        threat = True

        alert_message = f"SUSPICIOUS RED PLATE : {plate}"

    # ---------- Siren ----------
    if threat:

        threading.Thread(
            target=play_siren,
            daemon=True
        ).start()

    state = find_state(plate)

    # =========================
    # SAVE TO DATABASE
    # =========================
    now = datetime.now()

    date_now = now.strftime("%Y-%m-%d")
    time_now = now.strftime("%H:%M:%S")

    conn = sqlite3.connect(DB_NAME)

    cur = conn.cursor()

    cur.execute("""
    INSERT INTO scans
    (
        plate,
        state,
        threat,
        alert,
        image,
        date,
        time
    )

    VALUES (?, ?, ?, ?, ?, ?, ?)
    """,

    (
        plate,
        state,
        "YES" if threat else "NO",
        alert_message,
        crop_name,
        date_now,
        time_now
    ))

    conn.commit()
    conn.close()

    return render_template(
        "index.html",

        plate_text=plate if plate else "Unreadable",

        cropped_img_url=url_for(
            'result_file',
            filename=crop_name
        ),

        state=state,

        country="India" if state != "Unknown" else "Unknown",

        threat=threat,

        alert_message=alert_message
    )

# =========================
# HISTORY PAGE
# =========================
@app.route('/history')
def history():

    if 'user' not in session:
        return redirect('/login')

    conn = sqlite3.connect(DB_NAME)

    cur = conn.cursor()

    cur.execute("""
    SELECT * FROM scans
    ORDER BY id DESC
    """)

    data = cur.fetchall()

    conn.close()

    return render_template(
        "history.html",
        data=data
    )

@app.route('/stop_siren')
def stop_siren_route():

    stop_siren()

    return redirect('/')

@app.route('/results/<filename>')
def result_file(filename):

    return send_from_directory(
        RESULT,
        filename
    )

# ---------- Auto-open ----------
def open_browser():

    webbrowser.open(
        "http://127.0.0.1:5000/"
    )

if __name__=="__main__":

    threading.Timer(
        1,
        open_browser
    ).start()

    app.run(debug=True)
