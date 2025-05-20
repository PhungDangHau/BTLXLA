# app.py (hoàn chỉnh với YOLO, IPFS, Blockchain, CSV Export)
from flask import Flask, request, render_template, send_from_directory, send_file
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
import threading
import cv2
import uuid
import webbrowser
from web3 import Web3
import json
import requests
import datetime
import csv
import hashlib

# ========== Flask setup ==========
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'runs/detect/predict'
app.config['DASHBOARD_FOLDER'] = 'dashboard'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DASHBOARD_FOLDER'], exist_ok=True)

# ========== YOLO model ==========
model = YOLO('weights/best.pt')
video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
ESP32_STREAM_URL = "http://192.168.109.61/"

# ========== Web3 & Contract Setup ==========
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))
account = w3.eth.accounts[1]

with open("abi.json") as f:
    abi = json.load(f)

if not os.path.exists("contract_address.txt"):
    raise FileNotFoundError("❌ Lỗi: Chưa tìm thấy file contract_address.txt. Hãy tạo file này với địa chỉ contract đã deploy trong Remix.")
with open("contract_address.txt") as f:
    contract_address = f.read().strip()

contract = w3.eth.contract(address=contract_address, abi=abi)

# ========== IPFS Config ==========
PINATA_API_KEY = '395644cd1afc16115605'
PINATA_SECRET_API_KEY = '33f6e83cb40f066e4590f50a5a0733b0ccdcb209c35cec8361352f26c5c2ab0f'

def upload_to_ipfs(file_path):
    url = "https://api.pinata.cloud/pinning/pinFileToIPFS"
    headers = {
        "pinata_api_key": PINATA_API_KEY,
        "pinata_secret_api_key": PINATA_SECRET_API_KEY
    }
    with open(file_path, 'rb') as file:
        response = requests.post(url, files={"file": file}, headers=headers)
    if response.status_code == 200:
        return response.json()['IpfsHash']
    else:
        print("IPFS upload error:", response.text)
        return None

# ========== Export to CSV ==========
def export_to_csv(start_date=None, end_date=None):
    count = contract.functions.getRecordCount().call()
    with open("records.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "DateTime", "Rotten", "Fresh"])
        for i in range(count):
            ts, r, fsh = contract.functions.getRecord(i).call()
            dt = datetime.datetime.fromtimestamp(ts)
            if start_date and end_date:
                if not (start_date <= dt.date() <= end_date):
                    continue
            writer.writerow([ts, dt.strftime("%Y-%m-%d %H:%M:%S"), r, fsh])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/download-records')
def download_records():
    start = request.args.get('start')
    end = request.args.get('end')
    if start and end:
        start_date = datetime.datetime.strptime(start, "%Y-%m-%d").date()
        end_date = datetime.datetime.strptime(end, "%Y-%m-%d").date()
    else:
        start_date = end_date = None
    export_to_csv(start_date, end_date)
    return send_file("records.csv", as_attachment=True)

@app.route('/filter')
def filter_form():
    return render_template('filter_form.html')

@app.route('/esp32-live')
def esp32_live():
    threading.Thread(target=stream_yolo_from_esp32).start()
    webbrowser.open("http://127.0.0.1:5000/dashboard")
    return "\U0001f680 ESP32-CAM stream started."

def stream_yolo_from_esp32():
    cap = cv2.VideoCapture(ESP32_STREAM_URL)
    if not cap.isOpened():
        print("ESP32 not connected.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (320, 240))
        results = model(frame, conf=0.5)

        labels = []
        save_image = False

        if results[0].boxes.shape[0] > 0:
            for box in results[0].boxes:
                label = model.names[int(box.cls[0])]
                labels.append(label)
                if 'rotten' in label:
                    save_image = True

        if save_image:
            annotated_frame = results[0].plot()
            count_rotten = sum(1 for l in labels if 'rotten' in l)
            count_fresh = len(labels) - count_rotten

            try:
                tx = contract.functions.updateCounts(count_rotten, count_fresh).transact({'from': account})
                w3.eth.wait_for_transaction_receipt(tx)
            except Exception as e:
                print("Blockchain error:", e)

            filename = f"{uuid.uuid4().hex}.jpg"
            path = os.path.join(app.config['DASHBOARD_FOLDER'], filename)
            cv2.imwrite(path, annotated_frame)
            upload_to_ipfs(path)

        display_frame = annotated_frame if save_image else frame
        cv2.imshow("ESP32 + YOLO", display_frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/dashboard')
def dashboard():
    images = os.listdir(app.config['DASHBOARD_FOLDER'])
    images.sort(reverse=True)
    rotten_count = contract.functions.rottenCount().call()
    fresh_count = contract.functions.freshCount().call()
    return render_template('dashboard.html', images=images, rotten_count=rotten_count, fresh_count=fresh_count)

@app.route('/dashboard_img/<filename>')
def dashboard_image(filename):
    return send_from_directory(app.config['DASHBOARD_FOLDER'], filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/result/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)
@app.route('/compare-hash', methods=['GET', 'POST'])
def compare_hash():
    result = None
    computed_hash = None
    if request.method == 'POST':
        file = request.files['file']
        input_hash = request.form['input_hash'].strip().lower()
        if file:
            content = file.read()
            computed_hash = hashlib.sha256(content).hexdigest()
            result = "Khớp" if computed_hash == input_hash else "Không khớp"
    return render_template('compare_hash.html', result=result, computed_hash=computed_hash)

# ========== Chạy Flask ==========
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)