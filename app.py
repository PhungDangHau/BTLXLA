from flask import Flask, request, render_template, send_from_directory
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
import threading
import cv2
import uuid
import webbrowser

# Fix lỗi load weights YOLO
from torch.serialization import add_safe_globals
from ultralytics.nn.modules.conv import Conv
add_safe_globals([Conv])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'runs/detect/predict'
app.config['DASHBOARD_FOLDER'] = 'dashboard'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DASHBOARD_FOLDER'], exist_ok=True)

# Load YOLO model
model = YOLO('weights/best.pt')
video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

ESP32_STREAM_URL = "http://192.168.55.61/"  # 🔁 Thay bằng IP ESP32-CAM của bạn

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_files = request.files.getlist('files')
        original_images = []
        detection_results = []
        video_results = []

        for file in uploaded_files:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            _, ext = os.path.splitext(filename)
            ext = ext.lower()

            try:
                if ext in video_extensions:
                    model(file_path, save=True, stream=True, project='runs/detect', name='predict', exist_ok=True)
                    video_results.append(filename)
                else:
                    model(file_path, save=True, project='runs/detect', name='predict', exist_ok=True)
                    original_images.append(filename)
                    detection_results.append(filename)  # Map gốc -> detect cùng tên
            except Exception as e:
                print("LỖI YOLO:", e)

        zipped_images = zip(original_images, detection_results)

        return render_template('result.html',
                               zipped_images=zipped_images,
                               images=os.listdir(app.config['RESULT_FOLDER']),
                               videos=video_results)

    return render_template('index.html')

@app.route('/esp32-live')
def esp32_live():
    threading.Thread(target=stream_yolo_from_esp32).start()
    webbrowser.open("http://127.0.0.1:5000/dashboard")
    return "🚀 Đang xử lý luồng từ ESP32-CAM. Dashboard đang hiển thị kết quả!"

def stream_yolo_from_esp32():
    cap = cv2.VideoCapture(ESP32_STREAM_URL)

    if not cap.isOpened():
        print("❌ Không thể kết nối ESP32-CAM.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Không nhận được khung hình.")
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        filename = f"{uuid.uuid4().hex}.jpg"
        path = os.path.join(app.config['DASHBOARD_FOLDER'], filename)
        cv2.imwrite(path, annotated_frame)

        cv2.imshow("ESP32-CAM + YOLO", annotated_frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/dashboard')
def dashboard():
    images = os.listdir(app.config['DASHBOARD_FOLDER'])
    images.sort(reverse=True)
    return render_template('dashboard.html', images=images)

@app.route('/dashboard_img/<filename>')
def dashboard_image(filename):
    return send_from_directory(app.config['DASHBOARD_FOLDER'], filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/result/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
