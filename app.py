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
model = YOLO('weights/best.pt')  # Đường dẫn model
video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

ESP32_STREAM_URL = "http://192.168.55.61/"  # Thay IP ESP32-CAM của bạn

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
                    detection_results.append(filename)
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

        # Resize để giảm delay
        frame = cv2.resize(frame, (320, 240))

        # Nhận diện
        results = model(frame, conf=0.5)

        save_image = False
        rotten_detected = False  # ✅ Biến kiểm tra có phát hiện rotten

        if results[0].boxes.shape[0] > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                if label == 'rottenoranges':
                    save_image = True
                    rotten_detected = True
                    break

        if save_image:
            annotated_frame = results[0].plot()

            filename = f"{uuid.uuid4().hex}.jpg"
            path = os.path.join(app.config['DASHBOARD_FOLDER'], filename)
            cv2.imwrite(path, annotated_frame)

        # Nếu phát hiện rotten thì overlay màu đỏ
        if rotten_detected:
            if save_image:
                display_frame = annotated_frame.copy()
            else:
                display_frame = frame.copy()

            overlay = display_frame.copy()
            red_overlay = (0, 0, 255)  # BGR đỏ
            cv2.rectangle(overlay, (0, 0), (overlay.shape[1], overlay.shape[0]), red_overlay, -1)
            alpha = 0.3
            display_frame = cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0)
        else:
            display_frame = annotated_frame if save_image else frame

        cv2.imshow("ESP32-CAM + YOLO Detection", display_frame)

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
