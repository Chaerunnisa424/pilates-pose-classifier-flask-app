from flask import Flask, flash, redirect, request, session, jsonify, url_for, Response, render_template
from flask_mysqldb import MySQL
import bcrypt
from bs4 import BeautifulSoup
import requests
from indobert import SentimentAnalyzer
from model.model import get_response  # Mengimpor fungsi untuk mendapatkan respons dari model
import base64
from io import BytesIO
from collections import defaultdict
from flask_cors import CORS
import cv2
import pickle
import numpy as np
import mediapipe as mp
import math
import pygame  # Import pygame untuk memutar suara
from PIL import Image, ImageDraw, ImageFont
import os
from werkzeug.utils import secure_filename
from datetime import datetime


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change to a secure secret key

# MySQL configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_DB'] = 'pilapose'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)

# Load the Sentiment Analyzer model for IndoBert (Not used in this part)
model_indobert = 'model'
analyzer_indobert = SentimentAnalyzer(model_indobert)




import pygame
import pickle
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp
from flask import Flask, render_template, Response, request
from flask_mysqldb import MySQL
import os

# Inisialisasi pygame mixer untuk pemutaran suara
pygame.mixer.init()

# Memuat model klasifikasi pose
def load_model(model_path):
    """Memuat model machine learning dari file pickle."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Mendapatkan keypoints dari frame video menggunakan MediaPipe
def get_keypoints_from_frame(frame):
    """Ekstraksi keypoints dari frame menggunakan MediaPipe."""
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            keypoints = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark])
            return keypoints, results.pose_landmarks
        else:
            print("No landmarks detected.")
    return None, None

# Menambahkan teks ke frame menggunakan PIL
def put_text_with_pil(frame, text, position, font_path, font_size, color, shadow_offset=(2, 2), shadow_color=(0, 0, 0)):
    """Menambahkan teks ke frame menggunakan PIL."""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()  # Menggunakan font default jika font khusus tidak ditemukan

    # Menambahkan bayangan pada teks
    draw.text((position[0] + shadow_offset[0], position[1] + shadow_offset[1]), text, font=font, fill=shadow_color)
    draw.text(position, text, font=font, fill=color)
    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return frame

# Klasifikasi pose dari kamera
def classify_pose_from_camera(model, class_labels):
    """Klasifikasi pose menggunakan input dari kamera."""
    cap = cv2.VideoCapture(0)  # Menggunakan kamera default

    if not cap.isOpened():
        print("Error: Tidak dapat membuka kamera.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    font_path = "C:/Windows/Fonts/times.ttf"  # Ganti sesuai path font yang ada di sistem Anda
    sound_wrong = "D:/PHB/TEKNIK INFORMATIKA/SEMESTER 5/CAPSTONE PROJECT/PILAPOSE/flask-pilates/static/sounds/error.mp3"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints, pose_landmarks = get_keypoints_from_frame(frame)

        if keypoints is not None:
            keypoints = keypoints.flatten().reshape(1, -1)
            prediction = model.predict_proba(keypoints)
            predicted_class_idx = np.argmax(prediction)
            predicted_class_name = class_labels[predicted_class_idx]
            predicted_probability = prediction[0][predicted_class_idx]

            if predicted_probability > 0.8:
                color = (0, 255, 0)  # Green for Correct
                status_text = "Correct"
            else:
                color = (0, 0, 255)  # Red for Incorrect
                status_text = "Incorrect"
                # Memutar suara jika pose salah
                if os.path.exists(sound_wrong):  # Pastikan file ada
                    pygame.mixer.Sound(sound_wrong).play()  # Play the sound when pose is incorrect
                else:
                    print(f"Error: File {sound_wrong} tidak ditemukan.")

            frame = put_text_with_pil(frame, f"Probability: {predicted_probability*100:.2f}%", (50, 100), font_path, 30, color)
            frame = put_text_with_pil(frame, f"Status: {status_text}", (50, 150), font_path, 30, color)

            # Menggambar landmark pose pada frame
            mp.solutions.drawing_utils.draw_landmarks(
                frame, 
                pose_landmarks, 
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

        cv2.imshow("Pose Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Mendapatkan detail pose berdasarkan ID
def get_pose_details(pose_id):
    """Mengambil detail pose berdasarkan ID."""
    pose_data = {
        1: {'nama_pose': 'BirdDog-Pose', 'gambar': 'gambar1.jpg', 'durasi': 30},
        2: {'nama_pose': 'Cobra-Pose', 'gambar': 'gambar2.jpg', 'durasi': 40},
    }
    return pose_data.get(pose_id, None)

# Rute untuk halaman klasifikasi pose
@app.route('/classification', methods=['GET'])
def classification():
    """Menampilkan halaman klasifikasi pose."""
    pose_id = request.args.get('pose_id')
    if not pose_id:
        return "Pose ID tidak diberikan!", 400

    pose_details = get_pose_details(int(pose_id))
    if not pose_details:
        return "Pose tidak ditemukan!", 404

    return render_template('classification.html', pose_details=pose_details)

# Fungsi untuk streaming video dengan format MJPEG
def generate_frames(model, class_labels):
    """Stream frame video."""
    cap = cv2.VideoCapture(0)  # Coba menggunakan kamera lain jika kamera default gagal
    if not cap.isOpened():
        print("Error: Kamera tidak ditemukan atau tidak dapat dibuka.")
        return

    while True:
        ret, frame = cap.read()  # Membaca frame
        if not ret:
            print("Error: Tidak dapat membaca frame.")
            break

        # Ekstraksi keypoints dari frame
        keypoints, pose_landmarks = get_keypoints_from_frame(frame)
        if keypoints is not None:
            # Prediksi pose (gunakan model untuk prediksi, jika perlu)
            keypoints = keypoints.flatten().reshape(1, -1)
            prediction = model.predict_proba(keypoints)
            predicted_class_idx = np.argmax(prediction)
            predicted_class_name = class_labels[predicted_class_idx]
            predicted_probability = prediction[0][predicted_class_idx]

            # Tentukan warna berdasarkan probabilitas
            if predicted_probability > 0.8:
                color = (0, 255, 0)  # Hijau untuk probabilitas > 90%
            else:
                color = (0, 0, 255)  # Merah untuk probabilitas < 90%

            # Menambahkan teks nama pose dan confidence ke frame
            cv2.putText(frame, f"Probability: {predicted_probability*100:.2f}%", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Menggambar landmark pada frame dengan warna berdasarkan probabilitas
            if pose_landmarks:  # Pastikan landmark ada
                drawing_spec_landmarks = mp.solutions.drawing_utils.DrawingSpec(color=color, thickness=2, circle_radius=3)
                drawing_spec_connections = mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2)
                
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, 
                    pose_landmarks, 
                    mp.solutions.pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec_landmarks,
                    connection_drawing_spec=drawing_spec_connections
                )

        # Mengonversi frame ke format JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        # Mengirim frame dalam format multipart/x-mixed-replace untuk streaming MJPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    cap.release()  # Melepaskan kamera setelah selesai

# Rute untuk video feed
@app.route('/video_feed')
def video_feed():
    """Menyajikan streaming video untuk klasifikasi pose."""
    model_weights = r'model-1.pkl'  # Ganti dengan path model yang sebenarnya
    class_labels = ['BirdDog-Pose', 'BridgeOnBall-Pose', 'CatCowStretch-Pose', 'ChestLift-Pose', 'Cobra-Pose',
                    'HamstringSwissBall-Pose', 'OverheadReachOnBall-Pose', 'PelvicCurl-Pose', 'ReversePlank-Pose', 'V-UpBall-Pose']
    model = load_model(model_weights)  # Memuat model
    return Response(generate_frames(model, class_labels),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Fungsi untuk menyimpan hasil klasifikasi ke dalam database
def save_classification_to_db(user_id, exercise_id, pose_probability):
    """Menyimpan hasil klasifikasi ke dalam database MySQL."""
    cursor = mysql.connection.cursor()

    query = """
    INSERT INTO pose_classification_results (user_id, exercise_id, pose_probability)
    VALUES (%s, %s, %s)
    """
    cursor.execute(query, (user_id, exercise_id, pose_probability))
    mysql.connection.commit()
    cursor.close()




# Route utama (main route)
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/programs')
def show_programs():
    cur = mysql.connection.cursor()
    cur.execute("SELECT id, program_name, image, duration_days FROM programs")
    programs = cur.fetchall() 
    
    # Encode gambar dengan base64
    for program in programs:
        program['image'] = base64.b64encode(program['image']).decode('utf-8')  # Encode gambar dan ubah menjadi string

    cur.close()
    return render_template('programs.html', programs=programs)


@app.route('/programs/<int:program_id>/poses')
def show_program_poses(program_id):
    cursor = mysql.connection.cursor()

    # Ambil data program berdasarkan program_id untuk mendapatkan nama program
    cursor.execute("SELECT program_name, duration_days FROM programs WHERE id = %s", (program_id,))
    program = cursor.fetchone()

    if not program:
        return "Program not found", 404  # Jika program tidak ditemukan, kembalikan error

    # Ambil data poses untuk program berdasarkan program_id
    cursor.execute("""
        SELECT id, pose_name, duration_secs, day, calories, image 
        FROM poses WHERE program_id = %s
    """, (program_id,))
    poses = cursor.fetchall()

    # Encode gambar dengan base64
    for pose in poses:
        pose['image'] = base64.b64encode(pose['image']).decode('utf-8')  # Encode gambar dan ubah menjadi string

    cursor.close()

    # Kelompokkan poses berdasarkan hari
    poses_by_day = defaultdict(list)
    for pose in poses:
        poses_by_day[pose['day']].append(pose)

    return render_template('poses.html', program=program, poses_by_day=poses_by_day)


@app.route('/timer/<int:pose_id>')
def timer(pose_id):
    try:
        user_id = session.get('user_id')  # Ambil user_id dari sesi
        if not user_id:
            return "User not logged in", 401

        cursor = mysql.connection.cursor()

        # Ambil data pose
        cursor.execute("SELECT * FROM poses WHERE id = %s", (pose_id,))
        pose = cursor.fetchone()
        if not pose:
            return f"Pose with ID {pose_id} not found", 404

        # Encode gambar jika ada
        if pose.get('image'):
            pose['image'] = base64.b64encode(pose['image']).decode('utf-8')

        # Ambil program terkait pose
        cursor.execute("SELECT program_name, duration_days FROM programs WHERE id = %s", (pose['program_id'],))
        program = cursor.fetchone()
        if not program:
            return f"Program with ID {pose['program_id']} not found", 404

        # Ambil progres pengguna
        cursor.execute("""
            SELECT current_day, completed
            FROM user_progress
            WHERE user_id = %s AND program_id = %s
        """, (user_id, pose['program_id']))
        progress = cursor.fetchone()

        if not progress:
            # Tambahkan progres baru untuk hari pertama jika belum ada
            cursor.execute("""
                INSERT INTO user_progress (user_id, program_id, current_day, total_calories_burned, completed)
                VALUES (%s, %s, %s, %s, %s)
            """, (user_id, pose['program_id'], 1, 0, False))
            mysql.connection.commit()
            progress = {'current_day': 1, 'completed': False}

        # Cek apakah ini adalah pose terakhir dalam hari ini untuk program tertentu
        cursor.execute("""
            SELECT id FROM poses
            WHERE program_id = %s AND day = %s
            ORDER BY id DESC LIMIT 1
        """, (pose['program_id'], pose['day']))
        last_pose_in_day = cursor.fetchone()

        is_last_pose = (last_pose_in_day and pose['id'] == last_pose_in_day['id'])

        # Ambil pose berikutnya hanya dalam hari dan program yang sama
        cursor.execute("""
            SELECT * FROM poses
            WHERE program_id = %s AND day = %s AND id > %s
            ORDER BY id ASC LIMIT 1
        """, (pose['program_id'], pose['day'], pose_id))
        next_pose = cursor.fetchone()

        # Encode gambar pose berikutnya jika ada
        if next_pose and next_pose.get('image'):
            next_pose['image'] = base64.b64encode(next_pose['image']).decode('utf-8')

        cursor.close()

        return render_template(
            'timer.html',
            pose=pose,
            program=program,
            progress=progress,
            next_pose=next_pose,
            is_last_pose=is_last_pose  # Kirim informasi apakah pose terakhir
        )
    except Exception as e:
        app.logger.error(f"Unexpected error in /timer: {str(e)}")
        return "An unexpected error occurred", 500




@app.route('/save_progress', methods=['POST'])
def save_progress():
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"status": "error", "message": "User not logged in"}), 401

        data = request.get_json()
        program_id = data.get('program_id')
        pose_id = data.get('pose_id')
        completed = data.get('completed', False)

        # Validasi input
        if not isinstance(program_id, int) or program_id <= 0:
            return jsonify({"status": "error", "message": "Invalid program_id"}), 400
        if not isinstance(pose_id, int) or pose_id <= 0:
            return jsonify({"status": "error", "message": "Invalid pose_id"}), 400

        cursor = mysql.connection.cursor()

        # Ambil day berdasarkan pose_id
        cursor.execute("""
            SELECT day, program_id
            FROM poses
            WHERE id = %s AND program_id = %s
        """, (pose_id, program_id))
        pose_data = cursor.fetchone()
        app.logger.info(f"Pose validation result: {pose_data}")
        if not pose_data:
            return jsonify({"status": "error", "message": f"Invalid pose_id={pose_id} or program_id={program_id}"}), 400

        current_day = pose_data['day']
        program_id_from_pose = pose_data['program_id']

        if program_id != program_id_from_pose:
            return jsonify({"status": "error", "message": "Pose does not belong to the specified program"}), 400

        # Hitung total kalori
        cursor.execute("""
            SELECT SUM(calories) AS total_calories
            FROM poses
            WHERE program_id = %s AND day = %s
        """, (program_id, current_day))
        calories_data = cursor.fetchone()
        total_calories = calories_data['total_calories'] if calories_data['total_calories'] else 0

        # Simpan progres
        cursor.execute("""
            INSERT INTO user_progress (user_id, program_id, current_day, total_calories_burned, completed)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                current_day = VALUES(current_day),
                total_calories_burned = user_progress.total_calories_burned + VALUES(total_calories_burned),
                completed = VALUES(completed)
        """, (user_id, program_id, current_day, total_calories, completed))

        mysql.connection.commit()
        cursor.close()

        return jsonify({
            "status": "success",
            "message": "Progress saved successfully",
            "current_day": current_day,
            "total_calories_burned": total_calories,
            "completed": completed
        }), 200

    except Exception as e:
        app.logger.error(f"Error in save_progress: {str(e)}")
        return jsonify({"status": "error", "message": "An error occurred while saving progress"}), 500





@app.route('/exercise')
def exercise():
    cur = mysql.connection.cursor()
    
    # Ambil hanya kolom yang diperlukan
    cur.execute("SELECT id, nama_pose, gambar, tipe_pose, tipe_latihan, video_url FROM exercises")
    exercises = cur.fetchall()  # Menyimpan hasil query
    
    # Encode gambar dengan base64
    for i, pose in enumerate(exercises):
        if pose['gambar']:
            exercises[i]['gambar'] = base64.b64encode(pose['gambar']).decode('utf-8')
    
    cur.close()
    return render_template('exercise.html', exercises=exercises)

@app.route('/exercise_video/<int:pose_id>')
def exercise_video(pose_id):
    cur = mysql.connection.cursor()
    cur.execute("SELECT id, nama_pose, video_url, instruksi FROM exercises WHERE id = %s", (pose_id,))
    pose = cur.fetchone()
    cur.close()

    if pose:
        # Ubah URL YouTube ke format embed
        if pose['video_url']:
            pose['video_url'] = pose['video_url'].replace('watch?v=', 'embed/')

        # Ubah instruksi dari string JSON menjadi dict
        try:
            import json
            pose['instruksi'] = json.loads(pose['instruksi'])  # Parse JSON dari database
        except (json.JSONDecodeError, TypeError):
            pose['instruksi'] = {"persiapan": [], "langkah-langkah": []}  # Fallback jika parsing gagal

        return render_template('exercise_video.html', pose=pose)
    else:
        return "Video tidak ditemukan.", 404


# @app.route('/timer')
# def timer():
#     return render_template('timer.html')

# Route untuk chatbot
@app.route('/chatbot')
def flexbot():
    return render_template('chatbot.html')  # Pastikan file chatbot.html ada di folder templates

# Route untuk mendapatkan respons dari chatbot
@app.route('/get_response', methods=['POST'])
def get_response_route():
    user_message = request.json.get('message')  # Ambil pesan dari JSON request
    bot_response = get_response(user_message)  # Dapatkan respons dari fungsi get_response
    return jsonify({'response': bot_response})  # Kirim respons dalam format JSON


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/healthtips')
def health_tips():
    """Menampilkan daftar artikel dari database."""
    try:
        cur = mysql.connection.cursor()
        # Mengambil ID, judul, author, sumber artikel, dan gambar (jika ada)
        cur.execute("SELECT id, title, author, source, image FROM articles ORDER BY created_at DESC")
        articles = cur.fetchall()
        cur.close()
        
        # Mengubah gambar (MEDIUMBLOB) menjadi base64 untuk ditampilkan di HTML
        for article in articles:
            if article['image']:
                # Encode gambar menjadi base64
                article['image_url'] = 'data:image/jpeg;base64,' + base64.b64encode(article['image']).decode('utf-8')
            else:
                article['image_url'] = 'static/assets/images/img_1.jpg'  # Default image jika tidak ada gambar
            
        return render_template('healthtips.html', articles=articles)
    except Exception as e:
        print(f"Kesalahan saat mengambil artikel: {e}")
        return "Terjadi kesalahan saat mengambil data artikel.", 500


@app.route('/article/<int:article_id>')
def article(article_id):
    """Menampilkan detail artikel berdasarkan ID."""
    try:
        # Menggunakan cursor biasa tanpa parameter dictionary=True
        cur = mysql.connection.cursor()
        
        # Mengambil artikel berdasarkan ID
        cur.execute("SELECT * FROM articles WHERE id = %s", (article_id,))
        article = cur.fetchone()
        cur.close()

        # Jika artikel ditemukan
        if article:
            # Cek apakah ada gambar dalam bentuk binary (BLOB)
            if article['image']:
                # Mengonversi gambar dari BLOB ke base64
                article['image'] = base64.b64encode(article['image']).decode('utf-8')
            
            # Menyajikan artikel di template
            return render_template('article.html', article=article)

        # Jika artikel tidak ditemukan
        return "Artikel tidak ditemukan.", 404

    except Exception as e:
        # Menangani error jika ada kesalahan dalam query atau lainnya
        print(f"Kesalahan saat mengambil artikel berdasarkan ID: {e}")
        return "Terjadi kesalahan saat mengambil data artikel. Silakan coba lagi nanti.", 500





@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        # Getting form or JSON data
        username = request.form.get('username') if request.form.get('username') else request.json.get('username')
        email = request.form.get('email') if request.form.get('email') else request.json.get('email')
        password = request.form.get("password") if request.form.get("password") else request.json.get('password')
        confirm_password = request.form.get("confirm_password") if request.form.get("confirm_password") else request.json.get('confirm_password')

        # Validate form
        if not username or not email or not password or not confirm_password:
            if request.is_json:
                return jsonify({"error": "Semua kolom harus diisi"}), 400
            flash("Semua kolom harus diisi", "danger")
            return redirect(url_for('register'))

        if password != confirm_password:
            if request.is_json:
                return jsonify({"error": "Password tidak cocok"}), 400
            flash("Password tidak cocok", "danger")
            return redirect(url_for('register'))

        # Hash password with bcrypt
        try:
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        except Exception as e:
            if request.is_json:
                return jsonify({"error": f"Terjadi kesalahan saat hashing password: {str(e)}"}), 500
            flash(f"Terjadi kesalahan saat hashing password: {str(e)}", "danger")
            return redirect(url_for('register'))

        # Save user data to database
        cur = mysql.connection.cursor()
        try:
            # Check if email is already registered
            cur.execute("SELECT * FROM users WHERE email = %s", [email])
            if cur.fetchone():
                if request.is_json:
                    return jsonify({"error": "Email sudah terdaftar"}), 400
                flash("Email sudah terdaftar", "danger")
                return redirect(url_for('register'))

            # Insert user data
            cur.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)", 
                        (username, email, hashed_password.decode('utf-8')))
            mysql.connection.commit()

            # Set the flash message to notify success
            flash("Pengguna berhasil mendaftar! Silakan login.", "success")

            # Store session variables
            session['username'] = username
            session['email'] = email

            # Redirect to login page
            return redirect(url_for('login'))  # Redirect to login page

        except Exception as e:
            mysql.connection.rollback()
            if request.is_json:
                return jsonify({"error": f"Terjadi kesalahan: {str(e)}"}), 500
            flash(f"Terjadi kesalahan: {str(e)}", "danger")
            return redirect(url_for('register'))
        finally:
            cur.close()

    # If GET request, render register page
    return render_template('register.html')




@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.headers.get('Content-Type') == 'application/json':
            data = request.get_json()
            username_or_email = data.get('username_or_email')
            password = data.get('password')
        else:
            username_or_email = request.form.get('username_or_email')
            password = request.form.get('password')

        if not username_or_email or not password:
            flash("Username/Email dan password harus diisi", "danger")
            return redirect(url_for('login'))

        # Tidak menggunakan dictionary=True, karena DictCursor sudah diatur di konfigurasi
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email=%s OR username=%s", (username_or_email, username_or_email))
        user = cur.fetchone()
        cur.close()

        if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['email'] = user['email']

            flash(f"Selamat datang, {user['username']}!", "success")
            return redirect(url_for('home'))
        else:
            flash("Username atau password salah", "danger")
            return redirect(url_for('login'))

    return render_template('login.html')




@app.route('/home', methods=["GET", "POST"])
def home():
    if 'user_id' not in session:  # Pastikan user_id tersedia di sesi
        flash("Silakan login terlebih dahulu", "warning")
        return redirect(url_for('login'))

    user_id = session['user_id']
    username = session['username']

    if request.method == "POST":
        review_text = request.form.get('review')

        if review_text:
            # Analisis sentimen
            predicted_class, probabilities = analyzer_indobert.predict_sentiment(review_text)
            sentiment = "Positif" if predicted_class == 1 else "Negatif"

            # Simpan ulasan ke database
            cur = mysql.connection.cursor()
            cur.execute("""
                INSERT INTO user_reviews (user_id, text, sentiment) 
                VALUES (%s, %s, %s)
            """, (user_id, review_text, sentiment))
            mysql.connection.commit()
            cur.close()

            flash("Ulasan berhasil ditambahkan!", "success")
        else:
            flash("Ulasan tidak boleh kosong!", "danger")

        return redirect(url_for('home'))

    if request.method == "GET":
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT text, sentiment, created_at 
            FROM user_reviews 
            WHERE user_id = %s 
            ORDER BY created_at DESC
        """, (user_id,))
        reviews = cur.fetchall()
        cur.close()

        # Hitung distribusi sentimen
        positive_count = sum(1 for review in reviews if review['sentiment'] == 'Positif')
        negative_count = sum(1 for review in reviews if review['sentiment'] == 'Negatif')

        return render_template(
            'home.html',
            username=username,
            reviews=reviews,
            sentiment_counts={
                "positive": positive_count,
                "negative": negative_count
            }
        )



@app.route('/profile', methods=['GET', 'POST'])
def profile():
    # Pastikan pengguna sudah login
    if 'email' not in session:
        return redirect(url_for('login'))

    email = session['email']
    cur = mysql.connection.cursor()

    # Ambil data pengguna berdasarkan email
    cur.execute("SELECT * FROM users WHERE email = %s", [email])
    user = cur.fetchone()

    if not user:
        return redirect(url_for('home'))

    # Ambil semua program
    cur.execute("SELECT id, program_name, duration_days FROM programs")
    programs = cur.fetchall()

    # Ambil data progres pengguna
    progress_data = []
    for program in programs:
        program_id = program['id']
        program_name = program['program_name']
        duration_days = program['duration_days']

        # Pastikan setiap hari dari 1 hingga duration_days dimasukkan
        for day in range(1, duration_days + 1):
            cur.execute("""
                SELECT 
                    up.total_calories_burned, 
                    up.completed
                FROM 
                    user_progress up
                WHERE 
                    up.program_id = %s 
                    AND up.user_id = %s 
                    AND up.current_day = %s
            """, (program_id, user['id'], day))
            progress = cur.fetchone()

            # Tambahkan data ke progress_data, entri default jika tidak ditemukan
            progress_data.append({
                'program_id': program_id,
                'program_name': program_name,
                'current_day': day,
                'total_calories_burned': progress['total_calories_burned'] if progress else 0,
                'completed': progress['completed'] if progress else False
            })

    # Tambahkan status program (sudah selesai atau belum)
    completed_programs = [
        program['program_name'] for program in programs
        if all(
            day['completed'] for day in progress_data
            if day['program_name'] == program['program_name']
        )
    ]

    success_message = None
    error_message = None

    # Tangani pembaruan profil
    if request.method == 'POST':
        new_name = request.form.get('name', user['username'])
        new_email = request.form.get('email', user['email'])
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')

        try:
            # Perbarui nama
            if new_name and new_name != user['username']:
                cur.execute("UPDATE users SET username = %s WHERE email = %s", [new_name, email])
                session['username'] = new_name  # Perbarui sesi

            # Perbarui email
            if new_email and new_email != user['email']:
                cur.execute("UPDATE users SET email = %s WHERE email = %s", [new_email, email])
                session['email'] = new_email  # Perbarui sesi

            # Perbarui password
            if current_password and new_password:
                # Verifikasi password lama
                if not bcrypt.checkpw(current_password.encode('utf-8'), user['password'].encode('utf-8')):
                    error_message = "Password lama salah"
                elif new_password != confirm_password:
                    error_message = "Password baru dan konfirmasi password tidak cocok"
                else:
                    # Hash password baru dan perbarui
                    hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
                    cur.execute("UPDATE users SET password = %s WHERE email = %s", [hashed_password.decode('utf-8'), email])
                    success_message = "Password berhasil diperbarui"

            if not error_message:
                mysql.connection.commit()
                success_message = success_message or "Profil berhasil diperbarui"
        except Exception as e:
            mysql.connection.rollback()
            error_message = f"Terjadi kesalahan: {str(e)}"

    cur.close()

    return render_template(
        'profile.html',
        user=user,
        progress_data=progress_data,
        programs=programs,
        success_message=success_message,
        error_message=error_message,
        completed_programs=completed_programs
    )

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('index'))

#=========================================================================Admin Routes==================================================================================================#





@app.route('/admin')
def index_admin(): 
    return render_template('admin/login_admin.html')

# Route to view users
@app.route('/admin/users')
def users_admin():
    try:
        # Membuka koneksi ke database
        cur = mysql.connection.cursor()
        
        # Menjalankan query untuk mendapatkan data pengguna
        cur.execute("SELECT * FROM users")
        users = cur.fetchall()
        
        # Menutup cursor setelah digunakan
        cur.close()
        
        # Mengirimkan data pengguna ke template 'users_admin.html'
        return render_template('admin/users_admin.html', users=users)
    
    except Exception as e:
        # Menangani error dan memberikan pesan flash
        flash(f"Error fetching users: {str(e)}", "error")
        
        # Mengarahkan kembali ke halaman index admin jika terjadi error
        return redirect(url_for('index_admin'))


# Route to add a new user
@app.route('/add_user', methods=['POST'])
def add_user():
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')

    if not username or not email or not password:
        flash("All fields are required!", "error")
        return redirect(url_for('users_admin'))

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    try:
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)", (username, email, hashed_password))
        mysql.connection.commit()
        cur.close()
        flash("User added successfully!", "success")
    except Exception as e:
        flash(f"Error adding user: {str(e)}", "error")
    return redirect(url_for('users_admin'))

# Route to edit a user
@app.route('/edit_user', methods=['POST'])
def edit_user():
    user_id = request.form.get('id')
    username = request.form.get('username')
    email = request.form.get('email')

    if not user_id or not username or not email:
        flash("All fields are required!", "error")
        return redirect(url_for('users_admin'))

    try:
        cur = mysql.connection.cursor()
        cur.execute("UPDATE users SET username=%s, email=%s WHERE id=%s", (username, email, user_id))
        mysql.connection.commit()
        cur.close()
        flash("User updated successfully!", "success")
    except Exception as e:
        flash(f"Error updating user: {str(e)}", "error")
    return redirect(url_for('users_admin'))

# Route to view exercises
@app.route('/admin/exercises')
def exercises_admin():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM exercises")
    exercises = cur.fetchall()
    cur.close()
    return render_template('admin/exercises_admin.html', exercises=exercises)

@app.route('/add_exercise', methods=['POST'])
def add_exercise():
    if request.method == 'POST':
        nama_pose = request.form['nama_pose']
        gambar = request.files['gambar']
        tipe_pose = request.form['tipe_pose']
        tipe_latihan = request.form['tipe_latihan']
        video_url = request.form['video_url']
        instruksi = request.form['instruksi']
        
        cur = mysql.connection.cursor()
        cur.execute("""
            INSERT INTO exercises (nama_pose, gambar, tipe_pose, tipe_latihan, video_url, instruksi)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (nama_pose, gambar.filename, tipe_pose, tipe_latihan, video_url, instruksi))
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('exercises_admin'))

@app.route('/edit_exercise', methods=['POST'])
def edit_exercise():
    if request.method == 'POST':
        exercise_id = request.form['id']
        nama_pose = request.form['nama_pose']
        gambar = request.files['gambar']
        tipe_pose = request.form['tipe_pose']
        tipe_latihan = request.form['tipe_latihan']
        video_url = request.form['video_url']
        instruksi = request.form['instruksi']
        
        cur = mysql.connection.cursor()
        cur.execute("""
            UPDATE exercises 
            SET nama_pose=%s, gambar=%s, tipe_pose=%s, tipe_latihan=%s, video_url=%s, instruksi=%s
            WHERE id=%s
        """, (nama_pose, gambar.filename, tipe_pose, tipe_latihan, video_url, instruksi, exercise_id))
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('exercises_admin'))

@app.route('/delete_exercise/<int:id>')
def delete_exercise(id):
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM exercises WHERE id=%s", (id,))
    mysql.connection.commit()
    cur.close()
    return redirect(url_for('exercises_admin'))

@app.route('/admin/poses')
def poses_admin():
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM poses")
        poses = cur.fetchall()
        
        cur.execute("SELECT id, program_name FROM programs")  # Mengambil daftar program
        programs = cur.fetchall()
        
        cur.close()
        return render_template('admin/poses_admin.html', poses=poses, programs=programs)
    except Exception as e:
        flash(f"Error fetching poses or programs: {str(e)}", "error")
        return redirect(url_for('index_admin'))

@app.route('/add_pose', methods=['POST'])
def add_pose():
    try:
        if request.method == 'POST':
            program_id = request.form['program_id']
            pose_name = request.form['pose_name']
            duration_secs = request.form['duration_secs']
            day = request.form['day']
            gambar = request.files['image']
            calories = request.form['calories']
            
            cur = mysql.connection.cursor()
            cur.execute("INSERT INTO poses (program_id, pose_name, duration_secs, day, image, calories) VALUES (%s, %s, %s, %s, %s, %s)", 
                        (program_id, pose_name, duration_secs, day, gambar.filename, calories))
            mysql.connection.commit()
            cur.close()
            flash("Pose added successfully", "success")
            return redirect(url_for('poses_admin'))
    except Exception as e:
        flash(f"Error adding pose: {str(e)}", "error")
        return redirect(url_for('poses_admin'))

@app.route('/edit_pose', methods=['POST'])
def edit_pose():
    try:
        if request.method == 'POST':
            pose_id = request.form['id']
            program_id = request.form['program_id']
            pose_name = request.form['pose_name']
            duration_secs = request.form['duration_secs']
            day = request.form['day']
            gambar = request.files['image']
            calories = request.form['calories']
            
            cur = mysql.connection.cursor()
            cur.execute("UPDATE poses SET program_id=%s, pose_name=%s, duration_secs=%s, day=%s, image=%s, calories=%s WHERE id=%s", 
                        (program_id, pose_name, duration_secs, day, gambar.filename, calories, pose_id))
            mysql.connection.commit()
            cur.close()
            flash("Pose updated successfully", "success")
            return redirect(url_for('poses_admin'))
    except Exception as e:
        flash(f"Error updating pose: {str(e)}", "error")
        return redirect(url_for('poses_admin'))

@app.route('/delete_pose/<int:id>')
def delete_pose(id):
    try:
        cur = mysql.connection.cursor()
        cur.execute("DELETE FROM poses WHERE id=%s", (id,))
        mysql.connection.commit()
        cur.close()
        flash("Pose deleted successfully", "success")
        return redirect(url_for('poses_admin'))
    except Exception as e:
        flash(f"Error deleting pose: {str(e)}", "error")
        return redirect(url_for('poses_admin'))


@app.route('/admin/programs')
def programs_admin():
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM programs")
        programs = cur.fetchall()
        cur.close()
        return render_template('admin/programs_admin.html', programs=programs)
    except Exception as e:
        flash(f"Error fetching programs: {str(e)}", "error")
        return redirect(url_for('index_admin'))

@app.route('/add_program', methods=['POST'])
def add_program():
    try:
        if request.method == 'POST':
            program_name = request.form['program_name']
            gambar = request.files['image']
            duration_days = request.form['duration_days']
            
            cur = mysql.connection.cursor()
            cur.execute("INSERT INTO programs (program_name, image, duration_days) VALUES (%s, %s, %s)", 
                        (program_name, gambar.filename, duration_days))
            mysql.connection.commit()
            cur.close()
            flash("Program added successfully", "success")
            return redirect(url_for('programs_admin'))
    except Exception as e:
        flash(f"Error adding program: {str(e)}", "error")
        return redirect(url_for('programs_admin'))

@app.route('/edit_program', methods=['POST'])
def edit_program():
    try:
        if request.method == 'POST':
            program_id = request.form['id']
            program_name = request.form['program_name']
            gambar = request.files['image']
            duration_days = request.form['duration_days']
            
            cur = mysql.connection.cursor()
            cur.execute("UPDATE programs SET program_name=%s, image=%s, duration_days=%s WHERE id=%s", 
                        (program_name, gambar.filename, duration_days, program_id))
            mysql.connection.commit()
            cur.close()
            flash("Program updated successfully", "success")
            return redirect(url_for('programs_admin'))
    except Exception as e:
        flash(f"Error updating program: {str(e)}", "error")
        return redirect(url_for('programs_admin'))

@app.route('/delete_program/<int:id>')
def delete_program(id):
    try:
        cur = mysql.connection.cursor()
        cur.execute("DELETE FROM programs WHERE id=%s", (id,))
        mysql.connection.commit()
        cur.close()
        flash("Program deleted successfully", "success")
        return redirect(url_for('programs_admin'))
    except Exception as e:
        flash(f"Error deleting program: {str(e)}", "error")
        return redirect(url_for('programs_admin'))


# Fungsi untuk scraping artikel
def scrape_article(url):
    """Scrape judul dan isi artikel dari URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Pastikan request berhasil
        soup = BeautifulSoup(response.text, 'html.parser')

        # Mengambil judul artikel
        title = soup.find('h1').text.strip() if soup.find('h1') else "Judul tidak ditemukan"

        # Mengambil konten utama artikel
        content_wrapper = soup.find('div', class_='unique-content-wrapper')
        if content_wrapper:
            paragraphs = content_wrapper.find_all('p')
            content = "\n\n".join([p.text.strip() for p in paragraphs if p.text.strip()])
        else:
            content = "Konten tidak ditemukan."

        return {"title": title, "content": content}

    except requests.exceptions.RequestException as e:
        print(f"Terjadi kesalahan saat scraping: {e}")
        return {"title": "Artikel tidak tersedia", "content": ""}

# Fungsi untuk menyimpan artikel ke database
def save_article_to_db(title, url, content, source='scraped'):
    """Simpan artikel ke database."""
    cur = mysql.connection.cursor()
    cur.execute("""
        INSERT INTO articles (title, url, content, source, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (title, url, content, source, datetime.now(), datetime.now()))
    mysql.connection.commit()
    cur.close()

# Route untuk melihat dan menambahkan artikel
@app.route('/admin/articles', methods=['GET', 'POST'])
def articles_admin():
    if request.method == 'POST':
        # Ambil URL artikel dari formulir
        url = request.form['url']
        scraped_data = scrape_article(url)
        
        # Simpan artikel yang telah di-scrape ke database
        save_article_to_db(
            title=scraped_data['title'],
            url=url,
            content=scraped_data['content'],
            source='scraped'
        )
        flash('Article scraped and added successfully!', 'success')
        return redirect(url_for('articles_admin'))

    # Ambil semua artikel dari database
    cursor = mysql.connection.cursor()
    cursor.execute('SELECT * FROM articles')
    articles = cursor.fetchall()
    cursor.close()
    return render_template('admin/articles_admin.html', articles=articles)

# Route untuk menambahkan artikel secara manual
@app.route('/add_article', methods=['GET', 'POST'])
def add_article():
    if request.method == 'POST':
        title = request.form['title']
        url = request.form['url']
        content = request.form['content']
        author = request.form['author']
        source = request.form['source']
        created_at = request.form['created_at']
        updated_at = request.form['updated_at']

        # Mengonversi tanggal ke format datetime
        created_at = datetime.strptime(created_at, '%Y-%m-%dT%H:%M')
        updated_at = datetime.strptime(updated_at, '%Y-%m-%dT%H:%M')

        # Menambahkan artikel baru ke database
        cursor = mysql.connection.cursor()
        cursor.execute('''
            INSERT INTO articles (title, url, content, author, source, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        ''', (title, url, content, author, source, created_at, updated_at))
        mysql.connection.commit()
        cursor.close()
        flash('Article added successfully!', 'success')
        return redirect(url_for('articles_admin'))

    return render_template('add_article.html')

# Route untuk mengedit artikel
@app.route('/edit_article/<int:id>', methods=['GET', 'POST'])
def edit_article(id):
    cursor = mysql.connection.cursor()
    cursor.execute('SELECT * FROM articles WHERE id = %s', [id])
    article = cursor.fetchone()
    cursor.close()

    if request.method == 'POST':
        title = request.form['title']
        url = request.form['url']
        content = request.form['content']
        author = request.form['author']
        source = request.form['source']
        created_at = datetime.strptime(request.form['created_at'], '%Y-%m-%dT%H:%M')
        updated_at = datetime.strptime(request.form['updated_at'], '%Y-%m-%dT%H:%M')

        cursor = mysql.connection.cursor()
        cursor.execute('''
            UPDATE articles SET title = %s, url = %s, content = %s, author = %s,
            source = %s, created_at = %s, updated_at = %s WHERE id = %s
        ''', (title, url, content, author, source, created_at, updated_at, id))
        mysql.connection.commit()
        cursor.close()
        flash('Article updated successfully!', 'success')
        return redirect(url_for('articles_admin'))

    return render_template('edit_article.html', article=article)

# Route untuk menghapus artikel
@app.route('/delete_article/<int:id>', methods=['GET'])
def delete_article(id):
    cursor = mysql.connection.cursor()
    cursor.execute('DELETE FROM articles WHERE id = %s', [id])
    mysql.connection.commit()
    cursor.close()
    flash('Article deleted successfully!', 'danger')
    return redirect(url_for('articles_admin'))

@app.route('/admin/sentiment')
def sentiment_admin():
    try:
        cur = mysql.connection.cursor()
        # Query untuk mengambil review dan sentiment
        cur.execute("SELECT text, sentiment FROM user_reviews")
        sentiment_data = cur.fetchall()
        cur.close()

        # Menghitung jumlah sentimen Positif dan Negatif
        positive_count = sum(1 for row in sentiment_data if row['sentiment'].strip().lower() == 'positif')
        negative_count = sum(1 for row in sentiment_data if row['sentiment'].strip().lower() == 'negatif')

        return render_template('admin/sentiment_admin.html', sentiment_data=sentiment_data, positive_count=positive_count, negative_count=negative_count)
    except Exception as e:
        flash(f"Error fetching sentiment data: {str(e)}", "error")
        return redirect(url_for('index_admin'))


@app.route('/admin/login', methods=["GET", "POST"])
def login_admin():
    if request.method == "POST":
        username_or_email = request.form.get('username_or_email')
        password = request.form.get('password')

        if not username_or_email or not password:
            flash("Username/Email dan password diperlukan", "error")
            return redirect(url_for('login_admin'))

        try:
            cur = mysql.connection.cursor()
            cur.execute("SELECT * FROM admin WHERE email=%s OR username=%s", (username_or_email, username_or_email))
            user = cur.fetchone()
            cur.close()

            if user and password == user['password']:
                session['username'] = user['username']
                session['email'] = user['email']
                return redirect(url_for('home_admin'))
            else:
                flash("Username/email atau password tidak valid", "error")
                return redirect(url_for('login_admin'))
        except Exception as e:
            flash(f"Error during login: {str(e)}", "error")
            return redirect(url_for('login_admin'))

    return render_template('admin/login_admin.html')


@app.route('/admin/home')
def home_admin():
    # Cek apakah user sudah login
    if 'username' not in session:
        flash("Silakan login terlebih dahulu", "error")
        return redirect(url_for('login_admin'))

    try:
        cur = mysql.connection.cursor()  # Hasil query sebagai dictionary

        # Menghitung jumlah total users
        cur.execute("SELECT COUNT(id) AS total FROM users")
        total_users = cur.fetchone()['total']

        # Menghitung jumlah total reviews
        cur.execute("SELECT COUNT(id) AS total FROM user_reviews WHERE id IS NOT NULL AND id != ''")
        total_reviews = cur.fetchone()['total']

        # Menghitung jumlah total programs
        cur.execute("SELECT COUNT(id) AS total FROM programs")
        total_programs = cur.fetchone()['total']

        # Menghitung jumlah total poses
        cur.execute("SELECT COUNT(id) AS total FROM poses")
        total_poses = cur.fetchone()['total']

        # Menghitung jumlah total exercises
        cur.execute("SELECT COUNT(id) AS total FROM exercises")
        total_exercises = cur.fetchone()['total']

        # Menghitung jumlah total articles
        cur.execute("SELECT COUNT(id) AS total FROM articles")
        total_articles = cur.fetchone()['total']

        cur.close()

        # Render halaman home_admin.html dengan data
        return render_template(
            'admin/home_admin.html',
            total_users=total_users,
            total_reviews=total_reviews,
            total_programs=total_programs,
            total_poses=total_poses,
            total_exercises=total_exercises,
            total_articles=total_articles
        )
    except Exception as e:
        # Tangani error dengan aman
        flash(f"Error fetching data: {str(e)}", "error")
        return redirect(url_for('login_admin'))






#========================================================================flutter Routes====================================================================================================#


@app.route('/api/register', methods=["POST"])
def register_api():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get("password")
    confirm_password = data.get("confirm_password")

    # Validasi input
    if not username or not email or not password or not confirm_password:
        return jsonify({"status": False, "message": "Semua kolom harus diisi"}), 400

    if password != confirm_password:
        return jsonify({"status": False, "message": "Password tidak cocok"}), 400

    try:
        # Hash password menggunakan bcrypt
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        cur = mysql.connection.cursor()

        # Cek apakah email sudah terdaftar
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        if cur.fetchone():
            cur.close()
            return jsonify({"status": False, "message": "Email sudah terdaftar"}), 400

        # Insert data pengguna baru ke database
        cur.execute(
            "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
            (username, email, hashed_password.decode('utf-8'))
        )
        mysql.connection.commit()
        cur.close()

        return jsonify({"status": True, "message": "Pengguna berhasil terdaftar"}), 201

    except Exception as e:
        return jsonify({"status": False, "message": f"Terjadi kesalahan: {str(e)}"}), 500


# Route untuk login
@app.route('/api/login', methods=["POST"])
def login_api():
    data = request.get_json()
    
    # Validasi data input
    if not isinstance(data, dict) or 'username_or_email' not in data or 'password' not in data:
        return jsonify({"status": False, "message": "Data tidak valid"}), 400
    
    username_or_email = data.get('username_or_email')
    password = data.get('password')

    if not username_or_email or not password:
        return jsonify({"status": False, "message": "Semua kolom harus diisi"}), 400

    try:
        cur = mysql.connection.cursor()
        cur.execute(
            "SELECT * FROM users WHERE email=%s OR username=%s LIMIT 1",
            (username_or_email, username_or_email),
        )
        user = cur.fetchone()
        cur.close()

        if user:
            # Validasi password dengan bcrypt
            if bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
                # Simpan data pengguna ke session
                session['user_id'] = user['id']
                session['username'] = user['username']
                print(f"User {session['user_id']} logged in.")  # Debug log
                
                return jsonify({
                    "status": True,
                    "message": "Login berhasil",
                    "data": {
                        "id": user['id'],
                        "username": user['username']
                    }
                }), 200
            else:
                return jsonify({"status": False, "message": "Password salah"}), 400
        else:
            return jsonify({"status": False, "message": "Username/email tidak ditemukan"}), 400

    except Exception as e:
        print(f"Error saat login: {e}")  # Debug log
        return jsonify({"status": False, "message": "Terjadi kesalahan pada server"}), 500


@app.route('/api/users/upload_profile_image', methods=['POST'])
def upload_profile_image_api():
    try:
        print("Request diterima di endpoint /api/users/upload_profile_image")

        if 'profile_image' not in request.files:
            print("File tidak ditemukan dalam request")
            return jsonify({'error': 'No file part'}), 400

        file = request.files['profile_image']
        if file.filename == '':
            print("Tidak ada file yang dipilih")
            return jsonify({'error': 'No selected file'}), 400

        user_id = request.form.get('id')
        if not user_id:
            print("ID pengguna tidak disediakan")
            return jsonify({'error': 'User ID not provided'}), 400

        print(f"File diterima: {file.filename}, ID pengguna: {user_id}")

        # Simpan file ke direktori lokal
        file_path = f'uploads/{user_id}.jpg'
        os.makedirs('uploads', exist_ok=True)
        file.save(file_path)
        print(f"File berhasil disimpan di: {file_path}")

        # Respons sukses
        return jsonify({'success': 'File uploaded successfully'}), 200

    except Exception as e:
        print(f"Terjadi error: {str(e)}")
        return jsonify({'error': f"Server error: {str(e)}"}), 500


@app.route('/api/users/delete_profile_image', methods=['POST'])
def delete_profile_image():
    try:
        print("Request diterima di endpoint /api/users/delete_profile_image")

        # Ambil ID pengguna dari request
        user_id = request.form.get('id')
        if not user_id:
            print("ID pengguna tidak disediakan")
            return jsonify({'error': 'User ID not provided'}), 400

        print(f"ID pengguna yang diterima: {user_id}")

        # Hapus file dari direktori lokal
        file_path = f'uploads/{user_id}.jpg'
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File {file_path} berhasil dihapus")
        else:
            print(f"File {file_path} tidak ditemukan, melanjutkan proses")

        # Update database untuk mengganti kolom image menjadi NULL
        conn = get_db_connection()
        cursor = conn.cursor()

        query = "UPDATE users SET image = NULL WHERE id = %s"
        cursor.execute(query, (user_id,))
        conn.commit()

        print(f"Database berhasil diperbarui untuk ID pengguna: {user_id}")

        # Tutup koneksi database
        cursor.close()
        conn.close()

        # Respons sukses
        return jsonify({'success': 'Foto profil berhasil dihapus'}), 200

    except Exception as e:
        print(f"Terjadi error: {str(e)}")
        return jsonify({'error': f"Server error: {str(e)}"}), 500






@app.route('/update_username', methods=['POST'])
def update_username():
    data = request.json
    user_id = data.get('id')
    new_username = data.get('username')

    if not user_id or not new_username:
        return jsonify({'error': 'ID dan username harus diisi'}), 400

    connection = None
    cursor = None
    try:
        # Koneksi ke database menggunakan db_config
        connection = pymysql.connect(**db_config)
        cursor = connection.cursor()

        # Query untuk memperbarui username
        query = "UPDATE users SET username = %s WHERE id = %s"
        cursor.execute(query, (new_username, user_id))
        connection.commit()

        # Cek apakah ada baris yang diubah
        if cursor.rowcount == 0:
            return jsonify({'error': 'User ID tidak ditemukan'}), 404

        return jsonify({'message': 'Username berhasil diperbarui'}), 200
    except Exception as e:
        print(f'Error: {e}')
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500
    finally:
        # Tutup cursor dan koneksi jika sudah dibuat
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@app.route('/get_user', methods=['GET'])
def get_user():
    user_id = request.args.get('id')  # Ambil ID pengguna dari query parameter

    if not user_id:
        return jsonify({'error': 'ID pengguna harus disediakan'}), 400

    connection = None
    cursor = None
    try:
        # Koneksi ke database
        connection = pymysql.connect(**db_config)
        cursor = connection.cursor(pymysql.cursors.DictCursor)

        # Query untuk mendapatkan data pengguna berdasarkan ID
        query = "SELECT id, username FROM users WHERE id = %s"
        cursor.execute(query, (user_id,))
        user = cursor.fetchone()

        if not user:
            return jsonify({'error': 'Pengguna tidak ditemukan'}), 404

        return jsonify(user), 200  # Kembalikan data pengguna dalam format JSON
    except Exception as e:
        print(f'Error: {e}')
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()






# Route untuk home
@app.route('/api/home', methods=["GET"])
def home_api():
    if 'username' not in session:
        return jsonify({"error": "Pengguna belum login"}), 401
    return jsonify({"message": f"Selamat datang, {session['username']}!"}), 200


# Route untuk logout
@app.route('/api/logout', methods=["POST"])
def logout_api():
    session.clear()
    return jsonify({"message": "Logout berhasil"}), 200


# Route untuk mendapatkan data semua pengguna
@app.route('/api/users', methods=["GET"])
def get_users_api():
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT username, email FROM users")  # Sesuaikan kolom yang ingin diambil
        users = cur.fetchall()
        cur.close()
        
        # Kembalikan data dalam format JSON
        return jsonify(users), 200

    except Exception as e:
        return jsonify({"status": False, "message": f"Terjadi kesalahan: {str(e)}"}), 500
    


@app.route('/api/users/profile', methods=["GET", "POST"])
def manage_user_profile():
    try:
        # Jika GET, ambil data profil pengguna
        if request.method == "GET":
            # Ambil ID dari query string
            user_id = request.args.get('id', type=int)  # Pastikan konsisten dengan 'id'
            if not user_id:
                return jsonify({"status": False, "message": "ID tidak diberikan"}), 400

            # Query ke database
            cur = mysql.connection.cursor()
            cur.execute("SELECT username, email, profile_image FROM users WHERE id = %s", (user_id,))
            user = cur.fetchone()
            cur.close()

            if user:
                return jsonify({
                    "status": True,
                    "data": {
                        "username": user[0],
                        "email": user[1],
                        "profile_image": user[2] or ""  # Default ke string kosong jika None
                    }
                }), 200
            else:
                return jsonify({"status": False, "message": "Pengguna tidak ditemukan"}), 404

        # Jika POST, perbarui data profil pengguna
        elif request.method == "POST":
            data = request.json
            user_id = data.get("id")
            username = data.get("username")
            email = data.get("email")
            profile_image = data.get("profile_image")  # URL atau path ke gambar

            if not user_id:
                return jsonify({"status": False, "message": "ID tidak diberikan"}), 400

            if not username and not email and not profile_image:
                return jsonify({"status": False, "message": "Tidak ada data untuk diperbarui"}), 400

            # Query untuk memperbarui profil
            query = "UPDATE users SET "
            params = []
            if username:
                query += "username = %s, "
                params.append(username)
            if email:
                query += "email = %s, "
                params.append(email)
            if profile_image:
                query += "profile_image = %s, "
                params.append(profile_image)

            query = query.rstrip(", ")  # Hapus koma terakhir
            query += " WHERE id = %s"
            params.append(user_id)

            # Eksekusi query
            cur = mysql.connection.cursor()
            cur.execute(query, tuple(params))
            mysql.connection.commit()
            cur.close()

            return jsonify({"status": True, "message": "Profil berhasil diperbarui"}), 200

    except Exception as e:
        return jsonify({"status": False, "message": f"Terjadi kesalahan: {str(e)}"}), 500




@app.route('/api/users/profile', methods=["PUT"])
def update_user_profile():
    try:
        # Ambil data dari JSON request
        data = request.json
        user_id = data.get('id')  # Ambil 'id' dari JSON
        username = data.get('username')
        email = data.get('email')  # Email opsional

        # Validasi ID
        if not user_id or not isinstance(user_id, int):
            return jsonify({"status": False, "message": "ID tidak valid"}), 400

        # Validasi username
        if not username or not isinstance(username, str) or username.strip() == "":
            return jsonify({"status": False, "message": "Username tidak valid"}), 400

        # Ambil email lama jika tidak diberikan
        cur = mysql.connection.cursor()
        if not email:
            cur.execute("SELECT email FROM users WHERE id = %s", (user_id,))
            existing_email = cur.fetchone()
            email = existing_email[0] if existing_email else ""

        # Perbarui data ke database
        cur.execute("""
            UPDATE users 
            SET username = %s, email = %s
            WHERE id = %s
        """, (username, email, user_id))
        mysql.connection.commit()
        cur.close()

        return jsonify({"status": True, "message": "Profil berhasil diperbarui"}), 200

    except Exception as e:
        return jsonify({"status": False, "message": f"Terjadi kesalahan: {str(e)}"}), 500






@app.route('/api/users/upload_profile_image', methods=['POST'])
def upload_profile_image():
    try:
        print("Request diterima di endpoint /api/users/upload_profile_image")

        # Cek apakah file ada di request
        if 'profile_image' not in request.files:
            print("File tidak ditemukan dalam request")
            return jsonify({'error': 'No file part'}), 400

        file = request.files['profile_image']
        if file.filename == '':
            print("Tidak ada file yang dipilih")
            return jsonify({'error': 'No selected file'}), 400

        # Ambil ID user dari request
        user_id = request.form.get('id')
        if not user_id:
            print("ID pengguna tidak disediakan dalam request")
            return jsonify({'error': 'User ID not provided'}), 400

        print(f"File diterima: {file.filename}, ID pengguna: {user_id}")

        # Simpan file ke direktori lokal
        file_path = f'uploads/{user_id}.jpg'
        os.makedirs('uploads', exist_ok=True)  # Buat folder jika belum ada
        file.save(file_path)
        print(f"File berhasil disimpan di direktori: {file_path}")

        # Baca file sebagai binary data
        with open(file_path, 'rb') as f:
            image_data = f.read()
        print("File berhasil dibaca sebagai binary data")

        # Simpan ke database
        conn = get_db_connection()
        cursor = conn.cursor()
        query = "UPDATE users SET image = %s WHERE id = %s"
        cursor.execute(query, (image_data, user_id))
        conn.commit()
        print("File berhasil disimpan ke database")

        # Tutup koneksi
        cursor.close()
        conn.close()
        print("Koneksi database ditutup")

        # Berikan respons sukses
        return jsonify({'success': 'File uploaded successfully'}), 200

    except mysql.connector.Error as db_error:
        # Tangkap error terkait MySQL
        print(f"Database error: {db_error}")
        return jsonify({'error': f'Database error: {db_error}'}), 500

    except Exception as e:
        # Tangkap error lainnya
        print(f"Server error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500



@app.route('/api/users/login', methods=["POST"])
def login_users_api():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    # Proses validasi user
    cur = mysql.connection.cursor()
    cur.execute("SELECT id, username, email FROM users WHERE username=%s AND password=%s", (username, password))
    user = cur.fetchone()
    cur.close()

    if user:
        print(f"Login berhasil, ID: {user[0]}")  # Log ID
        return jsonify({
            "status": True,
            "data": {
                "id": user[0],
                "username": user[1],
                "email": user[2]
            }
        }), 200
    else:
        print("Login gagal: Username atau password salah")
        return jsonify({"status": False, "message": "Username atau password salah"}), 401




# Jalankan aplikasi
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

