from flask import Flask, render_template, redirect, request, session, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_session import Session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///stressapp.db'
db = SQLAlchemy(app)

app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Database models
class Employee(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    department = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    stress_level = db.Column(db.String(100))

    def __repr__(self):
        return f"Employee('{self.name}', '{self.email}')"
    
ADMIN_USERNAME = 'admin@gmail.com'
ADMIN_PASSWORD = 'admin'

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(Employee, int(user_id))

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register_employee', methods=['GET', 'POST'])
def register_employee():  
    if request.method == 'POST':
        name = request.form['name']
        department = request.form['department']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        
        employee = Employee(name=name, department=department, email=email, password=password)
        db.session.add(employee)
        db.session.commit()
        return render_template('admin_dashboard.html')
    
    return render_template('employee_registration.html')

@app.route('/admin/view_employees')
def view_employees():
    if not session.get('admin'):
        return redirect('/login')
    employees = Employee.query.all()
    return render_template('view_employees.html', employees=employees)

@app.route('/admin_dashboard')
def admin_dashboard():
    return render_template('admin_dashboard.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            return redirect('/admin_dashboard')
        else:
            employee = Employee.query.filter_by(email=username).first()
            if employee and check_password_hash(employee.password, password):
                login_user(employee)
                return redirect('/employee/profile')
    return render_template('login.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('super_user_logged_in', None)
    return redirect(url_for('index'))


@app.route('/employee/profile')
@login_required
def employee_profile():
    if current_user.is_authenticated:
        employee = current_user
        return render_template('employee_profile.html', employee=employee)
    return redirect('/login')

@app.route('/save_video', methods=['POST'])
def save_video():
    if 'video' in request.files and 'employeeName' in request.form:
        employee_name = request.form['employeeName']

        # Create the employee's folder if it doesn't exist
        if not os.path.exists('videos/' + employee_name):
            os.makedirs('videos/' + employee_name)

        # Get the number of existing videos for this employee
        num_existing_videos = len(os.listdir('videos/' + employee_name))

        # Save the video in the employee's folder with the appropriate name (video_1.webm, video_2.webm, etc.)
        video_filename = f'video_{num_existing_videos + 1}.webm'
        video = request.files['video']
        video.save(os.path.join('videos', employee_name, video_filename))

        return 'Video saved successfully!', 200
    else:
        return 'No video or employee name found in the request.', 400
    
def load_stress_model():
    return keras.models.load_model('models/stressmodel.h5')

def predict_stress_from_video(video_path):
    # Load the stress model
    model = load_stress_model()
    face_cascade_path = 'models/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    video_capture = cv2.VideoCapture(video_path)
    results = []
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
        for (x, y, w, h) in faces:
            face_img = gray_frame[y:y + h, x:x + w]
            resized_face_img = cv2.resize(face_img, (48, 48))
            test_image = keras.utils.img_to_array(resized_face_img)
            test_image = np.expand_dims(test_image, axis=0)
            output = model.predict(test_image)
            op_label = np.argmax(output, axis=1)                  
                
            stress = [0,1,2,4]
            if op_label in stress:
                text=False
            else:
                text=True
            results.append(text)
    return results

def calculate_percentage_of_values(lst):
    total_count = len(lst)
    true_count = lst.count(True)
    false_count = lst.count(False)
    print("True- False :===========: ",true_count,false_count)

    true_percentage = (true_count / total_count) * 100
    false_percentage = (false_count / total_count) * 100

    return true_percentage, false_percentage

@app.route('/predict_stress/<employee_name><employee_id>')
def predict_stress(employee_name,employee_id):
    video_folder = os.path.join('videos', employee_id)
    video_files = os.listdir(video_folder)
    stress = []
    tp=0
    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        stress_prediction = predict_stress_from_video(video_path)

        true_percentage, false_percentage = calculate_percentage_of_values(stress_prediction)
        tp = tp + true_percentage
        print(true_percentage)
        stress.append(true_percentage)
    
    print(len(video_files))
    print("stress:",stress)
    print("tp: ",tp)
    tp_avg= tp/len(video_files) 

    emp = Employee.query.get(employee_id)

      
    if tp_avg>10:
        emp.stress_level='Stressed'
        db.session.commit()
        return render_template('stressed.html',employee_name=employee_name)
    else:
        emp.stress_level='Normal'
        db.session.commit()
        return render_template('normal.html',employee_name=employee_name)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
