from flask import Flask, render_template, Response
import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc, nullslast
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired


app=Flask(__name__)
camera = cv2.VideoCapture(0)

app.config['SQLALCHEMY_DATABASE_URI']="sqlite:///todo.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/images'
db=SQLAlchemy(app)

# To upload images in images folder
class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

# Table in database
class Todo(db.Model):
    sno = db.Column(db.Integer,primary_key=True)
    title=db.Column(db.String(200),nullable=True)
    apperance=db.Column(db.String(200),nullable=True)
    #date_created= db.Column(db.DateTime, default=datetime.utcnow)
    
path = 'static/images'
images = []
image_names = []
known_face_names = []
known_face_encodings = []

myList = os.listdir(path)

# Initialize some variables
face_locations = []
face_encodings = []
process_this_frame = True
size_of_myList = len(myList)
count = [0]*size_of_myList  

#clears database when app starts
Todo.query.delete()

counter1 =0

# Adds entries to database
for cu_img in myList:
    current_img = cv2.imread(f'{path}/{cu_img}')
    images.append(current_img)
    name_of_person = os.path.splitext(cu_img)[0]
    known_face_names.append(name_of_person)
    apperance = count[counter1]
    title = name_of_person
    todo1 = Todo(title=title , apperance= apperance)
    counter1 = counter1+1
    db.session.add(todo1)
    db.session.commit()



# Encodes images in images folder
def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList



known_face_encodings = faceEncodings(images)

def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
           
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    count[best_match_index]=count[best_match_index]+1
                face_names.append(name)
                
            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index1():
    return render_template('index.html')

@app.route('/criminals', methods=['GET',"POST"])
def criminals():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data # First grab the file
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # Then save the file
        return "File has been uploaded."
    return render_template('criminals.html', form=form)

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/criminalsDetected')
def criminalsDetected():
    counter=0
    for cu_imgs in known_face_names:
        todo = Todo.query.filter_by(sno=counter+1).first()
        todo.title = cu_imgs  
        todo.apperance = count[counter]
        counter=counter+1
        db.session.add(todo)
        db.session.commit()
    allTodo = Todo.query.all()
    return render_template('criminalsDetected.html', allTodo=allTodo)

if __name__=='__main__':
    app.run(debug=True)
