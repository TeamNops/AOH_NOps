import json
from flask import Flask, render_template, request,Response
from PIL import Image
import google.generativeai as genai
import os
import cv2
import mediapipe as mp
import numpy as np
import math
import base64
import io
import time
import threading
from dotenv import load_dotenv
import jsonify
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

app = Flask(__name__)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
def get_gemini_response(input, image, prompt):
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([input, image[0], prompt])
    return response.text
def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        image_parts = [{"mime_type": uploaded_file.content_type, "data": bytes_data}]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Index page route
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/signin', methods=['GET', 'POST'])
def signin():
  if request.method == 'POST':
    # Handle form submission (username, email, password)
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']
    # Add validation or logic here
    return f"Signin successful for {username}"  # Change this to actual logic
  else:
    return render_template('signin.html')


# Calories page route
@app.route('/calories', methods=['GET', 'POST'])
def calories():
    if request.method == 'POST':
        input_text = request.form['input']
        uploaded_file = request.files['file']
        #image = Image.open(uploaded_file)
        image_data = input_image_setup(uploaded_file)
        input_prompt = """
        You are an expert in nutritionist where you need to see the food items from the image
        and calculate the total calories, also provide the details of every food items with calories intake
        is below format
        
        1. Item 1 - no of calories
        2. Item 2 - no of calories
        also give amount of protiens,zinc,carbohydrates,fats
        ----
        ----
        """
        response = get_gemini_response(input_prompt, image_data, input_text)
        return render_template('calories.html', response=response)  # Pass 'response' to the template
    return render_template('calories.html', response=None)
#Planners
def get_gemini_response_planner(input_prompt):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(input_prompt)
    #parsed_data = json.loads(response.text)
    # diet_plan = parsed_data.get('Diet Plan', [])
    return response.text
@app.route('/planner', methods=['POST','GET'])
def planner():
    if request.method == 'POST':
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        age = int(request.form['age'])
        type = request.form['type']
        activity = request.form['activity']
        # input_prompt="""
        # You are an expert Nutritionist. 
        #     If the input contains weight,height,age,type wheather he is veg or non veg and activity that he does daily.
        #     Give me a Complete Food Diet for breakfast,lunch,evening snacks and dinner based on weight,height,age,type and
        #     activity.if he tells type as veg suggest all veg food  while if type as non veg suggest non veg food.
        #     Find the relation between weight,height and age ans suggest accordingly. 
        #     weight:{weight}
        # height:{height}
        # age:{age}
        # type:{type}
        # activity:{activity}
        # I want the response in one single string having the structure
        # {"Diet Plan":[],"calories":""}
        # """
        input_prompt="""
            You are an expert Nutritionist.

            Given the individual's weight, height, age, dietary preference (vegetarian/non-vegetarian), and daily activity level, you are tasked with devising a comprehensive food diet plan encompassing breakfast, lunch, evening snacks, and dinner. 

            Considering the individual's weight, height, and age, tailor the diet plan accordingly. If the dietary preference is specified as vegetarian, include only vegetarian food suggestions, while if it's non-vegetarian, suggest non-vegetarian options.

            Additionally, account for the relationship between weight, height, and age in formulating the diet plan.

            Details:
            Weight: {weight}
            Height: {height}
            Age: {age}
            Dietary Preference: {type}
            Daily Activity Level: {activity}

            Please provide the response in a single string structured as follows:
            {"Diet Plan":[],"calories":""}
            """

        diet_plan=get_gemini_response_planner(input_prompt)

        return render_template('planner.html', diet_plan=diet_plan)
    return render_template('planner.html',diet_pan=None)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from flask import Flask, Response
import cv2
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360-angle
    return angle 

def generate_frames():
    counter = 0
    stage = None
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to retrieve frame from camera.")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                angle = calculate_angle(shoulder, elbow, wrist)
                cv2.putText(image, str(angle), tuple(np.multiply(elbow, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                if angle > 160:
                    stage = "down"
                if angle < 30 and stage =='down':
                    stage="up"
                    counter +=1
                    print(counter)
                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')   
            except Exception as e:
                print("Error:", e)
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle

class VideoCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.mp_pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.counter = 0
        self.stage = None
        self.start_time = time.time()
        self.last_count_time = self.start_time
        self.calories_burned = 0
        self.last_counted_reps = 0
        self.lock = threading.Lock()

    def __del__(self):
        self.cap.release()

    def generate_frames(self):
        while True:
            with self.lock:
                ret, frame = self.cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = self.mp_pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                try:
                    landmarks = results.pose_landmarks.landmark

                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    angle = calculate_angle(shoulder, elbow, wrist)

                    if angle > 160:
                        self.stage = "down"
                    if angle < 30 and self.stage == 'down':
                        self.stage = "up"
                        self.counter += 1
                        current_time = time.time()
                        time_diff = current_time - self.last_count_time
                        if time_diff > 10:  # Consider 10 seconds for calorie calculation
                            self.calories_burned += 0.5 * (self.counter - self.last_counted_reps)  # Assuming each curl burns 0.5 calories
                            self.last_count_time = current_time
                            self.last_counted_reps = self.counter
                        print("Reps:", self.counter)
                        print("Calories burned:", self.calories_burned)

                except Exception as e:
                    print("Error:", e)

                cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

                cv2.putText(image, 'REPS', (15, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(self.counter),
                            (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, 'CALORIES', (240, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, "{:.2f}".format(self.calories_burned),
                            (240, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, 'STAGE', (550, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, self.stage,
                            (380, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

video_camera = VideoCamera()

def generate_frames():
    return video_camera.generate_frames()

@app.route('/VideoLive')
def VideoLive():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.3,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain 
@app.route('/bot', methods=['POST'])
def bot():
    user_input = request.form['user_input']
    response = genai.chat(messages=user_input)
    truncated_response = response.last[:100] if len(response.last) > 100 else response.last
    return jsonify({'response': truncated_response})
  
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    chat_history = []  # Initialize an empty list to store the chat history
    if request.method == 'POST':
        user_question = request.form['user_question']
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001")  # type: ignore
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question}, return_only_outputs=True)
        full_response = ''.join(response['output_text'])
        chat_history.append({'user': user_question, 'assistant':full_response})  # Add the current conversation to the chat history
        return render_template('chat.html', chat_history=chat_history)  # Pass the chat history to the template
    return render_template('chat.html', chat_history=chat_history)
if __name__ == "__main__":
    app.run(debug=True)



