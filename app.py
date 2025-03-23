from flask import Flask, render_template, redirect, url_for
import threading
import cv2
import torch
import logging
import mediapipe as mp
from ultralytics import YOLO
import telepot
from datetime import datetime
import time
import requests
app = Flask(__name__)

# Bot Setup



# Get location using IP info API
def get_location():
    try:
        api_key = "https://ip-geolocation.whoisxmlapi.com/api/v1?apiKey=&ipAddress=8.8.8.8" 
        response = requests.get(f"https://ipinfo.io/json?token={api_key}")
        if response.status_code == 200:
            data = response.json()
            city = data.get("city", "")
            region = data.get("region", "")
            country = data.get("country", "")
            location = f"at location {city}, {region}, {country}"
            return location
    except Exception as e:
        print("Error fetching location:", e)
    return "at unknown location"

# Set cameralocation using API


chat_id = 5091908919
bot = telepot.Bot('7946773041:AAGTSCXF0hCDJT7kE6hiXSs1wdheff7RB5c')
cameralocation = get_location()

# YOLO setup
logging.getLogger("ultralytics").setLevel(logging.ERROR)
weapon_model = YOLO("weapondetect.pt")
gender_model = YOLO("best_y11.pt")
class_names = weapon_model.names

# State Variables
weapondetectalert = 0
sosdetectalert = 0
sosresetcount = 0
singlefemalealert = 0
mengroupalert = 0
prevweaponclassname = None

def detect():
    global weapondetectalert, sosdetectalert, sosresetcount, singlefemalealert, mengroupalert, prevweaponclassname

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    class_labels = {0: "Female", 1: "Male"}
    colors = {0: (255, 0, 0), 1: (0, 255, 0)}

    sos_count = 0
    thumbs_up_count = 0
    required_frames = 5

    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame = cv2.flip(frame, 1)

            results_weapon = weapon_model(frame, conf=0.5, verbose=False)
            for result in results_weapon:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    class_id = int(box.cls[0])
                    class_name = class_names.get(class_id, "Unknown")
                    label = class_name + " Weapon"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    if weapondetectalert == 0:
                        weapondetectalert = 1
                        msg = label + " detected " + cameralocation
                        cv2.imwrite("pic.jpg", frame)
                        bot.sendPhoto(chat_id=chat_id, photo=open('pic.jpg', 'rb'))
                        bot.sendMessage(chat_id=chat_id, text=msg)
                        prevweaponclassname = class_name
            if weapondetectalert == 1 and prevweaponclassname != class_name:
                weapondetectalert = 0

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_hands = hands.process(rgb_image)
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    thumb_tip = hand_landmarks.landmark[4]
                    pinky_tip = hand_landmarks.landmark[20]
                    thumb_pinky_distance = abs(thumb_tip.y - pinky_tip.y)
                    is_sos = thumb_pinky_distance < 0.1
                    index_mcp = hand_landmarks.landmark[5]
                    is_thumbs_up = thumb_tip.y < index_mcp.y
                    if is_sos:
                        sos_count += 1
                        thumbs_up_count = 0
                    elif is_thumbs_up:
                        thumbs_up_count += 1
                        sos_count = 0
                    else:
                        sos_count = 0
                        thumbs_up_count = 0
                    if sos_count >= required_frames:
                        cv2.putText(frame, "SOS Detected! Need Help!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        if sosdetectalert == 0:
                            sosdetectalert = 1
                            msg = "SOS ALERT detected " + cameralocation
                            cv2.imwrite("pic.jpg", frame)
                            bot.sendPhoto(chat_id=chat_id, photo=open('pic.jpg', 'rb'))
                            bot.sendMessage(chat_id=chat_id, text=msg)
                            if sosresetcount > 0:
                                sosresetcount = 0
                    elif thumbs_up_count >= required_frames:
                        cv2.putText(frame, "Thumbs Up, I'm Safe!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if sosdetectalert == 1:
                sosresetcount += 1
                if sosresetcount > 60:
                    sosdetectalert = 0

            results_gender = gender_model(frame, conf=0.7, verbose=False)
            male_count, female_count = 0, 0
            for result in results_gender:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    cls = int(box.cls[0])
                    if conf > 0.5:
                        label = f"{class_labels.get(cls, 'Unknown')} ({conf:.2f})"
                        color = colors.get(cls, (0, 255, 255))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        if cls == 0:
                            female_count += 1
                        elif cls == 1:
                            male_count += 1
            count_text = f"Females: {female_count} | Males: {male_count}"
            cv2.putText(frame, count_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if female_count >= 1 and male_count >= 3:
                if mengroupalert == 0:
                    msg = "More Men than female detected " + cameralocation
                    cv2.imwrite("pic.jpg", frame)
                    bot.sendPhoto(chat_id=chat_id, photo=open('pic.jpg', 'rb'))
                    bot.sendMessage(chat_id=chat_id, text=msg)
                    mengroupalert = 1
            else:
                if mengroupalert == 1:
                    mengroupalert = 0

            now = datetime.now()
            current_hour = now.hour
            if current_hour >= 22 or current_hour < 5:
                if female_count >= 1 and singlefemalealert == 0:
                    msg = "Lonely Female detected " + cameralocation
                    cv2.imwrite("pic.jpg", frame)
                    bot.sendPhoto(chat_id=chat_id, photo=open('pic.jpg', 'rb'))
                    bot.sendMessage(chat_id=chat_id, text=msg)
                    singlefemalealert = 1

            if singlefemalealert == 1 and female_count == 0:
                singlefemalealert = 0

            cv2.imshow("Women Safety Analytics", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    thread = threading.Thread(target=detect)
    thread.start()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
