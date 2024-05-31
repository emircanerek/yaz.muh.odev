import cv2
import mediapipe as mp
import math
import os
import datetime
import time

# Kamera ayarları
camera = cv2.VideoCapture(0)
camera.set(3, 1280)  # Genişlik
camera.set(4, 720)  # Yükseklik

# MediaPipe
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
not_healthy_angle = 50
maybe_healthy_angle = 85
trigger_finger_angle_threshold = 60
trigger_finger_duration_threshold = 3  # 3 seconds
hand_open_close_duration_threshold = 2  # 2 seconds
finger_open_angle_threshold = 90

# Açıları hesaplama fonksiyonu
def calculate_angle(coords1, coords2, coords3):
    distance1 = math.dist(coords1, coords2)
    distance2 = math.dist(coords2, coords3)
    distance3 = math.dist(coords1, coords3)
    if distance1 * distance2 == 0:
        return 0
    angle = math.degrees(math.acos((distance1 ** 2 + distance2 ** 2 - distance3 ** 2) / (2 * distance1 * distance2)))
    return angle

# Landmark koordinatlarını elde etme fonksiyonu
def get_landmark_coords(landmark, img_width, img_height):
    return (landmark.x * img_width, landmark.y * img_height)

# Açılar hesaplama fonksiyonları
def calculate_thumb_angle(hand_landmarks, img_width, img_height):
    thumb_base = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.THUMB_CMC], img_width, img_height)
    thumb_mid = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.THUMB_IP], img_width, img_height)
    thumb_tip = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.THUMB_TIP], img_width, img_height)
    return calculate_angle(thumb_base, thumb_mid, thumb_tip)

def calculate_index_finger_angle(hand_landmarks, img_width, img_height):
    index_base = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_MCP], img_width, img_height)
    index_mid = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_PIP], img_width, img_height)
    index_tip = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP], img_width, img_height)
    return calculate_angle(index_base, index_mid, index_tip)

def calculate_middle_finger_angle(hand_landmarks, img_width, img_height):
    mid_base = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_MCP], img_width, img_height)
    mid_mid = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_DIP], img_width, img_height)
    mid_tip = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP], img_width, img_height)
    return calculate_angle(mid_base, mid_mid, mid_tip)

def calculate_ring_finger_angle(hand_landmarks, img_width, img_height):
    ring_base = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.RING_FINGER_PIP], img_width, img_height)
    ring_mid = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.RING_FINGER_DIP], img_width, img_height)
    ring_tip = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.RING_FINGER_TIP], img_width, img_height)
    return calculate_angle(ring_base, ring_mid, ring_tip)

def calculate_pinky_finger_angle(hand_landmarks, img_width, img_height):
    pinky_base = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.PINKY_PIP], img_width, img_height)
    pinky_mid = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.PINKY_DIP], img_width, img_height)
    pinky_tip = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.PINKY_TIP], img_width, img_height)
    return calculate_angle(pinky_base, pinky_mid, pinky_tip)

# El durumu tespiti fonksiyonu
def detect_hand_status(thumb_angle, index_angle, mid_angle, ring_angle, pinky_angle):
    if (thumb_angle < 90 or index_angle < not_healthy_angle or mid_angle < not_healthy_angle
            or ring_angle < not_healthy_angle or pinky_angle < not_healthy_angle):
        return "Hastasiniz"
    elif (thumb_angle < 110 or index_angle < maybe_healthy_angle
          or mid_angle < maybe_healthy_angle or ring_angle < maybe_healthy_angle or pinky_angle < maybe_healthy_angle):
        return "Hasta olabilirsiniz"
    else:
        return "Sagliklisiniz"

# Rapor dosyasını oluştur
report_folder = "C:/Raporlar"
if not os.path.exists(report_folder):
    os.makedirs(report_folder)

current_time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
report_file = os.path.join(report_folder, f"hand_report_{current_time}.txt")

# Rapor dosyasına yazma fonksiyonu
def write_to_report(report_file, thumb_angle, index_angle, mid_angle, ring_angle, pinky_angle, hand_status, trigger_finger_detected,hand_closed):
    with open(report_file, "w") as f:
        f.write("Thumb Angle: {}\n".format(thumb_angle))
        f.write("Index Finger Angle: {}\n".format(index_angle))
        f.write("Middle Finger Angle: {}\n".format(mid_angle))
        f.write("Ring Finger Angle: {}\n".format(ring_angle))
        f.write("Pinky Finger Angle: {}\n".format(pinky_angle))
        f.write("Hand Status: {}\n".format(hand_status))
        f.write("Trigger Finger: {}\n".format(trigger_finger_detected))
        f.write("Hand Closed Time: {}\n".format(hand_closed))
        f.close()

# Tetik Parmak ve El Açılıp Kapanma tespiti için veri yapıları
trigger_finger_start_time = None
trigger_finger_detected = False
hand_open_close_start_time = None
hand_closed_time = None
hand_open = False
hand_closed = False

while True:
    success, img = camera.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, channel = img.shape
    results = hands.process(imgRGB)

    # Ekrana basılacak tuş mesajını çiz
    cv2.putText(img, "Press 'n' to create a new report.", (25, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Açılar hesaplanır ve ekrana yazılır
            thumb_angle = calculate_thumb_angle(hand_landmarks, width, height)
            cv2.putText(img, f"Thumb_Angle: {int(thumb_angle)}", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            index_angle = calculate_index_finger_angle(hand_landmarks, width, height)
            cv2.putText(img, f"Index_Angle: {int(index_angle)}", (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mid_angle = calculate_middle_finger_angle(hand_landmarks, width, height)
            cv2.putText(img, f"Mid_Angle: {int(mid_angle)}", (25, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ring_angle = calculate_ring_finger_angle(hand_landmarks, width, height)
            cv2.putText(img, f"Ring_Angle: {int(ring_angle)}", (25, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            pinky_angle = calculate_pinky_finger_angle(hand_landmarks, width, height)
            cv2.putText(img, f"Pinky_Angle: {int(pinky_angle)}", (25, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if trigger_finger_detected or hand_closed:
                cv2.putText(img, f"Status: {hand_status}", (25, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # El durumu tespiti
            hand_status = detect_hand_status(thumb_angle, index_angle, mid_angle, ring_angle, pinky_angle)
            cv2.putText(img, f"Status: {hand_status}", (25, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Eklemleri çiz
            mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)

            # Tetik Parmak tespiti
            if any(angle < trigger_finger_angle_threshold for angle in [thumb_angle, index_angle, mid_angle, ring_angle, pinky_angle]):
                if not trigger_finger_start_time:
                    trigger_finger_start_time = time.time()
                elif time.time() - trigger_finger_start_time >= trigger_finger_duration_threshold:
                    trigger_finger_detected = True
                    cv2.putText(img, "Tetik Parmak Hastaligi Olabilir", (25, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                trigger_finger_start_time = None
                trigger_finger_detected = False

            # Elin açılıp kapanma hızını tespit etme
            angles = [thumb_angle, index_angle, mid_angle, ring_angle, pinky_angle]
            avg_angle = sum(angles) / len(angles)
            hand_closed_threshold = 100

            if avg_angle < hand_closed_threshold:  # El kapalı
                if not hand_closed:
                    hand_closed = True
                    hand_closed_time = time.time()
                elif time.time() - hand_closed_time > hand_open_close_duration_threshold:
                    cv2.putText(img, "El Yavas Acilip Kapaniyor: Romatizma, Artrit, Tuzak Noropati Olabilir ", (25, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:  # El açık
                if hand_closed:
                    hand_closed = False
                    hand_closed_time = None

            # Rapor dosyasına yaz
            write_to_report(report_file, thumb_angle, index_angle, mid_angle, ring_angle, pinky_angle, hand_status, trigger_finger_detected, hand_closed)

    cv2.imshow("Camera", img)

    # 'n' tuşuna basıldığında yeni bir rapor oluştur
    key = cv2.waitKey(1)
    if key == ord("n"):
        current_time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        report_file = os.path.join(report_folder, f"hand_report_{current_time}.txt")
        cv2.putText(img, "New report created. Press 'n' for another report.", (25, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Kamera Çıkış Tuşu
    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
