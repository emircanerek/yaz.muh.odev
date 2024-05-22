import cv2
import mediapipe as mp
import math
import os
import datetime

# Kamera ayarları
camera = cv2.VideoCapture(0)
camera.set(3, 1280)  # Genişlik
camera.set(4, 720)  # Yükseklik

# MediaPipe
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Açıları hesaplama fonksiyonu
def calculate_angle(coords1, coords2, coords3):
    # İki nokta arasındaki mesafeleri hesapla
    distance1 = math.dist(coords1, coords2)
    distance2 = math.dist(coords2, coords3)
    distance3 = math.dist(coords1, coords3)

    # Eğer mesafe sıfırsa, açı hesaplanamaz
    if distance1 * distance2 == 0:
        return 0

    # Kosinüs teoremi kullanarak açıyı hesapla
    angle = math.degrees(math.acos((distance1 ** 2 + distance2 ** 2 - distance3 ** 2) / (2 * distance1 * distance2)))
    return angle

# Landmark koordinatlarını elde etme fonksiyonu
def get_landmark_coords(landmark, img_width, img_height):
    return (landmark.x * img_width, landmark.y * img_height)

# Başparmak açısını hesaplama fonksiyonu
def calculate_thumb_angle(hand_landmarks, img_width, img_height):
    thumb_base = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.THUMB_CMC], img_width, img_height)
    thumb_mid = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.THUMB_IP], img_width, img_height)
    thumb_tip = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.THUMB_TIP], img_width, img_height)
    return calculate_angle(thumb_base, thumb_mid, thumb_tip)

# İşaret parmağı açısını hesaplama fonksiyonu
def calculate_index_finger_angle(hand_landmarks, img_width, img_height):
    index_base = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_MCP], img_width, img_height)
    index_mid = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_PIP], img_width, img_height)
    index_tip = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP], img_width, img_height)
    return calculate_angle(index_base, index_mid, index_tip)

# Orta parmak açısını hesaplama fonksiyonu
def calculate_middle_finger_angle(hand_landmarks, img_width, img_height):
    mid_base = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_MCP], img_width, img_height)
    mid_mid = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_DIP], img_width, img_height)
    mid_tip = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP], img_width, img_height)
    return calculate_angle(mid_base, mid_mid, mid_tip)

# Yüzük parmağı açısını hesaplama fonksiyonu
def calculate_ring_finger_angle(hand_landmarks, img_width, img_height):
    ring_base = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.RING_FINGER_PIP], img_width, img_height)
    ring_mid = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.RING_FINGER_DIP], img_width, img_height)
    ring_tip = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.RING_FINGER_TIP], img_width, img_height)
    return calculate_angle(ring_base, ring_mid, ring_tip)

# Serçe parmağı açısını hesaplama fonksiyonu
def calculate_pinky_finger_angle(hand_landmarks, img_width, img_height):
    pinky_base = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.PINKY_PIP], img_width, img_height)
    pinky_mid = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.PINKY_DIP], img_width, img_height)
    pinky_tip = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.PINKY_TIP], img_width, img_height)
    return calculate_angle(pinky_base, pinky_mid, pinky_tip)

# El durumu tespiti fonksiyonu
def detect_hand_status(thumb_angle, index_angle, mid_angle, ring_angle, pinky_angle):
    if thumb_angle < 30 and index_angle < 30 and mid_angle < 30 and ring_angle < 30 and pinky_angle < 30:
        return "Hastasiniz"
    elif thumb_angle < 60 and index_angle < 60 and mid_angle < 60 and ring_angle < 60 and pinky_angle < 60:
        return "Hasta olabilirsiniz"
    else:
        return "Sagliklisiniz"

# Rapor dosyasını oluştur
report_folder = "C:/Raporlar"
if not os.path.exists(report_folder):
    os.makedirs(report_folder)

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
report_file = os.path.join(report_folder, f"hand_report_{current_time}.txt")

# Rapor dosyasına yazma fonksiyonu
def write_to_report(report_file, thumb_angle, index_angle, mid_angle, ring_angle, pinky_angle, hand_status):
    with open(report_file, "w") as f:
        f.write("Thumb Angle: {}\n".format(thumb_angle))
        f.write("Index Finger Angle: {}\n".format(index_angle))
        f.write("Middle Finger Angle: {}\n".format(mid_angle))
        f.write("Ring Finger Angle: {}\n".format(ring_angle))
        f.write("Pinky Finger Angle: {}\n".format(pinky_angle))
        f.write("Hand Status: {}\n".format(hand_status))
        f.close()

while True:
    success, img = camera.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, channel = img.shape
    results = hands.process(imgRGB)

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

            # El durumu tespiti
            hand_status = detect_hand_status(thumb_angle, index_angle, mid_angle, ring_angle, pinky_angle)
            cv2.putText(img, f"Status: {hand_status}", (25, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Eklemleri çiz
            mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)

            # Rapor dosyasına yaz
            write_to_report(report_file, thumb_angle, index_angle, mid_angle, ring_angle, pinky_angle, hand_status)

    cv2.imshow("Camera", img)

    # Kamera Çıkış Tuşu
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
