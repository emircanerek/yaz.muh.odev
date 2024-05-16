import cv2
import mediapipe as mp
import pyttsx3
import math

# Kamera ayarları
camera = cv2.VideoCapture(0)
camera.set(3, 1280)  # Genişlik
camera.set(4, 720)  # Yükseklik
engine = pyttsx3.init()

# MediaPipe
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Açıları hesaplama
def calculate_angle(coords1, coords2, coords3):
    distance1 = math.dist(coords1, coords2)
    distance2 = math.dist(coords2, coords3)
    distance3 = math.dist(coords1, coords3)

    if distance1 * distance2 == 0:
        return 0

    angle = math.degrees(math.acos((distance1 ** 2 + distance2 ** 2 - distance3 ** 2) / (2 * distance1 * distance2)))
    return angle


def get_landmark_coords(landmark, img_width, img_height):
    return (landmark.x * img_width, landmark.y * img_height)


def calculate_thumb_angle(hand_landmarks, img_width, img_height):
    thumb_base = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.THUMB_CMC], img_width, img_height)
    thumb_mid = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.THUMB_IP], img_width, img_height)
    thumb_tip = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.THUMB_TIP], img_width, img_height)
    return calculate_angle(thumb_base, thumb_mid, thumb_tip)


def calculate_index_finger_angle(hand_landmarks, img_width, img_height):
    index_base = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_MCP], img_width,
                                     img_height)
    index_mid = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_PIP], img_width,
                                    img_height)
    index_tip = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP], img_width,
                                    img_height)
    return calculate_angle(index_base, index_mid, index_tip)


def calculate_middle_finger_angle(hand_landmarks, img_width, img_height):
    mid_base = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_MCP], img_width,
                                   img_height)
    mid_mid = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_DIP], img_width,
                                  img_height)
    mid_tip = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP], img_width,
                                  img_height)
    return calculate_angle(mid_base, mid_mid, mid_tip)


def calculate_ring_finger_angle(hand_landmarks, img_width, img_height):
    ring_base = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.RING_FINGER_PIP], img_width,
                                    img_height)
    ring_mid = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.RING_FINGER_DIP], img_width, img_height)
    ring_tip = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.RING_FINGER_TIP], img_width, img_height)
    return calculate_angle(ring_base, ring_mid, ring_tip)


def calculate_pinky_finger_angle(hand_landmarks, img_width, img_height):
    pinky_base = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.PINKY_PIP], img_width, img_height)
    pinky_mid = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.PINKY_DIP], img_width, img_height)
    pinky_tip = get_landmark_coords(hand_landmarks.landmark[mpHands.HandLandmark.PINKY_TIP], img_width, img_height)
    return calculate_angle(pinky_base, pinky_mid, pinky_tip)


while True:
    success, img = camera.read()
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

            # Eklemleri çiz
            mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Camera", img)

    # Kamera Çıkış Tuşu
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
