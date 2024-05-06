import cv2
import mediapipe
import pyttsx3
import math




camera = cv2.VideoCapture(0)
engine = pyttsx3.init()
mpHands = mediapipe.solutions.hands
hands = mpHands.Hands()
mpDraw = mediapipe.solutions.drawing_utils

while True:

    success, img = camera.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hlms = hands.process(imgRGB)
    height, width, channel = img.shape
    results = hands.process(imgRGB)

    print(hlms.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # İşaret parmağının eklemlerinin indeksleri (MediaPipe Hands kütüphanesine göre)
            index_base = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_MCP]
            index_mid = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_PIP]
            index_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]

            # İşaret parmağının eklemlerinin koordinatlarını alın
            base_coords = (index_base.x, index_base.y)
            mid_coords = (index_mid.x, index_mid.y)
            tip_coords = (index_tip.x, index_tip.y)

            # İki eklemin koordinatları arasındaki mesafeyi ve açıyı hesapla
            distance_base_mid = math.dist(base_coords, mid_coords)
            distance_mid_tip = math.dist(mid_coords, tip_coords)
            distance_base_tip = math.dist(base_coords, tip_coords)

            # İki vektör arasındaki açıyı hesapla (Cosinus teoremi kullanılarak)
            angle = math.degrees(math.acos((distance_base_mid ** 2 + distance_mid_tip ** 2 - distance_base_tip ** 2) /
                                           (2 * distance_base_mid * distance_mid_tip)))

            # Açıyı ekrana yazdır
            cv2.putText(img, f"Angle: {int(angle)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Eklemleri çiz
            mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)

    if hlms.multi_hand_landmarks:
        for handlandmarks in hlms.multi_hand_landmarks:

            for fingerNum, landmark in enumerate(handlandmarks.landmark):
                positionX, positionY = int(landmark.x * width), int(landmark.y * height)


            mpDraw.draw_landmarks(img, handlandmarks, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Camera", img)



    if cv2.waitKey(1) & 0xFF == ord("q"):
        break