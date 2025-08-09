import cv2
import numpy as np
import random
import time
from cvzone.HandTrackingModule import HandDetector
import pygame

# Initialize pygame for sound effects
pygame.init()
pop_sound = pygame.mixer.Sound("pop.wav")  # Ensure you have a pop sound file

# Load and resize images
balloon_img = cv2.imread("balloon.png", cv2.IMREAD_UNCHANGED)
balloon_img = cv2.resize(balloon_img, (120, 180), interpolation=cv2.INTER_AREA)  # Increased size

buddi_baabu_img = cv2.imread("buddi_baabu_img.png")  # Image displayed at the end

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Balloon class
class Balloon:
    def __init__(self):
        self.x = random.randint(100, 1100)
        self.y = 720
        self.width, self.height = balloon_img.shape[1], balloon_img.shape[0]

    def move(self):
        self.y -= 5
        if self.y < -self.height:
            self.y = 720
            self.x = random.randint(100, 1100)

    def check_collision(self, px, py):
        return self.x < px < self.x + self.width and self.y < py < self.y + self.height

    def draw(self, img):
        h, w, _ = balloon_img.shape
        img_h, img_w, _ = img.shape
        y1, y2 = max(0, self.y), min(self.y + h, img_h)
        x1, x2 = max(0, self.x), min(self.x + w, img_w)
        overlay = balloon_img[: y2 - y1, : x2 - x1, :3]
        mask = balloon_img[: y2 - y1, : x2 - x1, 3]
        roi = img[y1:y2, x1:x2]
        if roi.shape[:2] == mask.shape:
            for c in range(3):
                roi[:, :, c] = roi[:, :, c] * (1 - mask / 255.0) + overlay[:, :, c] * (mask / 255.0)
            img[y1:y2, x1:x2] = roi

# Create balloons
balloons = [Balloon() for _ in range(5)]

# Game variables
score = 0
start_time = time.time()
game_duration = 60  # Game duration set to 1 minute

cv2.namedWindow("Balloon Pop", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Balloon Pop", cv2.WND_PROP_TOPMOST, 1)

# Main loop
while True:
    success, img = cap.read()
    if not success:
        continue
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, draw=True)
    if hands:
        lmList = hands[0]["lmList"]
        index_x, index_y = lmList[8][0], lmList[8][1]
        for balloon in balloons:
            if balloon.check_collision(index_x, index_y):
                balloons.remove(balloon)
                balloons.append(Balloon())
                score += 1
                pop_sound.play()
    for balloon in balloons:
        balloon.draw(img)
        balloon.move()
    time_left = int(game_duration - (time.time() - start_time))
    if time_left <= 0:
        break
    cv2.putText(img, f"Score: {score}", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    cv2.putText(img, f"Time: {time_left}s", (1000, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    cv2.imshow("Balloon Pop", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Show final score and buddi_baabu_img
result_img = np.zeros((720, 1280, 3), np.uint8)  # Black background
cv2.putText(result_img, f"Your Score: {score}", (400, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
if buddi_baabu_img is not None:
    buddi_baabu_img = cv2.resize(buddi_baabu_img, (300, 300))
    result_img[200:500, 500:800] = buddi_baabu_img
cv2.imshow("Game Over", result_img)
cv2.waitKey(5000)
cv2.destroyAllWindows()



