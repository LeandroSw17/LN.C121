import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

finger_tips = [8, 12, 16, 20]
thumb_tip = 4

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(img)

    finger_fold_status = []

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            for tip_id in finger_tips:
                tip = lm_list[tip_id]
                x = int(tip.x * w)
                y = int(tip.y * h)

                if tip_id == thumb_tip:
                    reference_tip = lm_list[3] 
                    reference_x = int(reference_tip.x * w)

                    if x < reference_x:
                        finger_fold_status.append(True)
                        cv2.circle(img, (x, y), 10, (0, 255, 0), -1)
                    else:
                        finger_fold_status.append(False)
                        cv2.circle(img, (x, y), 10, (0, 0, 255), -1)
                else:
                    reference_tip = lm_list[tip_id - 2] 
                    reference_x = int(reference_tip.x * w)
                    if x < reference_x:
                        finger_fold_status.append(True)
                        cv2.circle(img, (x, y), 10, (0, 255, 0), -1)
                    else:
                        finger_fold_status.append(False)
                        cv2.circle(img, (x, y), 10, (0, 0, 255), -1)

        if all(finger_fold_status) and lm_list[thumb_tip].y > lm_list[thumb_tip - 1].y:
            cv2.putText(img, "NAO CURTI", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif all(finger_fold_status) and lm_list[thumb_tip].y < lm_list[thumb_tip - 1].y:
            cv2.putText(img, "CURTI", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        mp_draw.draw_landmarks(img, hand_landmark,
                               mp_hands.HAND_CONNECTIONS,
                               mp_draw.DrawingSpec((0, 0, 255), 2, 2),
                               mp_draw.DrawingSpec((0, 255, 0), 4, 2))

    cv2.imshow("detector de maos", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
