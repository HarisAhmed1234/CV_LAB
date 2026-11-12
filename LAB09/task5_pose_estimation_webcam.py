import math
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def angle(a, b, c):
    """Angle at joint b (degrees) formed by points a-b-c."""
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) -
                       math.atan2(a[1] - b[1], a[0] - b[0]))
    ang = abs(ang)
    return 360 - ang if ang > 180 else ang


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    h, w = frame.shape[:2]
    res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if res.pose_landmarks:
        mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lm = res.pose_landmarks.landmark
        P = mp_pose.PoseLandmark

        def pt(l):
            return (lm[l].x * w, lm[l].y * h)

        # Elbow angles (optional task item)
        for side, (s, e, wr) in {"L": (P.LEFT_SHOULDER, P.LEFT_ELBOW, P.LEFT_WRIST),
                                 "R": (P.RIGHT_SHOULDER, P.RIGHT_ELBOW, P.RIGHT_WRIST)}.items():
            a = angle(pt(s), pt(e), pt(wr))
            ex, ey = map(int, pt(e))
            cv2.putText(frame, f"{side}:{a:.0f}", (ex + 10, ey),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Real-time coordinates update (pixel coordinates of a few joints)
        nx, ny = map(int, pt(P.NOSE))
        lx, ly = map(int, pt(P.LEFT_WRIST))
        rx, ry = map(int, pt(P.RIGHT_WRIST))
        print(f"nose=({nx},{ny}) L_wrist=({lx},{ly}) R_wrist=({rx},{ry})")

    cv2.imshow("Pose Estimation", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
