import cv2
import numpy as np
from ultralytics import YOLO

# ---------- Ayarlar ----------
CAPACITY = 5          # bölgenin kapasitesi(kişi)
CONF = 0.4            # tespit güven eşiği
CAMERA_INDEX = 0
USE_CENTER = True     # webcam testinde True; tavana/yukarıya takılı kamerada False(ayak noktası)
WINDOW = "Doluluk Prototipi"

model = YOLO("yolov8n.pt")
zone_points = []      # fareyle seçilen bölge köşeleri


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        zone_points.append((x, y))


def occupancy_color(ratio):
    if ratio < 0.5:
        return (0, 200, 0)      # yeşil
    if ratio < 0.8:
        return (0, 165, 255)    # turuncu
    return (0, 0, 255)          # kırmızı


cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Kamera açılamadı")

cv2.namedWindow(WINDOW)
cv2.setMouseCallback(WINDOW, on_mouse)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model.track(
        frame, persist=True, classes=[0], conf=CONF,
        tracker="bytetrack.yaml", verbose=False
    )
    boxes = results[0].boxes

    zone = np.array(zone_points, np.int32) if len(zone_points) >= 3 else None
    count = 0

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        track_id = int(box.id[0]) if box.id is not None else -1
        point = ((x1 + x2) // 2, (y1 + y2) // 2) if USE_CENTER else ((x1 + x2) // 2, y2)

        inside = True if zone is None else cv2.pointPolygonTest(zone, point, False) >= 0
        if inside:
            count += 1

        color = (0, 255, 0) if inside else (150, 150, 150)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, point, 4, color, -1)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # bölgeyi çiz
    if zone is not None:
        cv2.polylines(frame, [zone], True, (255, 200, 0), 2)
    elif zone_points:
        for p in zone_points:
            cv2.circle(frame, p, 5, (255, 200, 0), -1)
        if len(zone_points) > 1:
            cv2.polylines(frame, [np.array(zone_points, np.int32)], False, (255, 200, 0), 2)

    # doluluk bilgisi
    ratio = count / CAPACITY
    color = occupancy_color(ratio)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (30, 30, 30), -1)
    cv2.putText(frame, f"Kisi: {count} / {CAPACITY}   Doluluk: %{ratio * 100:.0f}",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    hint = "Bolge: fareyle en az 3 nokta tikla | r: sifirla | q: cik"
    if zone is None:
        hint = "Bolge yok: tum kare sayiliyor. " + hint
    cv2.putText(frame, hint, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow(WINDOW, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        zone_points.clear()

cap.release()
cv2.destroyAllWindows()