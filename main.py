import time
import cv2
import mediapipe as mp

model_path = "face_landmarker.task"

face_images = [{
    "surprised": "./face/surprised.jpg",
    "happy": "./face/happy.jpg",
    "sad": "./face/sad.jpg",
    "neutral": "./face/neutral.jpg",
    "angry": "./face/angry.jpg"
}]

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None

def callback(result: mp.tasks.vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

def infer_emotion_from_blendshapes(blendshapes):
    # blendshapes: list of (name, score)
    bs = {b.category_name: b.score for b in blendshapes}
    def g(name): 
        return bs.get(name, 0.0)

    # Core signals
    smile = (g("mouthSmileLeft") + g("mouthSmileRight")) / 2
    frown = (g("mouthFrownLeft") + g("mouthFrownRight")) / 2
    jaw_open = g("jawOpen")

    brow_down = (g("browDownLeft") + g("browDownRight")) / 2
    brow_inner_up = g("browInnerUp")
    brow_outer_up = (g("browOuterUpLeft") + g("browOuterUpRight")) / 2

    eye_squint = (g("eyeSquintLeft") + g("eyeSquintRight")) / 2
    eye_wide = (g("eyeWideLeft") + g("eyeWideRight")) / 2

    # Optional extra mouth tension cues (can help)
    lip_press = g("mouthPressLeft") + g("mouthPressRight")
    lip_pucker = g("mouthPucker")
    # mouth_stretch = (g("mouthStretchLeft") + g("mouthStretchRight")) / 2
    mouth_lower_down = (g("mouthLowerDownLeft") + g("mouthLowerDownRight")) / 2.0

    mouth_upper_up = (g("mouthUpperUpLeft") + g("mouthUpperUpRight")) / 2.0
    mouth_dimple = (g("mouthDimpleLeft") + g("mouthDimpleRight")) / 2.0    

    nose_sneer = (g("noseSneerLeft") + g("noseSneerRight")) / 2.0
 
    # Score each emotion instead of hard if/else

    scores = {k: 0.0 for k in ["happy", "surprised", "angry", "sad", "neutral"]}

    # HAPPY: strong smile, some squint, low frown/browDown
    scores["happy"] += (
        2.5 * smile +
        0.9 * eye_squint +
        0.4 * mouth_dimple -
        1.4 * frown -
        0.7 * brow_down
    )

    # SURPRISED: jaw open + eyes wide + brows up, penalize smile/frown
    scores["surprised"] += (
        2.2 * jaw_open +
        1.6 * eye_wide +
        1.1 * brow_inner_up +
        0.6 * brow_outer_up -
        0.6 * smile -
        0.6 * frown
    )

    # ANGRY: brow down + squint + lip press, penalize smile, slight penalty for brows up
    scores["angry"] += (

        5.0 * brow_down +
        2.0 * eye_squint +
        2.0 * nose_sneer +
        2.0 * lip_press -
        1.2 * smile -
        0.6 * brow_inner_up
    )

    # SAD: inner brow up + frown + mouth corners down / lower lip down, penalize wide eyes
    scores["sad"] += (
        1.8 * brow_inner_up +
        1.8 * frown +
        1.2 * mouth_lower_down -
        1.1 * smile -
        0.6 * eye_wide
    )

    # Neutral: low overall “expression energy”
    expression_strength = (
        1.6 * smile + 1.6 * frown + 1.3 * jaw_open +
        1.2 * brow_down + 1.0 * brow_inner_up +
        1.1 * eye_wide + 1.0 * eye_squint +
        0.8 * lip_press + 0.6 * lip_pucker +
        0.7 * mouth_lower_down + 0.6 * mouth_upper_up
    )
    scores["neutral"] += max(0.0, 1.1 - expression_strength)

    # Pick best
    emotion = max(scores, key=scores.get)

    # Stronger confidence gating
    sorted_items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_emotion, best_score = sorted_items[0]
    second_score = sorted_items[1][1]
    margin = best_score - second_score

    # If weak or ambiguous, neutral
    if best_emotion != "neutral" and (best_score < 0.55 or margin < 0.22):
        emotion = "neutral"
    else:
        emotion = best_emotion

    return face_images[0][emotion]

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=callback,
    output_face_blendshapes=True,
    num_faces=1
)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    start_time = time.time()

    with FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ok, frame_bgr = cap.read()
            if not ok:
                continue
        
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            flipped_frame = cv2.flip(frame_bgr, 1)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=flipped_frame)

            timestamp_ms = int((time.time() - start_time) * 1000)

            # LIVE_STREAM uses detect_async
            landmarker.detect_async(mp_image, timestamp_ms)

            # draw the latest async result (may lag by a frame)
            if latest_result and latest_result.face_landmarks:
                # h, w, _ = frame_bgr.shape
                # for face in latest_result.face_landmarks:
                #     for lm in face:
                #         x, y = int(lm.x * w), int(lm.y * h)
                #         cv2.circle(frame_bgr, (x, y), 1, (0, 255, 0), -1)

                emotion = infer_emotion_from_blendshapes(latest_result.face_blendshapes[0])
                img = cv2.imread(emotion)
                cv2.putText(flipped_frame, emotion, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Emotion", img)

            cv2.imshow("Face Landmarker (LIVE_STREAM)", flipped_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

