from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import imutils

app = Flask(__name__)
CORS(app)  # यह HTML फाइल को रिक्वेस्ट भेजने की अनुमति देता है

# --- OMR LOGIC FUNCTIONS ---
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def process_omr_image(image_bytes):
    # इमेज को बाइट्स से डिकोड करें
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None: return {"error": "Invalid Image"}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # कोनों को ढूँढना
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    docCnt = None

    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                docCnt = approx
                break
    
    if docCnt is None:
        return {"error": "Paper corners not found. Please capture a clear image with borders."}

    # सीधा करना और थ्रेशोल्ड
    warped = four_point_transform(gray, docCnt.reshape(4, 2))
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # --- 4 COLUMN LOGIC ---
    h, w = thresh.shape
    col_width = w // 4
    row_height = h // 25
    
    results = {}
    options = ['A', 'B', 'C', 'D']

    for col_idx in range(4):
        start_x = col_idx * col_width
        end_x = (col_idx + 1) * col_width
        col_img = thresh[:, start_x:end_x]

        for row_idx in range(25):
            q_num = (col_idx * 25) + row_idx + 1
            start_y = row_idx * row_height
            end_y = (row_idx + 1) * row_height
            q_img = col_img[start_y:end_y, :]

            # 4 Bubbles Check
            q_h, q_w = q_img.shape
            bubble_w = q_w // 4
            bubbles = []

            for k in range(4):
                bx_s = k * bubble_w
                bx_e = (k + 1) * bubble_w
                roi = q_img[:, bx_s:bx_e]
                total = cv2.countNonZero(roi)
                bubbles.append((total, k))

            bubbles.sort(key=lambda x: x[0], reverse=True)
            best_val, best_idx = bubbles[0]
            second_val, _ = bubbles[1]

            # Logic for Selection
            if best_val > 100: # Threshold can be adjusted
                if second_val > (best_val * 0.85):
                    results[str(q_num)] = "DUAL"
                else:
                    results[str(q_num)] = options[best_idx]
            else:
                results[str(q_num)] = "SKIP"
    
    return {"status": "success", "data": results}


# --- API ROUTES ---
@app.route('/')
def home():
    return "OMR Scanner API is Running! 🚀"

@app.route('/scan', methods=['POST'])
def scan():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400
    
    file = request.files['file']
    result = process_omr_image(file.read())
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
