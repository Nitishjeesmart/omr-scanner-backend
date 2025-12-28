from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import imutils
from imutils import contours

app = Flask(__name__)
CORS(app)

# --- 1. इमेज को सीधा करने वाला फंक्शन ---
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

# --- 2. यूनिवर्सल बबल हंटिंग (THE BRAIN) ---
def process_omr_image(image_bytes):
    # इमेज लोड करें
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None: return {"error": "Invalid Image"}

    # ग्रेस्केल और ब्लर
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # पेपर ढूँढना (Corners)
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
        warped = gray # अगर पेपर का बॉर्डर नहीं मिला, तो पूरी इमेज यूज़ करो
    else:
        warped = four_point_transform(gray, docCnt.reshape(4, 2))

    # --- BUBBLE HUNTING START ---
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    questionCnts = []

    # फिल्टर: सिर्फ गोले (Bubbles) रखो
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h) 
        # आकार चेक: न ज्यादा छोटा, न ज्यादा बड़ा
        if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
            questionCnts.append(c)

    if len(questionCnts) == 0:
        return {"status": "error", "message": "No bubbles found."}

    # --- UNIVERSAL SORTING ---
    # ऊपर से नीचे सॉर्ट करें
    questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]

    # गोलों को 4 कॉलम में बांटने का लॉजिक
    boundingBoxes = [cv2.boundingRect(c) for c in questionCnts]
    zipped = sorted(zip(boundingBoxes, questionCnts), key=lambda b: b[0][0]) # X-Axis Sort
    
    total_cols = 4 # आपकी शीट में 4 बड़े कॉलम हैं
    bubbles_per_col = len(zipped) // total_cols 
    
    results = {}
    options = ['A', 'B', 'C', 'D']
    q_counter = 1

    for col_i in range(total_cols):
        col_bubbles = zipped[col_i * bubbles_per_col : (col_i + 1) * bubbles_per_col]
        col_bubbles = sorted(col_bubbles, key=lambda b: b[0][1]) # Y-Axis Sort inside column
        
        # 4-4 के ग्रुप (Question) बनाओ
        for i in range(0, len(col_bubbles), 4):
            q_pack = col_bubbles[i : i + 4]
            q_pack = sorted(q_pack, key=lambda b: b[0][0]) # A,B,C,D order
            
            filled_count = 0
            filled_idx = -1
            
            for (j, (bbox, c)) in enumerate(q_pack):
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                total = cv2.countNonZero(mask)

                if total > 500: # काला रंग चेक
                    filled_count += 1
                    filled_idx = j
            
            val = "SKIP"
            if filled_count == 1: val = options[filled_idx]
            elif filled_count > 1: val = "DUAL"
            
            results[str(q_counter)] = val
            q_counter += 1

    return {"status": "success", "data": results}

@app.route('/')
def home(): return "Universal OMR Brain Active 🧠"

@app.route('/scan', methods=['POST'])
def scan():
    if 'file' not in request.files: return jsonify({"status": "error", "message": "No file"}), 400
    try:
        return jsonify(process_omr_image(request.files['file'].read()))
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run()
