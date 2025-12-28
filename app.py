import cv2
import numpy as np
import imutils
from imutils import contours
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# --- SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OMR_Strict_Proctor")
app = Flask(__name__)
CORS(app)

# --- HUMAN EYE (विजन) ---
class HumanEye:
    def load_image(self, file_bytes) -> np.ndarray:
        try:
            nparr = np.frombuffer(file_bytes, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except:
            return None

    def adjust_vision(self, img):
        # 1. ग्रेस्केल
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. CONTRAST BOOST (नीले पेन को गहरा काला बनाना)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # 3. ADAPTIVE THRESHOLD (छाया हटाना - Strict Mode)
        # Block Size 21, Constant 10 (Noise हटाने के लिए सख्त)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 21, 10)
        return gray, thresh

    def four_point_transform(self, image, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
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

# --- HUMAN BRAIN (सख्त दिमाग) ---
class HumanBrain:
    def __init__(self):
        self.eye = HumanEye()

    def think(self, img_bytes):
        img = self.eye.load_image(img_bytes)
        if img is None: return {"error": "Image Load Failed"}

        gray, thresh = self.eye.adjust_vision(img)

        # पेपर ढूँढो
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        
        if docCnt is not None:
            thresh_warped = self.eye.four_point_transform(thresh, docCnt.reshape(4, 2))
        else:
            thresh_warped = thresh

        # गोले ढूँढो (Universal)
        cnts = cv2.findContours(thresh_warped.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        questionCnts = []
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            # फिल्टर: साइज चेक (Standard OMR Bubbles)
            if w >= 16 and h >= 16 and ar >= 0.75 and ar <= 1.25:
                questionCnts.append(c)

        if not questionCnts: return {"status": "error", "message": "No bubbles found."}

        # Sorting (Top-to-Bottom)
        questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
        
        # Column Identification
        bbs = [cv2.boundingRect(c) for c in questionCnts]
        zipped = sorted(zip(bbs, questionCnts), key=lambda b: b[0][0]) 
        total_cols = 4
        bubbles_per_col = len(zipped) // total_cols
        
        results = {}
        options_map = ['A', 'B', 'C', 'D']
        q_num = 1

        for col_i in range(total_cols):
            col_bubbles = zipped[col_i * bubbles_per_col : (col_i + 1) * bubbles_per_col]
            col_bubbles = sorted(col_bubbles, key=lambda b: b[0][1])

            for i in range(0, len(col_bubbles), 4):
                q_pack = col_bubbles[i:i+4]
                q_pack = sorted(q_pack, key=lambda b: b[0][0]) 
                
                pixels = []
                for (bbox, c) in q_pack:
                    mask = np.zeros(thresh_warped.shape, dtype="uint8")
                    cv2.drawContours(mask, [c], -1, 255, -1)
                    mask = cv2.bitwise_and(thresh_warped, thresh_warped, mask=mask)
                    total = cv2.countNonZero(mask)
                    pixels.append(total)
                
                # --- S.P.C. LOGIC (Strict Proctor Code) ---
                
                # विनर ढूँढो
                max_pixels = max(pixels)
                max_index = pixels.index(max_pixels)
                
                detected = "SKIP"
                
                # RULE 1: LAXMAN REKHA (Minimum Ink Check)
                # अगर विनर के पास 450 पिक्सेल से कम स्याही है, तो उसे खाली मानो।
                # (यह खाली गोलों को DUAL बनने से रोकेगा)
                if max_pixels < 450:
                    detected = "SKIP" 
                
                else:
                    # RULE 2: Winner Valid
                    detected = options_map[max_index]
                    
                    # RULE 3: DUAL CHECK (Strict)
                    # दूसरा गोला तभी देखो जब वह भी 'लक्ष्मण रेखा' (450) पार करे
                    sorted_p = sorted(pixels, reverse=True)
                    second_max = sorted_p[1]
                    
                    # अगर दूसरा गोला भी 450+ है और पहले के करीब है
                    if second_max > 450 and second_max > (max_pixels * 0.8):
                        detected = "DUAL"

                results[str(q_num)] = detected
                q_num += 1

        return {"status": "success", "data": results}

brain = HumanBrain()

@app.route('/')
def home(): return "Strict Proctor OMR (S.P.C) Active 🛡️"

@app.route('/scan', methods=['POST'])
def scan():
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    try:
        res = brain.think(request.files['file'].read())
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()
