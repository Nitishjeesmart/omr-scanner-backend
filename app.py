import cv2
import numpy as np
import imutils
from imutils import contours
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# --- SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OMR_Human_Eye")
app = Flask(__name__)
CORS(app)

# --- HUMAN EYE CLASS (आंख) ---
class HumanEye:
    def load_image(self, file_bytes) -> np.ndarray:
        try:
            nparr = np.frombuffer(file_bytes, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except:
            return None

    def adjust_vision(self, img):
        """
        आंख की पुतली की तरह रौशनी सेट करना (Adaptive Vision)
        """
        # 1. ग्रेस्केल में बदलो
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. CONTRAST BOOST (नीले पेन को गहरा काला बनाना)
        # यह देखेगा कि इमेज में सबसे डार्क और लाइट पिक्सेल कौन से हैं और उन्हें फैला देगा
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # 3. ADAPTIVE THRESHOLD (छाया/Shadow हटाना)
        # यह हर छोटे हिस्से (21x21 block) को अलग-अलग चेक करेगा
        # 15 = Constant (Noise हटाने के लिए)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 21, 10)

        return gray, thresh

    def four_point_transform(self, image, pts):
        # इमेज को सीधा करना (Perspective Transform)
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

# --- HUMAN BRAIN CLASS (दिमाग) ---
class HumanBrain:
    def __init__(self):
        self.eye = HumanEye()

    def think(self, img_bytes):
        # 1. आंख से देखो
        img = self.eye.load_image(img_bytes)
        if img is None: return {"error": "Image Load Failed"}

        # 2. विजन एडजस्ट करो (Shadow & Blue Pen Logic)
        gray, thresh = self.eye.adjust_vision(img)

        # 3. पेपर का बॉर्डर ढूँढो
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
            warped = self.eye.four_point_transform(gray, docCnt.reshape(4, 2))
            thresh_warped = self.eye.four_point_transform(thresh, docCnt.reshape(4, 2))
        else:
            warped = gray
            thresh_warped = thresh

        # 4. गोले ढूँढो (Universal Bubble Hunting)
        cnts = cv2.findContours(thresh_warped.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        questionCnts = []
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            # फिल्टर: यूनिवर्सल साइज (छोटा गोला भी और बड़ा भी)
            if w >= 16 and h >= 16 and ar >= 0.75 and ar <= 1.25:
                questionCnts.append(c)

        if not questionCnts:
            return {"status": "error", "message": "No bubbles found."}

        # 5. Sorting (पहले ऊपर से नीचे, फिर कॉलम)
        questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
        
        # 4 कॉलम (Aryabhatta Pattern)
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

            # 4-4 का ग्रुप (Question)
            for i in range(0, len(col_bubbles), 4):
                q_pack = col_bubbles[i:i+4]
                q_pack = sorted(q_pack, key=lambda b: b[0][0]) # A,B,C,D
                
                # --- PIXEL COMPARISON (Relative Logic) ---
                # हम "Threshold" नहीं, "Comparison" करेंगे (कौन सबसे ज्यादा भरा है)
                pixels = []
                for (bbox, c) in q_pack:
                    mask = np.zeros(thresh_warped.shape, dtype="uint8")
                    cv2.drawContours(mask, [c], -1, 255, -1)
                    mask = cv2.bitwise_and(thresh_warped, thresh_warped, mask=mask)
                    total = cv2.countNonZero(mask)
                    pixels.append(total)
                
                # विनर ढूँढो (Winner Takes All)
                max_pixels = max(pixels)
                max_index = pixels.index(max_pixels)
                
                detected = "SKIP"
                
                # Logic: क्या विनर के पास कम से कम कुछ स्याही है? (Noise Filter)
                # 300 पिक्सेल = बहुत हल्की स्याही भी चलेगी
                if max_pixels > 300: 
                    detected = options_map[max_index]
                    
                    # Dual Check: क्या दूसरा नंबर वाला भी विनर के करीब है?
                    sorted_p = sorted(pixels, reverse=True)
                    second_max = sorted_p[1]
                    
                    # अगर दूसरा गोला पहले गोले के 90% जितना भरा है, तो डुअल है
                    if second_max > (max_pixels * 0.9): 
                        detected = "DUAL"

                results[str(q_num)] = detected
                q_num += 1

        return {"status": "success", "data": results}

# --- ROUTES ---
brain = HumanBrain()

@app.route('/')
def home(): return "Human-Like OMR Vision Active 👁️🧠"

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
