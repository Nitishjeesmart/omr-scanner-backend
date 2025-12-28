import cv2
import numpy as np
import imutils
from imutils import contours
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# --- 1. SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OMR_Brain")
app = Flask(__name__)
CORS(app)

# --- 2. DATA STRUCTURES (PDF Logic) ---
@dataclass
class BubbleDetection:
    """Single bubble result"""
    question_num: int
    option: str
    fill_percentage: float
    status: str  # 'FULL', 'PARTIAL', 'IGNORE', 'NO_RESPONSE'

@dataclass
class QuestionResult:
    """Complete question result"""
    question_num: int
    detected_options: List[str]
    status: str  # 'ANSWERED', 'MULTIPLE', 'NO_RESPONSE', 'SKIP'
    fill_details: List[BubbleDetection]

# --- 3. CLASS: HUMAN EYE (आंख) ---
class HumanEye:
    """आंख: जो इमेज को देखती है और गोले ढूंढती है"""
    
    def load_image(self, file_bytes) -> np.ndarray:
        try:
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            logger.error(f"Image Load Error: {e}")
            return None

    def preprocess_and_find_bubbles(self, img: np.ndarray):
        # 1. ग्रेस्केल और ब्लर (धुंधलापन हटाओ)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)

        # 2. पेपर का बॉर्डर ढूंढो (Perspective Transform)
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
        
        # पेपर को सीधा करो
        if docCnt is not None:
            warped = self.four_point_transform(gray, docCnt.reshape(4, 2))
        else:
            warped = gray # अगर बॉर्डर न मिले तो पूरी इमेज लो

        # 3. अब गोले (Bubbles) ढूंढो
        thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        bubbles = []
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            # फिल्टर: सिर्फ सही साइज के गोले ( यूनिवर्सल साइज )
            if w >= 20 and h >= 20 and ar >= 0.85 and ar <= 1.15:
                bubbles.append((c, (x, y, w, h)))
        
        return thresh, bubbles

    def four_point_transform(self, image, pts):
        # इमेज सीधा करने का गणित
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

# --- 4. CLASS: HUMAN BRAIN (दिमाग) ---
class HumanBrain:
    """दिमाग: जो सोचता है और फैसला लेता है"""
    
    def __init__(self):
        self.eye = HumanEye()

    def think_and_decide(self, img_bytes):
        # आंख से कहो कि इमेज देखे
        img = self.eye.load_image(img_bytes)
        if img is None: return {"error": "Eye cannot see image"}

        thresh, raw_bubbles = self.eye.preprocess_and_find_bubbles(img)
        
        if not raw_bubbles:
            return {"status": "error", "message": "No bubbles found"}

        # --- UNIVERSAL SORTING LOGIC ---
        # 1. सारे गोलों को उनके बॉक्स (Bounding Box) के साथ निकालो
        # raw_bubbles is list of (contour, (x,y,w,h))
        
        # Contours को अलग करो सॉर्टिंग के लिए
        contours_list = [b[0] for b in raw_bubbles]
        
        # ऊपर से नीचे सॉर्ट करो
        (contours_list, _) = contours.sort_contours(contours_list, method="top-to-bottom")

        # 2. कॉलम पहचानो (Column Logic)
        # हम x-axis के हिसाब से सॉर्ट करेंगे
        bbs = [cv2.boundingRect(c) for c in contours_list]
        zipped = sorted(zip(bbs, contours_list), key=lambda b: b[0][0]) # Sort by X
        
        # मान लो 4 कॉलम हैं (Aryabhatta Sheet)
        total_cols = 4
        bubbles_per_col = len(zipped) // total_cols
        
        questions_dict = {}
        options_map = ['A', 'B', 'C', 'D']
        q_counter = 1

        for col_i in range(total_cols):
            # कॉलम के गोले निकालो
            col_data = zipped[col_i * bubbles_per_col : (col_i + 1) * bubbles_per_col]
            # कॉलम के अंदर Y (ऊपर से नीचे) सॉर्ट करो
            col_data = sorted(col_data, key=lambda b: b[0][1])
            
            # 4-4 के ग्रुप (Questions) बनाओ
            for i in range(0, len(col_data), 4):
                q_pack = col_data[i : i + 4]
                # A, B, C, D के लिए बाएं से दाएं सॉर्ट करो
                q_pack = sorted(q_pack, key=lambda b: b[0][0])
                
                bubble_details = []
                filled_count = 0
                detected_opts = []

                for idx, (bbox, c) in enumerate(q_pack):
                    if idx >= 4: break # सेफ्टी
                    
                    # पिक्सेल गिनो (Intensity Check)
                    mask = np.zeros(thresh.shape, dtype="uint8")
                    cv2.drawContours(mask, [c], -1, 255, -1)
                    mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                    total_pixels = cv2.countNonZero(mask)
                    
                    # PDF Logic: Calculate Percentage (Approx)
                    area = bbox[2] * bbox[3] # w * h
                    fill_pct = (total_pixels / area) * 100 if area > 0 else 0
                    
                    # Decision (Threshold)
                    status = "IGNORE"
                    if total_pixels > 500: # काला ज्यादा है
                        status = "FULL"
                        filled_count += 1
                        detected_opts.append(options_map[idx])
                    
                    bubble_details.append(BubbleDetection(
                        question_num=q_counter,
                        option=options_map[idx],
                        fill_percentage=fill_pct,
                        status=status
                    ))

                # फाइनल फैसला (Result Status)
                if filled_count == 0:
                    r_status = "NO_RESPONSE"
                elif filled_count == 1:
                    r_status = "ANSWERED"
                else:
                    r_status = "MULTIPLE"

                questions_dict[q_counter] = QuestionResult(
                    question_num=q_counter,
                    detected_options=detected_opts,
                    status=r_status,
                    fill_details=bubble_details
                )
                q_counter += 1
        
        return questions_dict

# --- 5. CLASS: HUMAN MOUTH (मुंह) ---
class HumanMouth:
    """मुंह: जो यूजर को रिजल्ट बताता है"""
    
    def speak(self, results: Dict[int, QuestionResult]):
        # [span_4](start_span)PDF जैसा समरी फॉर्मेट[span_4](end_span)
        total = len(results)
        answered = sum(1 for r in results.values() if r.status == 'ANSWERED')
        multiple = sum(1 for r in results.values() if r.status == 'MULTIPLE')
        no_res = sum(1 for r in results.values() if r.status == 'NO_RESPONSE')

        # Simple Output Format for your App
        simple_data = {}
        for q, res in results.items():
            if res.status == 'ANSWERED':
                simple_data[str(q)] = res.detected_options[0]
            elif res.status == 'MULTIPLE':
                simple_data[str(q)] = "DUAL"
            else:
                simple_data[str(q)] = "SKIP"

        output = {
            "status": "success",
            "summary": {
                "total": total,
                "answered": answered,
                "multiple": multiple,
                "blank": no_res,
                "message": f"Scan Complete. Found {total} questions."
            },
            "data": simple_data  # यह पुराने App के लिए जरूरी है
        }
        return output

# --- 6. API ENDPOINTS ---
brain = HumanBrain()
mouth = HumanMouth()

@app.route('/')
def home():
    return "Universal OMR Scanner (Brain & Eye System) v2.0 is Active 🧠"

@app.route('/scan', methods=['POST'])
def scan():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    try:
        # 1. Brain thinks
        results = brain.think_and_decide(request.files['file'].read())
        
        if "error" in results:
            return jsonify(results), 400
            
        # 2. Mouth speaks
        output = mouth.speak(results)
        return jsonify(output)
        
    except Exception as e:
        logger.error(f"System Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()
