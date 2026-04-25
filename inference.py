import cv2
import re
import numpy as np
from ultralytics import YOLO
from rapidocr_onnxruntime import RapidOCR
import time

IMG_SIZE = 320

# Load models (cached logic will be in app.py or we can do it here)
# But doing it here means it loads on import. It's better to pass them as args or encapsulate in a class.

import os
class ALPRPipeline:
    def __init__(self, vehicle_weights="yolov8n.pt", plate_weights="best.pt"):
        import os
        from ultralytics import YOLO

        def _get_onnx(pt_path):
            if not pt_path.endswith(".pt"): return pt_path
            onnx_path = pt_path.replace(".pt", ".onnx")
            if not os.path.exists(onnx_path):
                print(f"Exporting {pt_path} to ONNX for faster CPU inference...")
                YOLO(pt_path).export(format="onnx", imgsz=IMG_SIZE, simplify=True, opset=12)
            return onnx_path if os.path.exists(onnx_path) else pt_path

        v_weights = _get_onnx(vehicle_weights)
        p_weights = _get_onnx(plate_weights)

        self.vehicle_model = YOLO(v_weights)
        self.plate_model = YOLO(p_weights)
        self.ocr = RapidOCR()
        
        # Warm-up
        _dummy = np.full((60, 200, 3), 200, dtype=np.uint8)
        for _ in range(3):
            try: self.ocr(_dummy, use_det=False, use_cls=False, use_rec=True)
            except Exception: pass

    VEHICLE_CLASSES = {2: "car", 3: "moto", 5: "bus", 7: "truck"}

    PLATE_FORMATS = {
        "indian_full":   r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$",
        "indian_short":  r"^[A-Z]{2,3}\d{1,4}$",
        "vietnam_car":   r"^\d{2}[A-Z]\d{3}\d{2,3}$",
        "vietnam_bike":  r"^\d{2}[A-Z]\d?\d{4,5}$",
        "european":      r"^[A-Z]{1,3}\d{1,4}[A-Z]{0,3}$",
        "us":            r"^[A-Z0-9]{5,8}$",
        "generic":       r"^[A-Z0-9]{4,12}$",
    }
    PLATE_REGEXES = {k: re.compile(v) for k, v in PLATE_FORMATS.items()}
    MIN_PLATE_LEN, MAX_PLATE_LEN = 4, 12

    def is_valid_plate(self, s):
        if not s or not (self.MIN_PLATE_LEN <= len(s) <= self.MAX_PLATE_LEN): return False
        if s == "0000" or (len(s) == 4 and s.isdigit()): return False  # Reject known hallucination and partial reads like 0084
        return any(rx.match(s) for rx in self.PLATE_REGEXES.values())

    def matched_formats(self, s):
        return [name for name, rx in self.PLATE_REGEXES.items() if rx.match(s or "")]

    def clean_text(self, s):
        return re.sub(r"[^A-Z0-9]", "", (s or "").upper())

    def deskew_plate(self, crop):
        if crop is None or crop.size == 0: return crop
        h, w = crop.shape[:2]
        if h < 30 or w < 60: return crop
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                                threshold=max(w//6, 30),
                                minLineLength=max(w//2, 40),
                                maxLineGap=5)
        if lines is None or len(lines) < 3: return crop
        angles = [np.degrees(np.arctan2(y2-y1, x2-x1))
                  for x1,y1,x2,y2 in lines[:,0]
                  if abs(np.degrees(np.arctan2(y2-y1, x2-x1))) < 20]
        if len(angles) < 3: return crop
        if np.std(angles) > 5: return crop
        angle = float(np.median(angles))
        if abs(angle) < 1.5 or abs(angle) > 12: return crop
        cos, sin = abs(np.cos(np.radians(angle))), abs(np.sin(np.radians(angle)))
        new_w = int(h*sin + w*cos); new_h = int(h*cos + w*sin)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        M[0,2] += (new_w - w) / 2
        M[1,2] += (new_h - h) / 2
        return cv2.warpAffine(crop, M, (new_w, new_h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    def prep_plate(self, crop):
        if crop is None or crop.size == 0: return None
        h, w = crop.shape[:2]
        dy, dx = int(h * 0.04), int(w * 0.02)
        tight = crop[dy:max(h-dy, dy+1), dx:max(w-dx, dx+1)]
        if tight.size == 0: tight = crop
        th = tight.shape[0]
        if th < 56:
            s = 56 / th
            tight = cv2.resize(tight, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        lab = cv2.cvtColor(tight, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def _parse_rec_only(self, result):
        if not result: return ""
        texts = []
        for r in result:
            if isinstance(r, (list, tuple)):
                if r and isinstance(r[0], (list, tuple)):
                    texts.append(str(r[0][0]) if r[0] else "")
                else:
                    texts.append(str(r[0]) if r else "")
            else:
                texts.append(str(r))
        return self.clean_text("".join(texts))

    def _parse_det_rec(self, result, force_concat=False):
        if not result: return ""
        items = []
        for r in result:
            if not (isinstance(r, (list, tuple)) and len(r) >= 2): continue
            box, txt = r[0], r[1]
            if isinstance(txt, (list, tuple)): txt = txt[0]
            if box and isinstance(box, list) and len(box) >= 1:
                ys = [p[1] for p in box]; xs = [p[0] for p in box]
                items.append((float(np.mean(ys)), float(np.mean(xs)), str(txt)))
            else:
                items.append((0.0, 0.0, str(txt)))
        if not items: return ""
        if force_concat:
            items.sort(key=lambda x: (round(x[0]/20), x[1]))
        else:
            items.sort(key=lambda x: x[1])
        return self.clean_text("".join(t for _,_,t in items))

    def _read_two_line(self, crop):
        h, w = crop.shape[:2]
        top = crop[:h//2 + 2, :]
        bot = crop[h//2 - 2:, :]
        out = ""
        for part in (top, bot):
            prep = self.prep_plate(part)
            if prep is None: continue
            part_text = ""
            try:
                r, _ = self.ocr(prep, use_det=False, use_cls=False, use_rec=True)
                part_text = self._parse_rec_only(r)
            except Exception: pass
            
            # Fallback to full detection on the part if fast-read fails
            if not part_text:
                try:
                    r, _ = self.ocr(part)
                    part_text = self._parse_det_rec(r, force_concat=True)
                except Exception: pass
                
            out += part_text
        return out

    def read_plate(self, plate_img):
        if plate_img is None or plate_img.size == 0: return ""
        h0, w0 = plate_img.shape[:2]
        py, px = int(h0*0.06), int(w0*0.06)
        plate_img = cv2.copyMakeBorder(plate_img, py, py, px, px, cv2.BORDER_REPLICATE)
        plate_img = self.deskew_plate(plate_img)

        h, w = plate_img.shape[:2]
        aspect = h / max(w, 1)
        is_two_line = aspect > 0.40

        def _accept(s):
            if not s or not (4 <= len(s) <= 11): return False
            if s == "0000" or (len(s) == 4 and s.isdigit()): return False
            return True

        if is_two_line:
            split = self._read_two_line(plate_img)
            if self.is_valid_plate(split): return split
            try:
                result, _ = self.ocr(plate_img)
                full = self._parse_det_rec(result, force_concat=True)
                if self.is_valid_plate(full): return full
                cand = full if _accept(full) else split
                return cand if _accept(cand) else ""
            except Exception:
                return split if _accept(split) else ""

        prep = self.prep_plate(plate_img)
        if prep is None: return ""
        fast = ""
        try:
            result, _ = self.ocr(prep, use_det=False, use_cls=False, use_rec=True)
            fast = self._parse_rec_only(result)
            if self.is_valid_plate(fast): return fast
        except Exception:
            fast = ""
        try:
            result, _ = self.ocr(plate_img)
            full = self._parse_det_rec(result)
            if self.is_valid_plate(full): return full
            if _accept(full) and _accept(fast):
                return full if len(full) > len(fast) else fast
            if _accept(full): return full
            if _accept(fast): return fast
            return ""
        except Exception:
            return fast if _accept(fast) else ""

    def process_image(self, img):
        t0 = time.perf_counter()
        detections = []
        v_out = self.vehicle_model(img, imgsz=IMG_SIZE,
                              classes=list(self.VEHICLE_CLASSES.keys()),
                              conf=0.05, verbose=False)[0]
        
        if len(v_out.boxes) == 0:
            # Try to find plates directly
            p_out = self.plate_model(img, imgsz=IMG_SIZE, conf=0.10, verbose=False)[0]
            if len(p_out.boxes) == 0:
                return {"processing_time_ms": int((time.perf_counter()-t0)*1000), "detections": []}
            
            plates = []
            for pb in p_out.boxes:
                px1,py1,px2,py2 = map(int, pb.xyxy[0].tolist())
                plate_img = img[py1:py2, px1:px2]
                text = self.read_plate(plate_img)
                plates.append({
                    "plate": text or "UNKNOWN", 
                    "valid": self.is_valid_plate(text), 
                    "raw": text, 
                    "bbox": [px1,py1,px2,py2],
                    "conf": float(pb.conf[0])
                })
            
            return {
                "processing_time_ms": int((time.perf_counter()-t0)*1000), 
                "detections": [{
                    "vehicle": "vehicle", 
                    "confidence": 1.0, 
                    "bbox": [0,0,img.shape[1],img.shape[0]], 
                    "plates": plates
                }]
            }

        for vb in v_out.boxes:
            vx1,vy1,vx2,vy2 = map(int, vb.xyxy[0].tolist())
            v_crop = img[vy1:vy2, vx1:vx2]
            p_out = self.plate_model(v_crop, imgsz=IMG_SIZE, conf=0.10, verbose=False)[0]
            plates = []
            for pb in p_out.boxes:
                px1,py1,px2,py2 = map(int, pb.xyxy[0].tolist())
                plate_img = v_crop[py1:py2, px1:px2]
                text = self.read_plate(plate_img)
                plates.append({
                    "plate": text or "UNKNOWN", 
                    "valid": self.is_valid_plate(text), 
                    "raw": text, 
                    "bbox": [vx1+px1,vy1+py1,vx1+px2,vy1+py2],
                    "conf": float(pb.conf[0])
                })
            
            cls_name = self.VEHICLE_CLASSES.get(int(vb.cls[0]), "vehicle")
            detections.append({
                "vehicle": cls_name,
                "bbox": [vx1,vy1,vx2,vy2],
                "confidence": float(vb.conf[0]),
                "plates": plates
            })
            
        return {
            "processing_time_ms": int((time.perf_counter()-t0)*1000),
            "detections": detections
        }
