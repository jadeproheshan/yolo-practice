import torch
import cv2
import numpy as np
from pathlib import Path
from models.experimental import attempt_load
from utils.general import non_max_suppression, check_img_size
from utils.torch_utils import select_device
from utils.dataloaders import LoadImages

class Detector:

    def __init__(self, weights='yolov5s.pt', img_size=640, conf_thres=0.25, iou_thres=0.45, device=''):
        self.weights = weights
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = select_device(device if device else ('0' if torch.cuda.is_available() else 'cpu'))
        self.model = self._load_model()
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def _load_model(self):
        try:
            model = attempt_load(self.weights)
            model.to(self.device).eval()
            model.half()  # Use FP16 for faster inference if supported
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _preprocess(self, img):
        img0 = img.copy()
        h0, w0 = img.shape[:2]  
        scale = min(self.img_size / h0, self.img_size / w0)
        new_h, new_w = int(h0 * scale), int(w0 * scale)
        img_resized = cv2.resize(img, (new_w, new_h))

        img = img_resized[:, :, ::-1].transpose(2, 0, 1)  
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() / 255.0  # Normalize
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img0, img, scale, new_w, new_h

    def _postprocess(self, preds, original_shape, scale, new_w, new_h):
        results = []
        h0, w0 = original_shape[:2]
        for det in preds:
            if det is not None and len(det):
                det[:, :4] /= scale
                det[:, :4] = det[:, :4].clamp(0, max(w0, h0))
                for *xyxy, conf, cls in det:
                    label = self.names[int(cls)]
                    bbox = [int(coord) for coord in xyxy]
                    results.append({"bbox": bbox, "label": label, "confidence": float(conf)})
        return results

    def draw_boxes(self, img, detections):
        for detection in detections:
            bbox = detection['bbox']
            label = detection['label']
            confidence = detection['confidence']
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return img

    def detect(self, img):
        img0, img, scale, new_w, new_h = self._preprocess(img)
        with torch.no_grad():
            preds = self.model(img, augment=False)[0]
            preds = non_max_suppression(preds, self.conf_thres, self.iou_thres)
        detections = self._postprocess(preds, img0.shape, scale, new_w, new_h)
        return img0, detections

    def detect_from_source(self, source, save_path=None, is_camera=False):
        if is_camera:
            cap = cv2.VideoCapture(source)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                img0, detections = self.detect(frame)
                img_with_boxes = self.draw_boxes(img0, detections)
                cv2.imshow("YOLOv5 Detection", img_with_boxes)
                if save_path:
                    cv2.imwrite(str(save_path / f"frame_{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}.jpg"), img_with_boxes)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
        else:
            dataset = LoadImages(source, img_size=self.img_size)
            for path, img, im0s, vid_cap, s in dataset:
                img0, detections = self.detect(im0s)
                img_with_boxes = self.draw_boxes(img0, detections)
                cv2.imshow("YOLOv5 Detection", img_with_boxes)
                if save_path:
                    save_file = save_path / Path(path).name
                    cv2.imwrite(str(save_file), img_with_boxes)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Choose input source:")
    print("1. Camera")
    print("2. Video/Image File")
    choice = input("Enter your choice (1/2): ")

    detector = Detector(weights='yolov5s.pt', img_size=640, conf_thres=0.25, iou_thres=0.45)
    save_dir = Path("output_images")
    save_dir.mkdir(parents=True, exist_ok=True)

    if choice == '1':
        print("Using camera for detection...")
        detector.detect_from_source(0, save_path=save_dir, is_camera=True)
    elif choice == '2':
        source = input("Enter the path to video/image file: ")
        print(f"Using file {source} for detection...")
        detector.detect_from_source(source, save_path=save_dir)
    else:
        print("Invalid choice. Exiting.")
