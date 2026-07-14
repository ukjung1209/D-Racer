import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import CompressedImage
from inference_msgs.msg import Detection, DetectionArray


# YOLO26n 학습 클래스 (best_320.onnx 메타데이터의 names와 동일)
CLASS_NAMES = {0: 'green', 1: 'left', 2: 'red', 3: 'right'}

# 디버그 오버레이 박스 색 (BGR)
CLASS_COLORS = {
    'green': (0, 255, 0),
    'red': (0, 0, 255),
    'left': (255, 180, 0),
    'right': (0, 180, 255),
    'obstacle': (255, 0, 255),   # 아루코 마커(동적 장애물)
}


def get_default_model_path():
    """리포 어딘가에 있는 best_320.onnx를 찾아 기본 경로로 쓴다.

    카메라는 320x160으로 잡으므로 imgsz 320 모델이면 업스케일 없이 원본 그대로
    추론된다. 해상도를 올려 재학습하면 model_file 파라미터로 경로만 바꾸면 된다.
    """
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'best_320.onnx'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/D-Racer/best_320.onnx'


class ObjectNode(Node):
    """카메라 이미지에서 신호등/표지판을 검출해 /object/detections로 발행한다.

    Ultralytics YOLO26n(best_320.onnx)을 onnxruntime으로 추론한다. onnx는
    end2end로 export돼 NMS가 모델 안에 포함(출력 [1,300,6] = x1,y1,x2,y2,score,
    class)돼 있어서 후처리는 confidence 필터링뿐이다. 입력 해상도는 모델에서
    자동으로 읽으므로(320) 재학습해 해상도가 바뀌어도 코드 수정 없이 model_file
    파라미터만 바꾸면 된다. 결과는 원본 프레임
    픽셀 좌표로 되돌려 Detection(class_name/id/confidence/box/area_ratio)에 담고,
    publish_debug=True면 박스를 그린 오버레이를 /object/debug로 함께 발행한다.
    """

    def __init__(self):
        super().__init__('object_node')

        # --- 토픽 (기동 시 고정) ---
        self.declare_parameter('image_topic', 'camera/image/compressed')
        self.declare_parameter('detection_topic', 'object/detections')
        self.declare_parameter('debug_topic', 'object/debug')
        self.declare_parameter('model_file', get_default_model_path())
        self.declare_parameter('publish_debug', True)
        self.declare_parameter('jpeg_quality', 90)

        # --- onnxruntime 스레드 캡 (기동 시 고정) ---
        # 이 보드는 4코어인데 기본값이면 YOLO가 추론 순간 4코어를 다 먹어 camera/lane/
        # control을 굶긴다(→ 프레임 드랍/조향 끊김). 스레드를 2개로 묶으면 개별 추론은
        # 조금 느려지지만(+20~40%) 나머지 노드가 코어를 확보해 전체가 안 끊긴다.
        # 320 작은 모델이라 스레드 많아도 이득이 적어 손해가 작다. yaml로 튜닝 가능.
        self.declare_parameter('onnx_intra_threads', 2)   # 연산 내부 병렬 스레드 수
        self.declare_parameter('onnx_inter_threads', 1)   # 연산 간 병렬(단일 브랜치라 1로 충분)

        # --- 검출 튜닝 (매 프레임 다시 읽어 live 튜닝) ---
        self.declare_parameter('conf_threshold', 0.35)

        # 처리율 제한: N프레임당 1장만 추론(표지판/신호등은 천천히 변해 30fps 불필요).
        # 3이면 30fps→10Hz로 YOLO 부하 1/3. 조향은 lane_node가 담당해 무관.
        self.declare_parameter('process_every_n', 3)

        # --- YOLO 입력 밝기 정규화 (매 프레임 다시 읽어 live 튜닝) ---
        # 반사광 억제하려고 카메라 노출을 낮추면 프레임 전체가 어두워져 YOLO가
        # 객체를 놓친다. 차선(lane_node)은 같은 어두운 원본을 그대로 쓰고, 여기서만
        # YOLO에 넣기 직전 밝기/대비를 복원해 두 목적의 노출 요구를 분리한다.
        #   yolo_gamma < 1.0 : 어두운 중간톤을 끌어올림(밝아짐). 어두우면 0.5~0.7부터.
        #   yolo_clahe True  : L채널 CLAHE 대비 평활화. 반사광/조명 편차에 강건.
        # 실시간: ros2 param set /object_node yolo_gamma 0.6
        self.declare_parameter('yolo_gamma', 1.0)
        self.declare_parameter('yolo_clahe', False)
        self._clahe = None   # CLAHE 객체는 처음 켤 때 한 번만 생성
        # 감마 LUT 캐시: gamma가 바뀐 프레임에서만 256칸 표를 다시 만들고 재사용.
        self._gamma_lut = None
        self._gamma_cached = None

        # --- 아루코 마커(동적 장애물) 검출 ---
        # enable_aruco=True면 매 프레임 아루코 마커도 찾아 같은 object/detections에
        # class_name='obstacle'로 함께 발행한다(class_id=마커 ID, area_ratio=근접도).
        # 기본은 꺼둬서(False) 신호등/갈림길 런치에는 영향이 없다.
        self.declare_parameter('enable_aruco', False)
        self.declare_parameter('aruco_dict', '4X4_50')   # cv2.aruco.DICT_<이 값>

        image_topic = str(self.get_parameter('image_topic').value)
        detection_topic = str(self.get_parameter('detection_topic').value)
        debug_topic = str(self.get_parameter('debug_topic').value)
        model_file = os.path.expanduser(str(self.get_parameter('model_file').value))
        self.publish_debug = bool(self.get_parameter('publish_debug').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.enable_aruco = bool(self.get_parameter('enable_aruco').value)

        intra_threads = int(self.get_parameter('onnx_intra_threads').value)
        inter_threads = int(self.get_parameter('onnx_inter_threads').value)
        self.session, self.input_name, self.input_size = self._load_model(
            model_file, intra_threads, inter_threads)

        self._init_aruco()

        # 밀린 프레임을 큐에 쌓지 않고 항상 최신 1장만 처리(실시간성 우선).
        # BEST_EFFORT 구독은 camera_node의 RELIABLE 발행과 호환된다(구독이 더 느슨).
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self._frame_count = 0
        self.subscription = self.create_subscription(
            CompressedImage, image_topic, self.image_callback, image_qos)
        self.publisher = self.create_publisher(DetectionArray, detection_topic, 10)
        self.debug_pub = self.create_publisher(CompressedImage, debug_topic, 10)

        self.get_logger().info(
            f'object_node started: image_topic={image_topic}, '
            f'detection_topic={detection_topic}, model={model_file}, '
            f'imgsz={self.input_size}, publish_debug={self.publish_debug}, '
            f'enable_aruco={self.enable_aruco}, '
            f'onnx_threads=intra{intra_threads}/inter{inter_threads}, '
            f'process_every_n={int(self.get_parameter("process_every_n").value)}'
        )

    # ------------------------------------------------------------------ #
    #  아루코 검출기 준비 (opencv 버전별 API 차이 흡수)
    # ------------------------------------------------------------------ #
    def _init_aruco(self):
        """enable_aruco면 아루코 딕셔너리/검출기를 만든다.

        ⚠️ 이 보드의 시스템 cv2 4.5.4(apt)에 이미 cv2.aruco가 들어 있으니
        pip로 opencv를 절대 깔지 마라(pip opencv는 GStreamer가 없어 camera_node가
        죽는다 — d-racer-hardware-gotchas 참고). 4.5.4는 detectMarkers 함수형 API,
        4.7+는 ArucoDetector 클래스라 둘 다 대응한다.
        """
        self.aruco_dict = None
        self.aruco_params = None
        self.aruco_detector = None
        if not self.enable_aruco:
            return
        if not hasattr(cv2, 'aruco'):
            self.get_logger().error(
                'enable_aruco=True인데 cv2.aruco가 없다. 시스템 cv2(4.5.4)엔 있어야 정상 — '
                'pip opencv가 시스템 걸 가리고 있는지 확인하라(pip opencv는 카메라를 죽인다). '
                '아루코 검출 비활성화.')
            self.enable_aruco = False
            return

        dict_name = str(self.get_parameter('aruco_dict').value)
        dict_id = getattr(cv2.aruco, f'DICT_{dict_name}', None)
        if dict_id is None:
            self.get_logger().warning(
                f'알 수 없는 aruco_dict={dict_name} → DICT_4X4_50 사용')
            dict_id = cv2.aruco.DICT_4X4_50

        if hasattr(cv2.aruco, 'getPredefinedDictionary'):
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        else:  # 아주 옛 버전
            self.aruco_dict = cv2.aruco.Dictionary_get(dict_id)

        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters_create()

        if hasattr(cv2.aruco, 'ArucoDetector'):  # opencv 4.7+
            self.aruco_detector = cv2.aruco.ArucoDetector(
                self.aruco_dict, self.aruco_params)
        self.get_logger().info(f'aruco enabled: dict=DICT_{dict_name}')

    # ------------------------------------------------------------------ #
    #  onnx 모델 로딩: 입력 이름/해상도를 세션에서 자동으로 읽는다
    # ------------------------------------------------------------------ #
    def _load_model(self, model_file, intra_threads=2, inter_threads=1):
        if not os.path.exists(model_file):
            raise FileNotFoundError(f'onnx model not found: {model_file}')
        # 이 보드는 GPU가 없어 CPU로 돈다. 사용 가능한 provider가 있으면 알아서 잡힌다.
        providers = ort.get_available_providers()
        # 4코어를 YOLO가 독식하지 않게 스레드 수를 묶는다(0 이하면 onnxruntime 기본=전체 코어).
        opts = ort.SessionOptions()
        if intra_threads > 0:
            opts.intra_op_num_threads = intra_threads
        if inter_threads > 0:
            opts.inter_op_num_threads = inter_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            model_file, sess_options=opts, providers=providers)
        inp = session.get_inputs()[0]
        # 입력 shape [1, 3, H, W] → 정사각(H==W) 가정, 한 변을 imgsz로 쓴다
        input_size = int(inp.shape[2])
        return session, inp.name, input_size

    # ------------------------------------------------------------------ #
    #  letterbox: 종횡비 유지한 채 imgsz 정사각으로 패딩. 역변환용 scale/pad 반환
    # ------------------------------------------------------------------ #
    def _letterbox(self, frame):
        h, w = frame.shape[:2]
        size = self.input_size
        scale = min(size / w, size / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_x = (size - new_w) // 2
        pad_y = (size - new_h) // 2
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)  # YOLO 회색 패딩
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        return canvas, scale, pad_x, pad_y

    # ------------------------------------------------------------------ #
    #  YOLO 입력 밝기/대비 정규화 (차선 원본은 그대로, YOLO 입력만 밝게)
    # ------------------------------------------------------------------ #
    def _normalize_brightness(self, frame):
        gamma = float(self.get_parameter('yolo_gamma').value)
        if abs(gamma - 1.0) > 1e-3 and gamma > 0.0:
            # 감마 LUT: out = (in/255)^gamma * 255. gamma<1이면 밝아짐.
            # gamma가 바뀐 프레임에서만 256칸 표를 다시 만들고, 아니면 캐시 재사용.
            if gamma != self._gamma_cached:
                self._gamma_lut = np.array(
                    [((i / 255.0) ** gamma) * 255 for i in range(256)],
                    dtype=np.uint8)
                self._gamma_cached = gamma
            frame = cv2.LUT(frame, self._gamma_lut)
        if bool(self.get_parameter('yolo_clahe').value):
            if self._clahe is None:
                self._clahe = cv2.createCLAHE(
                    clipLimit=2.0, tileGridSize=(8, 8))
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            l_ch = self._clahe.apply(l_ch)
            frame = cv2.cvtColor(
                cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
        return frame

    def _preprocess(self, frame):
        canvas, scale, pad_x, pad_y = self._letterbox(frame)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0          # 0~1 정규화
        tensor = np.transpose(tensor, (2, 0, 1))         # HWC → CHW
        tensor = np.expand_dims(tensor, 0)               # 배치 차원 추가
        return np.ascontiguousarray(tensor), scale, pad_x, pad_y

    # ------------------------------------------------------------------ #
    #  end2end 출력 [1,300,6] 디코딩 → 원본 프레임 좌표의 Detection 리스트
    # ------------------------------------------------------------------ #
    def _decode(self, output, scale, pad_x, pad_y, frame_w, frame_h, conf_thr):
        preds = output[0]  # (300, 6): x1,y1,x2,y2,score,class (letterbox 좌표계)
        detections = []
        frame_area = float(frame_w * frame_h)

        for row in preds:
            score = float(row[4])
            if score < conf_thr:
                continue  # end2end 출력은 score 내림차순이라 패딩 행은 여기서 걸러짐
            # letterbox 좌표 → 원본 좌표 (패딩 제거 후 스케일 복원)
            x1 = (float(row[0]) - pad_x) / scale
            y1 = (float(row[1]) - pad_y) / scale
            x2 = (float(row[2]) - pad_x) / scale
            y2 = (float(row[3]) - pad_y) / scale
            x1 = int(np.clip(x1, 0, frame_w - 1))
            y1 = int(np.clip(y1, 0, frame_h - 1))
            x2 = int(np.clip(x2, 0, frame_w - 1))
            y2 = int(np.clip(y2, 0, frame_h - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            cls_id = int(row[5])
            det = Detection()
            det.class_name = CLASS_NAMES.get(cls_id, str(cls_id))
            det.class_id = cls_id
            det.confidence = score
            det.xmin, det.ymin, det.xmax, det.ymax = x1, y1, x2, y2
            det.area_ratio = float((x2 - x1) * (y2 - y1)) / frame_area
            detections.append(det)

        return detections

    # ------------------------------------------------------------------ #
    #  아루코 마커 검출 → Detection(class_name='obstacle') 리스트
    # ------------------------------------------------------------------ #
    def _detect_aruco(self, frame, frame_w, frame_h):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.aruco_detector is not None:       # opencv 4.7+
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:                                     # 함수형 API
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params)

        detections = []
        if ids is None:
            return detections
        frame_area = float(frame_w * frame_h)
        for corner, marker_id in zip(corners, ids.flatten()):
            pts = corner.reshape(-1, 2)
            x1 = int(np.clip(pts[:, 0].min(), 0, frame_w - 1))
            y1 = int(np.clip(pts[:, 1].min(), 0, frame_h - 1))
            x2 = int(np.clip(pts[:, 0].max(), 0, frame_w - 1))
            y2 = int(np.clip(pts[:, 1].max(), 0, frame_h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            det = Detection()
            det.class_name = 'obstacle'
            det.class_id = int(marker_id)
            det.confidence = 1.0              # 아루코는 검출 자체가 확정
            det.xmin, det.ymin, det.xmax, det.ymax = x1, y1, x2, y2
            # 마커의 실제 면적(회전 반영)을 area_ratio로 → 가까울수록 커짐
            det.area_ratio = float(
                cv2.contourArea(pts.astype(np.float32))) / frame_area
            detections.append(det)
        return detections

    # ------------------------------------------------------------------ #
    def image_callback(self, msg: CompressedImage):
        # 처리율 제한: N프레임당 1장만 추론. 나머지는 디코딩·YOLO 전에 즉시 return.
        every_n = max(1, int(self.get_parameter('process_every_n').value))
        self._frame_count += 1
        if self._frame_count % every_n != 0:
            return

        raw = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warning('failed to decode compressed image')
            return

        frame_h, frame_w = frame.shape[:2]
        conf_thr = float(self.get_parameter('conf_threshold').value)

        # 밝기 정규화는 YOLO 입력에만 적용(차선 원본 토픽은 그대로). gamma/CLAHE는
        # 픽셀값만 바꾸고 좌표계는 그대로라 검출 박스는 원본/보정본 어디든 맞는다.
        vis = self._normalize_brightness(frame)

        tensor, scale, pad_x, pad_y = self._preprocess(vis)
        output = self.session.run(None, {self.input_name: tensor})[0]
        detections = self._decode(
            output, scale, pad_x, pad_y, frame_w, frame_h, conf_thr)

        if self.enable_aruco:
            detections.extend(self._detect_aruco(vis, frame_w, frame_h))

        msg_out = DetectionArray()
        msg_out.header.stamp = self.get_clock().now().to_msg()
        msg_out.header.frame_id = 'camera'
        msg_out.detections = detections
        self.publisher.publish(msg_out)

        if self.publish_debug:
            # 대시보드에서 gamma/CLAHE 튜닝 효과를 바로 보도록 보정본에 박스를 그린다
            self._publish_debug(vis, detections, msg)

    # ------------------------------------------------------------------ #
    #  검출 박스를 그린 오버레이를 /object/debug로 발행
    # ------------------------------------------------------------------ #
    def _publish_debug(self, frame, detections, source_msg):
        overlay = frame.copy()
        for det in detections:
            color = CLASS_COLORS.get(det.class_name, (255, 255, 255))
            cv2.rectangle(overlay, (det.xmin, det.ymin), (det.xmax, det.ymax), color, 2)
            label = f'{det.class_name} {det.confidence:.2f}'
            cv2.putText(overlay, label, (det.xmin, max(0, det.ymin - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(overlay, f'objects={len(detections)}', (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        ok, encoded = cv2.imencode(
            '.jpg', overlay, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return
        out = CompressedImage()
        out.header.stamp = source_msg.header.stamp
        out.header.frame_id = 'object_debug'
        out.format = 'jpeg'
        out.data = encoded.tobytes()
        self.debug_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
