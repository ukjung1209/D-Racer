import math
import os
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage
from control_msgs.msg import Control
from inference_msgs.msg import LaneState


def get_default_vehicle_config_path():
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'src' / 'config' / 'vehicle_config.yaml'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/D-Racer/src/config/vehicle_config.yaml'


class LaneNode(Node):
    """카메라 이미지에서 차선을 검출해 /lane/state로 발행한다.

    트랙은 회색 매트 + 흰색 사이드 라인이라 밝기 이진화 마스크를 쓴다
    (lane_color='white', binary_threshold 위쪽만 남김. 오렌지 트랙이면 'orange' HSV로 전환).
    ROI를 자른 뒤 버드아이뷰(BEV)로 원근을 펴고(bev_enable), 가로 밴드마다
    차선 픽셀을 좌/우 라인으로 나눠 차선 중심을 잡는다. 결과는 offset(좌우 오차),
    angle(기울기), confidence로 발행하고, 디버그 오버레이(/lane/debug/compressed)는
    [원본+사다리꼴 | 펼친 BEV+검출점]을 나란히 보여줘 사다리꼴 보정을 돕는다.
    publish_control=True면 테스트용 PD 조향을 직접 /control로 보낸다.
    """

    def __init__(self):
        super().__init__('lane_node')

        # --- 토픽/발행 관련 (기동 시 고정) ---
        self.declare_parameter('image_topic', 'camera/image/compressed')
        self.declare_parameter('lane_topic', 'lane/state')
        self.declare_parameter('debug_raw_topic', 'lane/debug/raw')   # 원본+사다리꼴
        self.declare_parameter('debug_bev_topic', 'lane/debug/bev')   # 펼친 BEV+검출점
        self.declare_parameter('publish_debug', True)
        self.declare_parameter('jpeg_quality', 90)
        self.declare_parameter('vehicle_config_file', get_default_vehicle_config_path())

        # --- 검출 튜닝 파라미터 (매 프레임 다시 읽어 live 튜닝 가능) ---
        self.declare_parameter('lane_color', 'white')           # orange | white | dark
        self.declare_parameter('hsv_lower', [8.0, 90.0, 90.0])
        self.declare_parameter('hsv_upper', [26.0, 255.0, 255.0])
        self.declare_parameter('binary_threshold', 160)         # white/dark 모드용
        self.declare_parameter('roi_top_px', 45)                # -1이면 vehicle_config ROI_TOP
        self.declare_parameter('num_bands', 10)
        self.declare_parameter('min_band_pixels', 12)
        self.declare_parameter('line_split_gap_px', 40)         # 좌/우 라인 분리 gap
        self.declare_parameter('lane_half_width_px', 90)        # 한쪽만 보일 때 반대편 추정

        # --- 버드아이뷰(BEV) 원근변환: ROI 폭/높이 비율(0~1)로 사다리꼴 4점 지정 ---
        # 사다리꼴 양 옆변이 두 차선을 그대로 덮도록 맞춰야 BEV가 편다.
        self.declare_parameter('bev_enable', True)
        self.declare_parameter('bev_top_left', 0.28)      # 윗변 좌 = 위쪽 차선 위치
        self.declare_parameter('bev_top_right', 0.72)     # 윗변 우 = 위쪽 차선 위치
        self.declare_parameter('bev_top_y', 0.32)         # 윗변 y (ROI 높이 비율) — 배경 제외
        self.declare_parameter('bev_bottom_left', 0.0)    # 아랫변 좌 = 아래쪽 차선 위치
        self.declare_parameter('bev_bottom_right', 1.0)   # 아랫변 우 = 아래쪽 차선 위치

        # --- 테스트 조향(PD) 파라미터 ---
        self.declare_parameter('publish_control', False)
        self.declare_parameter('control_topic', '/control')
        self.declare_parameter('steer_kp', 0.8)
        self.declare_parameter('steer_kd', 0.3)
        self.declare_parameter('steering_sign', 1.0)            # 방향 반대면 -1
        self.declare_parameter('steer_trim', float('nan'))      # NaN이면 vehicle_config STEER_TRIM
        self.declare_parameter('test_throttle', 0.15)

        image_topic = str(self.get_parameter('image_topic').value)
        lane_topic = str(self.get_parameter('lane_topic').value)
        debug_raw_topic = str(self.get_parameter('debug_raw_topic').value)
        debug_bev_topic = str(self.get_parameter('debug_bev_topic').value)
        control_topic = str(self.get_parameter('control_topic').value)
        self.publish_debug = bool(self.get_parameter('publish_debug').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)

        self.vehicle_config = self._load_vehicle_config(
            os.path.expanduser(str(self.get_parameter('vehicle_config_file').value))
        )

        self.subscription = self.create_subscription(
            CompressedImage, image_topic, self.image_callback, 10)
        self.publisher = self.create_publisher(LaneState, lane_topic, 10)
        self.debug_raw_pub = self.create_publisher(CompressedImage, debug_raw_topic, 10)
        self.debug_bev_pub = self.create_publisher(CompressedImage, debug_bev_topic, 10)
        self.control_pub = self.create_publisher(Control, control_topic, 10)

        self.prev_offset = 0.0
        self.prev_center = None   # 직전 프레임 차선중심 (좌/우 오분류 방지용 시드)

        self.get_logger().info(
            f'lane_node started: image_topic={image_topic}, lane_topic={lane_topic}, '
            f'debug_raw={debug_raw_topic}, debug_bev={debug_bev_topic}, '
            f'publish_debug={self.publish_debug}'
        )

    # ------------------------------------------------------------------ #
    #  vehicle_config.yaml fallback 로딩
    # ------------------------------------------------------------------ #
    def _load_vehicle_config(self, path):
        if not os.path.exists(path):
            self.get_logger().warning(f'vehicle config not found: {path}')
            return {}
        try:
            import yaml
            with open(path, 'r', encoding='utf-8') as stream:
                return yaml.safe_load(stream) or {}
        except Exception as exc:
            self.get_logger().warning(f'failed to read vehicle config {path}: {exc}')
            return {}

    # ------------------------------------------------------------------ #
    #  오렌지/흰/어두운 라인 마스크 만들기
    # ------------------------------------------------------------------ #
    def _build_mask(self, roi_bgr, lane_color):
        if lane_color == 'orange':
            hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            lower = np.array(
                [float(v) for v in self.get_parameter('hsv_lower').value], dtype=np.float32)
            upper = np.array(
                [float(v) for v in self.get_parameter('hsv_upper').value], dtype=np.float32)
            mask = cv2.inRange(hsv, lower, upper)
        else:
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            thr = int(self.get_parameter('binary_threshold').value)
            if lane_color == 'dark':
                _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
            else:  # white
                _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)

        # 잡음 제거: open (침식 후 팽창)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    # ------------------------------------------------------------------ #
    #  버드아이뷰 원근변환: 사다리꼴(src) → 직사각형(dst) 로 펴기
    # ------------------------------------------------------------------ #
    def _bev_src_points(self, w, h):
        tl = float(self.get_parameter('bev_top_left').value)
        tr = float(self.get_parameter('bev_top_right').value)
        ty = float(self.get_parameter('bev_top_y').value)
        bl = float(self.get_parameter('bev_bottom_left').value)
        br = float(self.get_parameter('bev_bottom_right').value)
        return np.float32([
            [tl * w, ty * h],       # 윗변 좌 (먼 곳)
            [tr * w, ty * h],       # 윗변 우
            [br * w, h - 1],        # 아랫변 우 (가까운 곳)
            [bl * w, h - 1],        # 아랫변 좌
        ])

    def _warp_bev(self, img):
        h, w = img.shape[:2]
        src = self._bev_src_points(w, h)
        dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        try:
            matrix = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(
                img, matrix, (w, h), flags=cv2.INTER_LINEAR)
        except cv2.error as exc:
            self.get_logger().warning(f'BEV warp failed, using raw ROI: {exc}')
            return img, src
        return warped, src

    # ------------------------------------------------------------------ #
    #  가로 밴드별로 좌/우 라인 분리 → 차선 중심 추정 (아래→위 전파)
    # ------------------------------------------------------------------ #
    def _analyze_bands(self, mask, width, num_bands, min_pixels, split_gap,
                       half_width, seed_center):
        h = mask.shape[0]
        band_h = max(1, h // num_bands)
        bands = []
        # 맨 아래 밴드의 좌/우 판정 기준. 직전 프레임 중심으로 시드하면
        # 커브에서 라인이 화면중앙을 넘어와도 같은 쪽으로 계속 인식한다.
        running_center = seed_center

        # 아래(차에 가까운 쪽)부터 위로 올라가며 처리
        for i in range(num_bands):
            y1 = h - i * band_h
            y0 = max(0, y1 - band_h)
            if y1 <= 0:
                break
            band = mask[y0:y1, :]
            xs = np.nonzero(band.any(axis=0))[0]     # 마스크 픽셀이 있는 열 인덱스
            if xs.size < min_pixels:
                bands.append({'valid': False, 'y': (y0 + y1) // 2})
                continue

            xs_sorted = np.sort(xs)
            diffs = np.diff(xs_sorted)
            left_x = right_x = None

            if diffs.size > 0 and diffs.max() > split_gap:
                # 가장 큰 틈에서 좌/우 라인으로 분리
                split = int(np.argmax(diffs))
                left_x = float(xs_sorted[:split + 1].mean())
                right_x = float(xs_sorted[split + 1:].mean())
                center = (left_x + right_x) / 2.0
            else:
                # 한쪽 라인만 보임 → 이전 중심 기준으로 좌/우 판정 후 반대편 추정
                boundary = float(xs_sorted.mean())
                if boundary < running_center:
                    left_x = boundary
                    center = boundary + half_width
                else:
                    right_x = boundary
                    center = boundary - half_width

            center = float(np.clip(center, 0.0, width))
            running_center = center
            bands.append({
                'valid': True, 'y': (y0 + y1) // 2,
                'left': left_x, 'right': right_x, 'center': center,
            })

        return bands

    # ------------------------------------------------------------------ #
    def image_callback(self, msg: CompressedImage):
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warning('failed to decode compressed image')
            return

        height, width = frame.shape[:2]

        # --- 튜닝 파라미터 매 프레임 새로 읽기 (live 튜닝) ---
        lane_color = str(self.get_parameter('lane_color').value)
        num_bands = max(1, int(self.get_parameter('num_bands').value))
        min_pixels = int(self.get_parameter('min_band_pixels').value)
        split_gap = int(self.get_parameter('line_split_gap_px').value)
        half_width = int(self.get_parameter('lane_half_width_px').value)

        roi_top = int(self.get_parameter('roi_top_px').value)
        if roi_top < 0:
            roi_top = int(self.vehicle_config.get('ROI_TOP', 0))
        roi_top = int(np.clip(roi_top, 0, height - 1))

        roi = frame[roi_top:, :]

        # BEV로 펼친 뒤 분석 (bev_enable=False면 원본 ROI 그대로)
        bev_on = bool(self.get_parameter('bev_enable').value)
        if bev_on:
            analysis_img, bev_src = self._warp_bev(roi)
        else:
            analysis_img, bev_src = roi, None

        mask = self._build_mask(analysis_img, lane_color)
        seed_center = self.prev_center if self.prev_center is not None else width / 2.0
        bands = self._analyze_bands(
            mask, width, num_bands, min_pixels, split_gap, half_width, seed_center)

        valid = [b for b in bands if b['valid']]

        # 다음 프레임 시드: 가장 아래(가까운) 밴드 중심. 차선 잃으면 중앙으로 리셋.
        self.prev_center = valid[0]['center'] if valid else None

        state = LaneState()
        state.header.stamp = self.get_clock().now().to_msg()
        state.header.frame_id = 'camera'

        if len(valid) >= 2:
            cx = width / 2.0
            near = valid[0]                 # 가장 아래(가까운) 밴드
            far = valid[-1]                 # 가장 위(먼) 밴드
            offset = (near['center'] - cx) / (width / 2.0)
            angle = (far['center'] - near['center']) / (width / 2.0)
            state.detected = True
            state.offset = float(np.clip(offset, -1.0, 1.0))
            state.angle = float(np.clip(angle, -1.0, 1.0))
            state.confidence = float(len(valid) / num_bands)
        else:
            state.detected = False
            state.offset = 0.0
            state.angle = 0.0
            state.confidence = 0.0

        self.publisher.publish(state)

        if self.publish_debug:
            self._publish_debug(roi, analysis_img, bev_src, bands, state, msg)

        if bool(self.get_parameter('publish_control').value):
            self._publish_test_control(state)

    # ------------------------------------------------------------------ #
    #  테스트용 PD 조향을 /control로 직접 발행
    # ------------------------------------------------------------------ #
    def _publish_test_control(self, state: LaneState):
        trim = float(self.get_parameter('steer_trim').value)
        if math.isnan(trim):
            trim = float(self.vehicle_config.get('STEER_TRIM', 0.0))
        kp = float(self.get_parameter('steer_kp').value)
        kd = float(self.get_parameter('steer_kd').value)
        sign = float(self.get_parameter('steering_sign').value)
        throttle = float(self.get_parameter('test_throttle').value)

        derivative = state.offset - self.prev_offset
        self.prev_offset = state.offset

        control = Control()
        control.header.stamp = self.get_clock().now().to_msg()
        control.header.frame_id = 'lane_node'
        if state.detected:
            steer = sign * (kp * state.offset + kd * derivative) + trim
            control.steering = float(np.clip(steer, -1.0, 1.0))
            control.throttle = throttle
        else:
            control.steering = float(np.clip(trim, -1.0, 1.0))
            control.throttle = 0.0
        self.control_pub.publish(control)

    # ------------------------------------------------------------------ #
    #  검출점 그리기 (좌=파랑, 우=빨강, 중심=노랑) — analysis 이미지 좌표계
    # ------------------------------------------------------------------ #
    def _draw_bands(self, img, bands):
        w = img.shape[1]
        cv2.line(img, (w // 2, 0), (w // 2, img.shape[0]), (120, 120, 120), 1)
        for b in bands:
            if not b['valid']:
                continue
            y = int(b['y'])
            if b['left'] is not None:
                cv2.circle(img, (int(b['left']), y), 3, (255, 0, 0), -1)    # 파랑
            if b['right'] is not None:
                cv2.circle(img, (int(b['right']), y), 3, (0, 0, 255), -1)   # 빨강
            cv2.circle(img, (int(b['center']), y), 3, (0, 255, 255), -1)    # 노랑

    def _encode_publish(self, image, publisher, source_msg, frame_id):
        ok, encoded = cv2.imencode(
            '.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return
        out = CompressedImage()
        out.header.stamp = source_msg.header.stamp
        out.header.frame_id = frame_id
        out.format = 'jpeg'
        out.data = encoded.tobytes()
        publisher.publish(out)

    # ------------------------------------------------------------------ #
    #  디버그 2장: raw(원본+사다리꼴) / bev(펼친 BEV+검출점) 별도 토픽 발행
    # ------------------------------------------------------------------ #
    def _publish_debug(self, roi, analysis_img, bev_src, bands, state, source_msg):
        text = (f"det={state.detected} off={state.offset:+.2f} "
                f"ang={state.angle:+.2f} conf={state.confidence:.2f}")

        # raw 패널: 원본 ROI + 사다리꼴(초록) + 상태 텍스트
        raw = roi.copy()
        if bev_src is not None:
            cv2.polylines(raw, [bev_src.astype(np.int32)], True, (0, 255, 0), 1)
        cv2.putText(raw, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 255, 0), 1, cv2.LINE_AA)
        self._encode_publish(raw, self.debug_raw_pub, source_msg, 'lane_debug_raw')

        # bev 패널: 펼친 BEV(또는 BEV off면 ROI) + 검출점
        bev = analysis_img.copy()
        self._draw_bands(bev, bands)
        self._encode_publish(bev, self.debug_bev_pub, source_msg, 'lane_debug_bev')


def main(args=None):
    rclpy.init(args=args)
    node = LaneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
