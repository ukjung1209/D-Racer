import math
import os
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int8
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
        # 갈림길 hugging 힌트(-1=우 라인, 0=평소 양쪽, +1=좌 라인) 구독
        self.declare_parameter('branch_hint_topic', 'lane/branch_hint')
        self.declare_parameter('debug_raw_topic', 'lane/debug/raw')   # 원본+사다리꼴
        self.declare_parameter('debug_bev_topic', 'lane/debug/bev')   # 펼친 BEV+검출점
        self.declare_parameter('publish_debug', True)
        self.declare_parameter('jpeg_quality', 90)
        self.declare_parameter('vehicle_config_file', get_default_vehicle_config_path())

        # --- 검출 튜닝 파라미터 (매 프레임 다시 읽어 live 튜닝 가능) ---
        self.declare_parameter('lane_color', 'white')           # orange | white | dark
        self.declare_parameter('hsv_lower', [8.0, 90.0, 90.0])
        self.declare_parameter('hsv_upper', [26.0, 255.0, 255.0])
        self.declare_parameter('binary_threshold', 160)         # dark 모드용
        # white 모드: 채도 낮고(색 없음) 명도 높은(밝은) 픽셀 = 진짜 흰색만
        self.declare_parameter('white_s_max', 40)               # 채도 상한 (낮을수록 회색·색깔 배제)
        self.declare_parameter('white_v_min', 200)              # 명도 하한 (높을수록 완전 흰색만)
        # 빛반사 억제: top-hat은 '커널보다 작은 밝은 구조'(얇은 차선)만 남기고
        # 넓고 부드러운 밝은 덩어리(글레어)는 없앤다. 밝기 절대값이 아니라 국소
        # 대비를 보므로 어두운 프레임의 흰선도 그대로 산다. 0이면 끔.
        self.declare_parameter('white_tophat_ksize', 21)        # 글레어보다 작고 차선폭보다 큰 커널(px). 글레어 남으면 ↓, 차선 끊기면 ↑
        self.declare_parameter('white_tophat_min', 18)          # top-hat 대비 하한 (글레어 남으면 ↑, 차선 끊기면 ↓)
        self.declare_parameter('roi_top_px', 45)                # -1이면 vehicle_config ROI_TOP
        self.declare_parameter('num_bands', 10)
        self.declare_parameter('min_band_pixels', 12)
        self.declare_parameter('line_split_gap_px', 40)         # 좌/우 라인 분리 gap
        self.declare_parameter('lane_half_width_px', 90)        # 한쪽만 보일 때 반대편 추정
        # 갈림길 hugging bias: 추종할 라인에서 이만큼만 안쪽으로 붙는다(작을수록 라인에 밀착)
        self.declare_parameter('hug_bias_px', 45)
        # hugging 중 추종 라인이 '안쪽(중앙)'으로 이 픽셀보다 확 튀면 = 라인 사라지고
        # 가운데 V(섬)가 등장한 것 → 그건 무시하고 마지막 위치로 coast(직진 유지).
        self.declare_parameter('hug_track_tol_px', 40)
        # 차선을 잃어도 직전 좌/우 라인 앵커를 이만큼 프레임 유지.
        # 짧은 dropout에선 방향 기억을 살려 재등장 시 좌/우 오분류를 막는다.
        self.declare_parameter('lost_hold_frames', 15)
        # 한쪽 라인만 보일 때 '반대편으로 넘어간' 판정(오배정 의심)이 이 프레임 수만큼
        # 연속돼야 앵커를 뒤집는다. 순간 오검출 한두 프레임으로 좌/우가 뒤집히는 것 방지.
        self.declare_parameter('flip_confirm_frames', 3)

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

        branch_hint_topic = str(self.get_parameter('branch_hint_topic').value)

        self.subscription = self.create_subscription(
            CompressedImage, image_topic, self.image_callback, 10)
        self.branch_hint = 0   # -1=우 라인만, 0=평소 양쪽, +1=좌 라인만 (decision이 발행)
        self.create_subscription(
            Int8, branch_hint_topic, self.branch_hint_callback, 10)
        self.publisher = self.create_publisher(LaneState, lane_topic, 10)
        self.debug_raw_pub = self.create_publisher(CompressedImage, debug_raw_topic, 10)
        self.debug_bev_pub = self.create_publisher(CompressedImage, debug_bev_topic, 10)
        self.control_pub = self.create_publisher(Control, control_topic, 10)

        self.prev_offset = 0.0
        self.prev_center = None   # 직전 프레임 차선중심 (offset 시드 / cold-start 좌우판정)
        self.prev_left = None     # 직전 프레임 왼쪽 라인 x (per-line 연속 추적 앵커)
        self.prev_right = None    # 직전 프레임 오른쪽 라인 x
        self.flip_count = 0       # 좌/우 반전 후보가 연속된 프레임 수 (히스테리시스)
        self.prev_hug_line = None  # 직전 프레임에 hugging하던 라인 x (연속 추적 시드)
        self.lost_count = 0       # 연속으로 차선을 잃은 프레임 수

        self.get_logger().info(
            f'lane_node started: image_topic={image_topic}, lane_topic={lane_topic}, '
            f'debug_raw={debug_raw_topic}, debug_bev={debug_bev_topic}, '
            f'publish_debug={self.publish_debug}'
        )

    # ------------------------------------------------------------------ #
    #  vehicle_config.yaml fallback 로딩
    # ------------------------------------------------------------------ #
    def branch_hint_callback(self, msg: Int8):
        self.branch_hint = int(msg.data)

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
        elif lane_color == 'white':
            # 진짜 흰색 = 채도 낮음(S<=s_max) AND 명도 높음(V>=v_min).
            # 밝기만 보면 밝은 회색 매트도 잡히므로 HSV로 색 없는 밝은 픽셀만 고른다.
            hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            s_max = int(self.get_parameter('white_s_max').value)
            v_min = int(self.get_parameter('white_v_min').value)
            lower = np.array([0, 0, v_min], dtype=np.uint8)
            upper = np.array([180, s_max, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            # 빛반사(넓고 부드러운 밝은 덩어리) 제거: 커널보다 작은 밝은 구조만 남긴다.
            ks = int(self.get_parameter('white_tophat_ksize').value)
            if ks >= 3:
                tmin = int(self.get_parameter('white_tophat_min').value)
                kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
                tophat = cv2.morphologyEx(hsv[:, :, 2], cv2.MORPH_TOPHAT, kern)
                mask = cv2.bitwise_and(mask, (tophat >= tmin).astype(np.uint8) * 255)
        else:  # dark
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            thr = int(self.get_parameter('binary_threshold').value)
            _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)

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
                       half_width, seed_center, seed_left, seed_right,
                       branch_hint, hug_bias, hug_line_seed, hug_track_tol):
        h = mask.shape[0]
        band_h = max(1, h // num_bands)
        bands = []
        # per-line 앵커: 직전 프레임의 좌/우 라인 x로 시드한다. 한쪽 라인만 보일 때
        # '중심 대비 좌/우'가 아니라 '어느 라인에 더 가까운가'로 판정하므로,
        # 커브에서 라인이 화면중앙을 넘어와도 같은 라인으로 계속 인식된다.
        running_left = seed_left
        running_right = seed_right
        running_center = seed_center   # 앵커가 아직 없을 때(cold start)만 쓰는 폴백
        running_line = hug_line_seed   # hugging 중 추종하는 라인 x (밴드 아래→위 연속)

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
            # split_gap 넘는 틈으로 라인 클러스터(각 라인의 평균 x) 목록을 만든다
            cuts = np.nonzero(diffs > split_gap)[0] if diffs.size > 0 else np.empty(0, int)
            if cuts.size:
                bounds = [0] + [int(c) + 1 for c in cuts] + [xs_sorted.size]
                clusters = [float(xs_sorted[bounds[k]:bounds[k + 1]].mean())
                            for k in range(len(bounds) - 1)]
            else:
                clusters = [float(xs_sorted.mean())]

            if branch_hint != 0:
                # hugging: 추종하던 바깥 라인 하나만. 안쪽(중앙)으로 확 튀는 클러스터
                # (=가운데 V/섬)는 무시하고, 진짜 라인이 사라진 구간은 coast(직진 유지).
                if running_line is None:
                    # 시작 앵커: 힌트 쪽 최외곽 라인
                    running_line = clusters[0] if branch_hint > 0 else clusters[-1]
                else:
                    nearest = min(clusters, key=lambda c: abs(c - running_line))
                    # 좌 hugging이면 안쪽=오른쪽(+), 우 hugging이면 안쪽=왼쪽(-).
                    # 안쪽으로 hug_track_tol 넘게 튀면 라인 소실로 보고 running_line 유지(coast).
                    inner_jump = (
                        (branch_hint > 0 and nearest > running_line + hug_track_tol) or
                        (branch_hint < 0 and nearest < running_line - hug_track_tol))
                    if not inner_jump:
                        running_line = nearest      # 바깥/근처 → 진짜 라인, 따라감(벌어져도 OK)
                    # inner_jump이면 running_line 그대로 두고 coast
                if branch_hint > 0:
                    left_x = running_line           # 왼쪽 라인만 표시(오른쪽 무시)
                    center = running_line + hug_bias
                else:
                    right_x = running_line          # 오른쪽 라인만 표시(왼쪽 무시)
                    center = running_line - hug_bias
            elif len(clusters) >= 2:
                # 평소, 라인 둘 이상: 직전 좌/우 앵커에 '가장 가까운' 클러스터로 배정한다.
                # 앵커가 아직 없으면(cold start) 가장 큰 틈으로 좌/우를 나눈다.
                if running_left is not None and running_right is not None:
                    left_x = min(clusters, key=lambda c: abs(c - running_left))
                    right_x = min(clusters, key=lambda c: abs(c - running_right))
                    if left_x >= right_x:   # 같은/역전 클러스터가 잡히면 틈 분리로 폴백
                        split = int(np.argmax(diffs))
                        left_x = float(xs_sorted[:split + 1].mean())
                        right_x = float(xs_sorted[split + 1:].mean())
                else:
                    split = int(np.argmax(diffs))
                    left_x = float(xs_sorted[:split + 1].mean())
                    right_x = float(xs_sorted[split + 1:].mean())
                center = (left_x + right_x) / 2.0
            else:
                # 평소, 한쪽 라인만: 직전 좌/우 라인 중 '더 가까운' 쪽으로 판정하고
                # 반대편은 lane 폭으로 추정. (중심 대비 판정이 아니라 라인 연속성 기반)
                boundary = clusters[0]
                dl = abs(boundary - running_left) if running_left is not None else float('inf')
                dr = abs(boundary - running_right) if running_right is not None else float('inf')
                if dl == float('inf') and dr == float('inf'):
                    # 앵커 없음(초기/장기 소실) → 최선으로 화면 중심 기준 판정
                    if boundary < running_center:
                        dl = 0.0
                    else:
                        dr = 0.0
                if dl <= dr:
                    left_x = boundary
                    center = boundary + half_width
                else:
                    right_x = boundary
                    center = boundary - half_width

            center = float(np.clip(center, 0.0, width))
            running_center = center
            # 본 라인만 앵커 갱신, 안 보인 쪽은 직전값 유지(안정적 앵커)
            if left_x is not None:
                running_left = left_x
            if right_x is not None:
                running_right = right_x
            bands.append({
                'valid': True, 'y': (y0 + y1) // 2,
                'left': left_x, 'right': right_x, 'center': center,
            })

        # running_line = 마지막으로 알던 추종 라인 위치(사라짐 구간에도 유지) → 다음 프레임 시드
        return bands, running_line

    # ------------------------------------------------------------------ #
    #  프레임 간 좌/우 앵커 확정 (플립 히스테리시스)
    # ------------------------------------------------------------------ #
    def _commit_bottom_anchors(self, band, half_width, flip_confirm):
        """가장 아래(가까운) 밴드로 다음 프레임의 좌/우 앵커를 갱신한다.

        두 라인이 다 보이면 명확하니 바로 확정. 한쪽만 보일 때는 그 라인이 반대편
        앵커를 침범(좌 라인인데 우 앵커보다 오른쪽 등)하면 오배정으로 의심해,
        flip_confirm 프레임 연속으로 그럴 때만 앵커를 뒤집는다. 그 전까지는 이전
        앵커를 유지해 순간 오검출로 좌/우가 뒤집히는 것을 막는다.
        """
        nl, nr = band['left'], band['right']
        if nl is not None and nr is not None:
            self.prev_left, self.prev_right = nl, nr
            self.flip_count = 0
            return

        min_gap = float(half_width)   # 두 라인 최소 간격 가정
        if nl is not None:            # 왼쪽 라인만 봄
            crossed = (self.prev_right is not None and nl > self.prev_right - min_gap)
            if crossed:
                self.flip_count += 1
                if self.flip_count >= flip_confirm:
                    self.prev_left = nl
                    self.flip_count = 0
            else:
                self.prev_left = nl
                self.flip_count = 0
        elif nr is not None:          # 오른쪽 라인만 봄
            crossed = (self.prev_left is not None and nr < self.prev_left + min_gap)
            if crossed:
                self.flip_count += 1
                if self.flip_count >= flip_confirm:
                    self.prev_right = nr
                    self.flip_count = 0
            else:
                self.prev_right = nr
                self.flip_count = 0

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
        hug_bias = int(self.get_parameter('hug_bias_px').value)
        hug_track_tol = int(self.get_parameter('hug_track_tol_px').value)

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
        # hugging 중이면 직전 프레임 추종 라인을 시드로(연속 추적), 아니면 None(앵커 초기화)
        hug_line_seed = self.prev_hug_line if self.branch_hint != 0 else None
        bands, hug_line = self._analyze_bands(
            mask, width, num_bands, min_pixels, split_gap, half_width, seed_center,
            self.prev_left, self.prev_right,
            self.branch_hint, hug_bias, hug_line_seed, hug_track_tol)

        valid = [b for b in bands if b['valid']]

        # 다음 프레임 시드: 가장 아래(가까운) 밴드로 좌/우 앵커를 갱신한다.
        # 평소(hugging 아님)에만 플립 히스테리시스로 앵커를 확정한다. 차선을 잠깐
        # 잃어도 앵커를 lost_hold_frames만큼 유지해야 재등장 시 좌/우가 안 뒤집힌다.
        # 오래 잃으면(홀드 초과) 앵커를 버리고 다음 재등장 때 중앙 기준으로 재시작.
        if valid:
            if self.branch_hint == 0:
                self._commit_bottom_anchors(
                    valid[0], half_width,
                    int(self.get_parameter('flip_confirm_frames').value))
            self.prev_center = valid[0]['center']
            self.lost_count = 0
        else:
            self.lost_count += 1
            if self.lost_count > int(self.get_parameter('lost_hold_frames').value):
                self.prev_center = None
                self.prev_left = None
                self.prev_right = None
                self.flip_count = 0

        # hugging 라인 연속 추적 시드: 마지막으로 알던 추종 라인 위치(사라짐 구간에도 유지).
        # hugging 아니면 리셋해 다음 hugging 시작 때 최외곽 라인으로 재앵커.
        self.prev_hug_line = hug_line if self.branch_hint != 0 else None

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
                f"ang={state.angle:+.2f} conf={state.confidence:.2f} "
                f"hint={self.branch_hint:+d}")

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
