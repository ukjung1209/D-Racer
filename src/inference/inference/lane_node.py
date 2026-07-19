"""차선 검출 노드 (회색 매트 + 흰색 사이드라인, OpenCV+numpy, 딥러닝 없음).

파이프라인: ROI 크롭 → BEV 원근 → 흰색 마스크 → (평소) 베이스 탐색 + 슬라이딩 윈도우
체인 → 차선 중심 → LaneState(offset/angle/confidence) 발행. 순수 분석은 lane_core.py에
있고(rclpy 없이 pytest 가능), 이 노드는 ROS 파라미터만 읽어 넘긴다.

── 평소 경로(branch_hint==0) 요약 ──
  1) 베이스 탐색  : 하단 1/3 히스토그램으로 좌/우 베이스 x를 프레임당 한 번 정한다(정체성
                    판정). 피크 2개(간격 hist_width_tol) → 콜드스타트, 피크 1개 → 직전 베이스에
                    base_match_tol_px 이내로 매칭. 앵커 경쟁·유령·유도 없음(좌/우 x 순서가 정체성).
  2) 윈도우 체인  : 각 베이스에서 위로 num_bands개 윈도우(중심 ± window_margin_px)를 쌓아
                    라인을 따라 올라간다. 빈 창 chain_max_gap 연속이면 종료. 체인은 자기 창만
                    보므로 좌/우가 서로 먹는 게 구조적으로 불가능 → 플립 근절.
  3) center·피팅  : 두 체인 점이면 중점, 한 점이면 그 점 ± half_est(실측 반폭). center 전체를
                    가중(픽셀수) 1차 피팅해 offset/angle. angle 슬루(angle_max_delta) 유지.
  4) coast        : 베이스 없음/유효 center<2면 직전 offset/angle을 coast_max_frames 동안 유지
                    (detected=True, confidence 0.7^n 감쇠). 만료 시 정지(detected=False).
  5) 백분위 마스크: white V 하한을 프레임 상위 백분위(white_v_percentile)로 적응 → 밝은 매트 배제.
  6) 디버그       : BEV에 베이스 삼각형(좌파랑·우빨강)/체인 점/center(노랑)/피팅선(노랑)/state 표시.
  ※ hugging(branch_hint±1) 경로는 lane_core.analyze_bands가 담당 — 체인 도입 전과 동작 불변.

── 튜닝 가이드(전부 decision.yaml lane_node) ──
  · 휜 라인 추종 느림 → window_margin_px ↑ (잡음 유입되면 ↓)
  · 라인 끊겨 체인 종료 → chain_max_gap ↑
  · 상실 후 너무 빨리 정지 → coast_max_frames ↑ (급발진 위험 시 ↓)
  · 재획득 시 좌/우 흔들림 → base_match_tol_px ↓ 또는 hist_width_tol ↓
  · 잡음 오검출     → cluster_min_pixels ↑
  · 프레임 떨림      → angle_max_delta ↓
  · 글레어/밝은 매트 → white_v_percentile ↑ (또는 white_tophat_min ↑)
"""

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

from inference import lane_core


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
        self.declare_parameter('white_v_min', 200)              # 명도 하한 절대 바닥(흰선 없는 프레임 보호)
        # (mod 5) V 하한을 프레임 밝기 상위 백분위로 적응 조정 → 밝은 매트 배제
        self.declare_parameter('white_v_percentile', 98.0)
        # 빛반사 억제: top-hat은 '커널보다 작은 밝은 구조'(얇은 차선)만 남기고
        # 넓고 부드러운 밝은 덩어리(글레어)는 없앤다. 밝기 절대값이 아니라 국소
        # 대비를 보므로 어두운 프레임의 흰선도 그대로 산다. 0이면 끔.
        self.declare_parameter('white_tophat_ksize', 21)        # 글레어보다 작고 차선폭보다 큰 커널(px). 글레어 남으면 ↓, 차선 끊기면 ↑
        self.declare_parameter('white_tophat_min', 18)          # top-hat 대비 하한 (글레어 남으면 ↑, 차선 끊기면 ↓)
        self.declare_parameter('roi_top_px', 45)                # -1이면 vehicle_config ROI_TOP
        self.declare_parameter('num_bands', 10)
        self.declare_parameter('line_split_gap_px', 40)         # 클러스터 분리 gap (hugging이 사용)
        self.declare_parameter('lane_half_width_px', 90)        # 한쪽 체인만 있을 때 반대편 추정(반폭 초기값)
        # 클러스터 픽셀수 하한: hugging 후보 필터 + 체인 윈도우 점유 하한(min_band_pixels).
        self.declare_parameter('cluster_min_pixels', 30)
        # 클러스터 폭 상한(hugging 외 valid_clusters용). 십자 가로획 판별.
        self.declare_parameter('cluster_max_width_px', 60)
        # 갈림길 hugging bias: 추종할 라인에서 이만큼만 안쪽으로 붙는다(작을수록 라인에 밀착)
        self.declare_parameter('hug_bias_px', 45)
        # hugging 중 추종 라인이 '안쪽(중앙)'으로 이 픽셀보다 확 튀면 = 라인 사라지고
        # 가운데 V(섬)가 등장한 것 → 그건 무시하고 마지막 위치로 coast(직진 유지).
        self.declare_parameter('hug_track_tol_px', 40)
        # 체인 상실 유예(coast 만료 전 유지 프레임). lost_hold_frames는 호환 위해 남겨둠.
        self.declare_parameter('lost_hold_frames', 15)
        # 베이스 탐색: 하단 1/3 히스토그램 두 피크 간격 허용오차(차선폭 대비).
        self.declare_parameter('hist_width_tol', 0.25)
        self.declare_parameter('half_width_est_alpha', 0.2)  # 실측 반폭 EMA 계수
        # 발행 angle 프레임당 최대 변화량(초과분 클램프). offset에는 미적용.
        self.declare_parameter('angle_max_delta', 0.3)
        # === 슬라이딩 윈도우 체인 신규 파라미터 (branch_hint==0) ===
        # 윈도우 폭: 현재 중심 x ± 이 값. ↑=휜 라인 더 잘 따라가나 잡음 유입, ↓=엄격.
        self.declare_parameter('window_margin_px', 40)
        # 빈 창이 이만큼 연속이면 그 체인 종료(끊긴 라인 위로 무한 추정 방지).
        self.declare_parameter('chain_max_gap', 2)
        # coast: 베이스/center 상실 후 직전 방향을 유지하는 최대 프레임(초과 시 정지).
        self.declare_parameter('coast_max_frames', 8)
        # 베이스 1피크 매칭 허용치(px): 직전 베이스에서 이 안이면 그 side로 매칭.
        self.declare_parameter('base_match_tol_px', 45)

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

        self.prev_offset = None   # None=D항 보호(첫 프레임/재획득). offset-0 가짜 킥 방지 (PD용)
        self.prev_hug_line = None  # 직전 프레임에 hugging하던 라인 x (연속 추적 시드)

        # ── 체인 경로(branch_hint==0)의 프레임 간 상태: 이것이 전부다 ──
        self.prev_base_left = None    # 직전 프레임 좌 베이스 x (1피크 매칭 기준)
        self.prev_base_right = None   # 직전 프레임 우 베이스 x
        # 실측 반폭 EMA(좌/우 체인 공존 윈도우에서 갱신). 초기값=lane_half_width_px.
        self.half_est = float(self.get_parameter('lane_half_width_px').value)
        self.coast_count = 0          # 연속 coast 프레임 수(0=정상). coast_max 초과 시 정지.
        self.last_pub_offset = None   # coast 홀드용 직전 발행 offset/angle/confidence
        self.last_pub_angle = None
        self.last_pub_conf = 0.0
        self.prev_pub_angle = None    # angle 슬루용 직전 발행 angle. 미검출/hugging 시 리셋.

        self.track_state = 'LOST'     # TRACKING | COAST{n} | LOST
        self._angle_slew_active = False  # (디버그) 이번 프레임 angle 슬루 발동 여부
        self._dbg_base_left = None    # (디버그) 이번 프레임 베이스 마커 위치
        self._dbg_base_right = None

        self.get_logger().info(
            f'lane_node started: image_topic={image_topic}, lane_topic={lane_topic}, '
            f'debug_raw={debug_raw_topic}, debug_bev={debug_bev_topic}, '
            f'publish_debug={self.publish_debug}'
        )

    # ------------------------------------------------------------------ #
    #  vehicle_config.yaml fallback 로딩
    # ------------------------------------------------------------------ #
    def branch_hint_callback(self, msg: Int8):
        new_hint = int(msg.data)
        if self.branch_hint != 0 and new_hint == 0:
            # hugging 해제(비0 → 0): 웜스타트. 콜드스타트(피크 2개)를 기다리는 대신
            # 추종하던 라인(prev_hug_line)으로 좌/우 베이스를 시드한다 — 갈림길 직후 두
            # 차선이 아직 선명하지 않아도 1피크 매칭으로 즉시 추적을 잇는다. 실제 관측이
            # 오면 실측으로 대체된다. 시드는 매칭 prior일 뿐이라 coast는 0에서 시작하고
            # 직전 발행값은 비워(재획득 실패 시 stale 대신 정지) 안전하게 복귀한다.
            self.prev_base_left, self.prev_base_right = lane_core.hug_warmstart_bases(
                self.branch_hint, self.prev_hug_line, self.half_est)
            self.coast_count = 0
            self.last_pub_offset = None
            self.last_pub_angle = None
            self.last_pub_conf = 0.0
            self.prev_pub_angle = None
        elif self.branch_hint == 0 and new_hint != 0:
            # hugging 진입(0 → 비0): coast 상태를 걷어낸다(hugging 중 coast 미발동 보장).
            self.coast_count = 0
        self.branch_hint = new_hint

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
            # (mod 5) V 하한을 프레임의 상위 백분위로 적응 조정: 흰선이 있으면 그 밝기
            # 근처만 통과시켜 밝은 회색 매트를 배제한다. 흰 픽셀이 없는 프레임에선
            # 백분위가 매트 밝기라 낮아지므로, 절대 하한 white_v_min으로 바닥을 받친다.
            v_pct = float(self.get_parameter('white_v_percentile').value)
            v_min = int(max(int(self.get_parameter('white_v_min').value),
                            np.percentile(hsv[:, :, 2], v_pct)))
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

        state = LaneState()
        state.header.stamp = self.get_clock().now().to_msg()
        state.header.frame_id = 'camera'
        cx = width / 2.0
        self._angle_slew_active = False
        self._dbg_base_left = None
        self._dbg_base_right = None

        if self.branch_hint != 0:
            # ── hugging 경로 (지시 방향 최외곽 라인 추종, 체인 로직과 무관) ──
            # hugging 첫 프레임(prev_hug_line=None)엔 최외곽 클러스터(반사광·잡음 위험)를
            # 콜드픽업하는 대신, 진입 직전 cruise 베이스(그 방향 라인의 마지막 위치)를
            # 시드로 넘긴다. 갈림길 진입에서 엉뚱한 바깥 조각을 잡는 것을 줄인다. 그 베이스도
            # None이면 seed=None으로 두어 lane_core의 최외곽 픽업을 폴백으로 쓴다.
            hug_seed = self.prev_hug_line
            if hug_seed is None:
                hug_seed = (self.prev_base_left if self.branch_hint > 0
                            else self.prev_base_right)
            bands, hug_line = lane_core.analyze_bands(
                mask, width, num_bands, split_gap, half_width,
                self.branch_hint, hug_bias, hug_seed, hug_track_tol,
                int(self.get_parameter('cluster_min_pixels').value))
            self.prev_hug_line = hug_line     # 사라짐 구간에도 마지막 위치 유지
            self.coast_count = 0
            self.prev_pub_angle = None         # hugging은 slew 미적용, 기억 리셋
            self.track_state = 'TRACKING'
            valid = [b for b in bands if b['valid']]
            if len(valid) >= 2:
                near, far = valid[0], valid[-1]
                state.detected = True
                state.offset = float(np.clip((near['center'] - cx) / cx, -1.0, 1.0))
                state.angle = float(np.clip(
                    (far['center'] - near['center']) / cx, -1.0, 1.0))
                state.confidence = float(len(valid) / num_bands)
            else:
                state.detected = False
                state.offset = 0.0
                state.angle = 0.0
                state.confidence = 0.0
        else:
            # ── 평소 경로: 베이스 탐색 → 슬라이딩 윈도우 체인 → 피팅 → (상실 시) coast ──
            self.prev_hug_line = None
            base_left, base_right = lane_core.find_bases(
                mask, self.half_est,
                float(self.get_parameter('hist_width_tol').value),
                self.prev_base_left, self.prev_base_right,
                float(self.get_parameter('base_match_tol_px').value))
            self._dbg_base_left = base_left
            self._dbg_base_right = base_right

            bands = []
            centers = []
            new_half = self.half_est
            cbl = cbr = None
            if base_left is not None or base_right is not None:
                bands, centers, new_half, cbl, cbr = lane_core.analyze_chains(
                    mask, width, num_bands, base_left, base_right,
                    int(self.get_parameter('window_margin_px').value),
                    int(self.get_parameter('cluster_min_pixels').value),
                    int(self.get_parameter('chain_max_gap').value),
                    self.half_est, float(half_width),
                    float(self.get_parameter('half_width_est_alpha').value))

            coast_max = int(self.get_parameter('coast_max_frames').value)
            if len(centers) >= 2:
                # 정상: 유효 밴드 center 전체에 가중 직선 피팅 → offset/angle.
                ys = [c[0] for c in centers]
                xs = [c[1] for c in centers]
                ws = [c[2] for c in centers]
                offset, angle = lane_core.fit_lane_line(ys, xs, ws, width)
                offset = float(np.clip(offset, -1.0, 1.0))
                angle = float(np.clip(angle, -1.0, 1.0))
                # 발행 angle 변화율 제한(한 프레임 피팅선 폭주 흡수).
                angle, self._angle_slew_active = lane_core.slew_limit(
                    angle, self.prev_pub_angle,
                    float(self.get_parameter('angle_max_delta').value))
                confidence = float(len(centers) / num_bands)
                # 프레임 간 상태 갱신: 베이스·반폭·직전 발행값. coast 리셋.
                self.half_est = new_half
                if cbl is not None:
                    self.prev_base_left = cbl
                if cbr is not None:
                    self.prev_base_right = cbr
                self.coast_count = 0
                self.last_pub_offset = offset
                self.last_pub_angle = angle
                self.last_pub_conf = confidence
                self.prev_pub_angle = angle
                self.track_state = 'TRACKING'
                state.detected = True
                state.offset = offset
                state.angle = angle
                state.confidence = confidence
            else:
                # 베이스 없음 또는 유효 center < 2 → coast(추측 금지, confidence 감쇠).
                self.coast_count += 1
                det, off, ang, conf, expired = lane_core.coast_decision(
                    self.coast_count, coast_max, self.last_pub_offset,
                    self.last_pub_angle, self.last_pub_conf)
                if expired:
                    # coast 만료 → 정지 계약(detected=False). 상태 전면 리셋, 재획득은
                    # 피크 2개 콜드스타트만 허용(직전 베이스 없어 1피크 매칭 자연 배제).
                    self.coast_count = 0
                    self.prev_base_left = None
                    self.prev_base_right = None
                    self.last_pub_offset = None
                    self.last_pub_angle = None
                    self.last_pub_conf = 0.0
                    self.prev_pub_angle = None
                    self.track_state = 'LOST'
                    state.detected = False
                    state.offset = 0.0
                    state.angle = 0.0
                    state.confidence = 0.0
                else:
                    # 직전 방향 유지(베이스·half_est 홀드). detected=True로 정지 방지.
                    self.prev_pub_angle = ang
                    self.track_state = 'COAST%d' % self.coast_count
                    state.detected = det
                    state.offset = off
                    state.angle = ang
                    state.confidence = conf

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

        control = Control()
        control.header.stamp = self.get_clock().now().to_msg()
        control.header.frame_id = 'lane_node'
        if state.detected:
            # 재획득 첫 tick(prev_offset=None)엔 D항 0, 이후 정상 미분(offset-0 킥 방지).
            if self.prev_offset is None:
                derivative = 0.0
            else:
                derivative = state.offset - self.prev_offset
            self.prev_offset = state.offset
            steer = sign * (kp * state.offset + kd * derivative) + trim
            control.steering = float(np.clip(steer, -1.0, 1.0))
            control.throttle = throttle
        else:
            # 차선 상실 → prev_offset None으로 리셋(재획득 첫 tick D항 킥 방지).
            self.prev_offset = None
            control.steering = float(np.clip(trim, -1.0, 1.0))
            control.throttle = 0.0
        self.control_pub.publish(control)

    # ------------------------------------------------------------------ #
    #  검출점 그리기 (좌 체인=파랑, 우 체인=빨강, 중심=노랑) — analysis 이미지 좌표계
    # ------------------------------------------------------------------ #
    def _draw_bands(self, img, bands):
        w = img.shape[1]
        h = img.shape[0]
        cv2.line(img, (w // 2, 0), (w // 2, h), (120, 120, 120), 1)

        # 이번 프레임 베이스 위치: 화면 하단에 좌=파랑/우=빨강 삼각형 마커.
        def _base_marker(x, color):
            if x is None:
                return
            cv2.drawMarker(img, (int(x), h - 5), color,
                           cv2.MARKER_TRIANGLE_UP, 12, 2)
        _base_marker(self._dbg_base_left, (255, 0, 0))
        _base_marker(self._dbg_base_right, (0, 0, 255))

        for b in bands:
            if not b['valid']:
                continue
            y = int(b['y'])
            if b.get('left') is not None:
                cv2.circle(img, (int(b['left']), y), 3, (255, 0, 0), -1)    # 파랑
            if b.get('right') is not None:
                cv2.circle(img, (int(b['right']), y), 3, (0, 0, 255), -1)   # 빨강
            cv2.circle(img, (int(b['center']), y), 3, (0, 255, 255), -1)    # 노랑

        # 가중 피팅 직선을 노란 선으로 (유효 밴드 2개 이상).
        valid = [b for b in bands if b['valid']]
        if len(valid) >= 2:
            try:
                ys = np.array([b['y'] for b in valid], dtype=np.float64)
                xs = np.array([b['center'] for b in valid], dtype=np.float64)
                ws = np.array([b.get('weight', 0) for b in valid], dtype=np.float64)
                if ws.sum() <= 0:
                    ws = np.ones_like(ys)
                # fit_lane_line과 동일하게 sqrt 가중(polyfit w는 잔차에 곱해져 w²로 들어감).
                slope, intercept = np.polyfit(ys, xs, 1, w=np.sqrt(ws))
                y_lo, y_hi = 0, h - 1
                p_lo = (int(slope * y_lo + intercept), y_lo)
                p_hi = (int(slope * y_hi + intercept), y_hi)
                cv2.line(img, p_lo, p_hi, (0, 255, 255), 1)
            except Exception:
                pass

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
        bl = f'{int(self._dbg_base_left)}' if self._dbg_base_left is not None else '-'
        br = f'{int(self._dbg_base_right)}' if self._dbg_base_right is not None else '-'
        # coast 중이면 잔여 프레임 수를 함께 표시.
        coast_max = int(self.get_parameter('coast_max_frames').value)
        state_str = self.track_state
        if self.track_state.startswith('COAST'):
            state_str = f'{self.track_state}(-{coast_max - self.coast_count})'
        text = (f"det={state.detected} off={state.offset:+.2f} "
                f"ang={state.angle:+.2f} conf={state.confidence:.2f} "
                f"hint={self.branch_hint:+d} state={state_str} "
                f"hw={self.half_est:.0f} base=L{bl}/R{br}"
                f"{' aSLEW' if self._angle_slew_active else ''}")

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
