"""대회 주행용 통합 결정 노드 — 차선추종(곡률 가변속도) + 신호등 + 갈림길 + 동적장애물.

── 분업용 3개(decision_arrow/light/obstacle_node)를 합친 것 ──
세 노드가 전부 /control에 20Hz로 publish 하던 걸 그대로 두면 서로 값을 덮어써
싸운다. 최종 주행은 반드시 이 노드 하나에서 조향/스로틀을 중재한다.

파이프라인:
  camera ─┬─ lane_node   → /lane/state          ─┐
          ├─ object_node → /object/detections    ─┼─ decision_node → /control
          └─ camera/image/compressed ────────────┘   └→ /lane/branch_hint

── 대회 순서(순방향)는 트랙 배치가 강제한다 (별도 시퀀서 없음) ──
  0. 빨간불 대기   : 신호등 STOPPED로 시작 → 초록 전엔 스로틀 0
  1. 초록불 출발   : green 연속 → GO
  2. S자 코스     : 곡률 가변속도(직선 빠르게 / 코너 미리 감속) + 슬루 제한
  3. 갈림길       : left/right 표지판 보이면 즉시 그 라인 hugging + 1회 stop-and-go
  4. 동적 장애물   : 빨강 구간 감속 → 아루코 정지 → 통과 후 가속
  5. 빨간불 정지   : red 연속 → STOPPED (도착)
  갈림길 표지판·빨강 바닥·아루코는 각 구간에만 나타나므로, 아래 스로틀 중재만으로
  순서가 저절로 지켜진다(단계 카운터가 어긋날 위험이 없다).

── 조향: 항상 /lane/state PD 차선추종 하나 ──
  갈림길 미션만 lane_node에 branch_hint(±1)를 보내 확정 방향 라인만 hugging
  하게 한다(조향식 자체는 동일, 라인 중점만 옮김).

── 스로틀: 곡률 가변속도를 기준(cruise)으로, 미션들이 min으로 깎는다 ──
  cruise = base × (1 − speed_ka·|angle| − speed_ko·|offset|)  [min_throttle, base] 클램프
  최종 target = min(cruise, 각 미션 목표):
    · 신호등 STOPPED      → 0   (초록 전/도착)
    · 동적장애물 STOP     → 0   (아루코 정지)
    · 동적장애물 APPROACH → cruise × slow_factor  (빨강 구간 감속)
    · 갈림길 stop-and-go  → 0   (표지판 앞 1회 정지)
  target에 슬루 제한(_slew_throttle): 가속은 완만, 감속/정지는 즉시.
  → 모든 정지(신호등/갈림길/장애물)가 0으로 통일돼, 재출발 시 항상 0부터 부드럽게
    가속한다(S자 변곡점 순간 가속 튐 방지).

각 미션은 enable_*_mission 플래그로 끌 수 있다. 한 미션만 켜면 노드 하나로 단일
미션 테스트가 된다(끈 미션은 스로틀/힌트에 관여 안 함).

파라미터는 src/config/decision.yaml. 실시간 튜닝: ros2 param set /decision_node ...

⚠️ 아루코는 시스템 cv2 4.5.4(apt)에 있음. pip로 opencv 깔지 말 것(카메라가 죽음).
"""

import math
import os
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from std_msgs.msg import Int8
from sensor_msgs.msg import CompressedImage
from control_msgs.msg import Control
from inference_msgs.msg import LaneState, DetectionArray


def get_default_vehicle_config_path():
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'src' / 'config' / 'vehicle_config.yaml'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/D-Racer/src/config/vehicle_config.yaml'


def clamp(value, low, high):
    return max(low, min(high, value))


class DecisionNode(Node):
    """곡률 가변속도 PD 위에 신호등/갈림길/동적장애물을 얹어 /control을 낸다."""

    def __init__(self):
        super().__init__('decision_node')

        # --- 토픽/공통 ---
        self.declare_parameter('lane_topic', 'lane/state')
        self.declare_parameter('detection_topic', 'object/detections')
        self.declare_parameter('image_topic', 'camera/image/compressed')
        self.declare_parameter('control_topic', '/control')
        self.declare_parameter('branch_hint_topic', 'lane/branch_hint')
        self.declare_parameter('debug_topic', 'obstacle/debug')
        self.declare_parameter('command_hz', 20.0)
        self.declare_parameter('vehicle_config_file', get_default_vehicle_config_path())
        self.declare_parameter('publish_debug', True)
        self.declare_parameter('jpeg_quality', 90)

        # --- 차선추종 PD 조향 (live 튜닝 가능) ---
        self.declare_parameter('steer_kp', 0.8)          # offset 비례
        self.declare_parameter('steer_kd', 0.4)          # offset 변화율
        self.declare_parameter('steer_ka', 0.0)          # angle 전방주시 피드포워드
        self.declare_parameter('steering_sign', -1.0)    # 방향 반대면 부호 뒤집기
        self.declare_parameter('steer_trim', float('nan'))  # NaN이면 vehicle_config STEER_TRIM
        self.declare_parameter('base_throttle', 0.28)    # 직선 최고속(가변속도가 코너에서 깎음)
        self.declare_parameter('lane_timeout_sec', 0.5)  # 이 시간 넘게 lane 없으면 정지

        # --- 곡률 기반 가변속도 (직선 빠르게 / 코너 미리 감속) ---
        # target = base × (1 − speed_ka·|angle| − speed_ko·|offset|), [min_throttle, base] 클램프.
        # angle(먼 밴드 곡률)은 코너 진입 전 예측 감속, offset(현재 이탈)은 보정. steer가 아니라
        # angle을 쓰는 이유: S자 변곡점에서 steer는 0을 지나며 속도가 튀지만 angle은 안 그렇다.
        self.declare_parameter('speed_ka', 0.4)          # |angle| 예측 감속 gain
        self.declare_parameter('speed_ko', 0.4)          # |offset| 보정 감속 gain
        # 감속 하한 + '움직일 때 최소 스로틀'(모터 데드밴드 회피). 이 밑으론 0(완전정지)만 나옴.
        self.declare_parameter('min_throttle', 0.20)
        # 슬루 제한: 가속은 초당 이 값까지만 올림(완만), 감속/정지는 즉시.
        self.declare_parameter('throttle_accel_rate', 0.5)

        # --- 미션 on/off ---
        self.declare_parameter('enable_light_mission', True)
        self.declare_parameter('enable_fork_mission', True)
        self.declare_parameter('enable_obstacle_mission', True)

        # --- 신호등 미션 (초록 출발 / 빨강 정지) ---
        self.declare_parameter('light_conf_min', 0.5)     # 신호등 최소 confidence
        self.declare_parameter('green_votes_needed', 3)   # 초록불 이만큼 연속 보면 출발
        self.declare_parameter('red_votes_needed_light', 2)  # 빨간불 이만큼 연속 보면 정지

        # --- 좌/우 갈림길 (단순화: 표지판 보이면 즉시 그 라인 hugging + 1회 stop-and-go) ---
        self.declare_parameter('sign_conf_min', 0.5)      # 표지판 최소 confidence
        self.declare_parameter('sign_area_min', 0.01)     # 이 크기 이상이어야 방향 인정(멀면 무시)
        self.declare_parameter('hug_hold_sec', 1.0)       # 표지판 놓치고 이 시간 뒤 hugging 해제
        self.declare_parameter('sign_stop_sec', 0.7)      # 표지판 앞 정지 시간(stop-and-go), 0=끔
        self.declare_parameter('stop_lane_conf_min', 0.5)  # 두 차선 이 conf 이상일 때만 정지

        # --- 동적 장애물 (camera/image에서 OpenCV로 직접 감지) ---
        self.declare_parameter('slow_factor', 0.7)          # 빨강 구간 감속 배율(cruise 대비)
        self.declare_parameter('red_roi_top_frac', 0.5)     # 빨강 ROI 상단 시작(높이 비율)
        self.declare_parameter('red_h_lo', 8)               # H 0근방 상한
        self.declare_parameter('red_h_hi', 172)             # H 180근방 하한
        self.declare_parameter('red_s_min', 90)             # 채도 하한
        self.declare_parameter('red_v_min', 60)             # 명도 하한
        self.declare_parameter('red_ratio_trigger', 0.15)   # ROI 빨강 비율↑이면 빨강구간
        self.declare_parameter('red_votes_needed_obstacle', 3)  # 감속 확정 연속 프레임 수
        self.declare_parameter('red_clear_time_sec', 1.0)   # 빨강 사라짐 판정 시간
        # 아루코 (이 마커 = 6X6_50 / ID3. 320x160 저해상도 실측 튜닝값)
        self.declare_parameter('aruco_dict', '6X6_50')
        self.declare_parameter('aruco_target_id', 3)        # 이 ID만 정지(-1=아무 ID나)
        self.declare_parameter('aruco_upscale', 2)          # 검출 전 업스케일(6X6 저해상도 보정)
        self.declare_parameter('aruco_clahe', True)         # 검출 전 국소대비 강화
        self.declare_parameter('aruco_min_area', 0.0)       # 마커 area_ratio 하한
        self.declare_parameter('aruco_votes_needed', 1)     # 정지 확정 표수(1='바로 멈춤')
        self.declare_parameter('clear_time_sec', 1.5)       # 마커 사라짐→재출발 판정 시간
        self.declare_parameter('aruco_min_perimeter_rate', 0.01)
        self.declare_parameter('aruco_poly_accuracy', 0.08)
        self.declare_parameter('aruco_adaptive_win_max', 45)
        self.declare_parameter('aruco_error_correction', 0.4)  # ↓일수록 오독을 valid로 안 만듦
        self.declare_parameter('aruco_max_border_bits', 0.35)

        lane_topic = str(self.get_parameter('lane_topic').value)
        detection_topic = str(self.get_parameter('detection_topic').value)
        image_topic = str(self.get_parameter('image_topic').value)
        control_topic = str(self.get_parameter('control_topic').value)
        branch_hint_topic = str(self.get_parameter('branch_hint_topic').value)
        debug_topic = str(self.get_parameter('debug_topic').value)
        command_hz = float(self.get_parameter('command_hz').value)
        self.command_dt = 1.0 / command_hz   # 슬루 제한용 tick 간격(초)
        self.publish_debug = bool(self.get_parameter('publish_debug').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)

        self.vehicle_config = self._load_vehicle_config(
            os.path.expanduser(str(self.get_parameter('vehicle_config_file').value))
        )

        # --- 차선/공통 상태 ---
        self.lane_state = None
        self.lane_stamp = None
        self.detections = None
        self.prev_offset = 0.0
        self.prev_throttle = 0.0       # 직전 tick throttle (가속 슬루 제한 기준)

        # --- 신호등 상태머신 ---
        self.light_state = 'STOPPED'   # STOPPED | GO
        self.vote_green = 0
        self.vote_red = 0

        # --- 갈림길 상태 (단순화) ---
        self.latched_dir = 0           # +1 = left, -1 = right, 0 = 평소 양쪽 추종
        self.last_sign_time = None     # 표지판을 마지막으로 본 시각
        self.stop_start = None         # 정지 시작 시각, None=정지 안 함
        self.stopped_done = False      # 이번 갈림길 정지 완료 여부(1회)

        # --- 동적 장애물 상태머신 ---
        self.obstacle_state = 'CRUISE'  # CRUISE | APPROACH | STOP | BOOST
        self.red_votes = 0
        self.aruco_votes = 0
        self.last_red_time = None
        self.last_aruco_time = None
        # 디버그용 최근 관측값
        self.red_ratio = 0.0
        self.red_roi_box = None
        self.aruco_boxes = []
        self.aruco_raw_ids = []
        self.lanes_visible = False

        self._init_aruco()

        self.create_subscription(LaneState, lane_topic, self.lane_callback, 10)
        self.create_subscription(
            DetectionArray, detection_topic, self.detection_callback, 10)
        self.create_subscription(
            CompressedImage, image_topic, self.image_callback, 10)
        self.control_pub = self.create_publisher(Control, control_topic, 10)
        self.branch_hint_pub = self.create_publisher(Int8, branch_hint_topic, 10)
        self.debug_pub = self.create_publisher(CompressedImage, debug_topic, 10)

        self.timer = self.create_timer(1.0 / command_hz, self.control_loop)

        self.get_logger().info(
            f'decision_node started: lane={lane_topic}, detect={detection_topic}, '
            f'image={image_topic}, control={control_topic}, hz={command_hz}, '
            f'light={bool(self.get_parameter("enable_light_mission").value)}, '
            f'fork={bool(self.get_parameter("enable_fork_mission").value)}, '
            f'obstacle={bool(self.get_parameter("enable_obstacle_mission").value)}'
        )

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

    # ================================================================== #
    #  구독 콜백
    # ================================================================== #
    def lane_callback(self, msg: LaneState):
        self.lane_state = msg
        self.lane_stamp = self.get_clock().now()

    def detection_callback(self, msg: DetectionArray):
        """object_node의 YOLO 검출(left/right 표지판 + green/red 신호등)."""
        self.detections = msg
        if bool(self.get_parameter('enable_light_mission').value):
            self._update_light(msg)
        if bool(self.get_parameter('enable_fork_mission').value):
            self._update_sign_latch(msg)

    def image_callback(self, msg: CompressedImage):
        """동적장애물: 카메라 원본에서 빨강 구간 + 아루코를 OpenCV로 직접 감지."""
        if not bool(self.get_parameter('enable_obstacle_mission').value):
            return
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warning('failed to decode compressed image')
            return

        now = self.get_clock().now()
        red_present, self.red_ratio, self.red_roi_box = self._detect_red(frame)
        self.aruco_boxes = self._detect_aruco(frame)
        self.lanes_visible = self._lane_is_fresh() and bool(self.lane_state.detected)

        if red_present:
            self.last_red_time = now

        # 감속 진입: 빨간 배경 + 양쪽 차선이 함께 보임 (연속 표수로 확정)
        if red_present and self.lanes_visible:
            self.red_votes += 1
        else:
            self.red_votes = 0

        min_area = float(self.get_parameter('aruco_min_area').value)
        aruco_seen = any(b[4] >= min_area for b in self.aruco_boxes)
        if aruco_seen:
            self.aruco_votes += 1
            self.last_aruco_time = now
        else:
            self.aruco_votes = 0

        # 이벤트 기반 전이 (시간 기반은 control_loop의 _update_obstacle_time)
        if self.obstacle_state == 'CRUISE':
            if self.red_votes >= int(self.get_parameter('red_votes_needed_obstacle').value):
                self.get_logger().info(
                    f'obstacle: CRUISE → APPROACH (red_ratio={self.red_ratio:.3f})')
                self.obstacle_state = 'APPROACH'
                self.red_votes = 0
        elif self.obstacle_state == 'APPROACH':
            if self.aruco_votes >= int(self.get_parameter('aruco_votes_needed').value):
                mid = max((b[4] for b in self.aruco_boxes), default=0.0)
                self.get_logger().info(
                    f'obstacle: APPROACH → STOP (aruco area={mid:.3f})')
                self.obstacle_state = 'STOP'
                self.aruco_votes = 0

        if self.publish_debug:
            self._publish_debug(frame, msg)

    # ================================================================== #
    #  신호등: green/red 연속 프레임 수로 출발/정지 전이
    # ================================================================== #
    def _has_light(self, msg: DetectionArray, class_name):
        conf_min = float(self.get_parameter('light_conf_min').value)
        for det in msg.detections:
            if det.class_name == class_name and det.confidence >= conf_min:
                return True
        return False

    def _update_light(self, msg: DetectionArray):
        green = self._has_light(msg, 'green')
        red = self._has_light(msg, 'red')

        if self.light_state == 'STOPPED':
            self.vote_green = self.vote_green + 1 if green else 0
            self.vote_red = 0
            if self.vote_green >= int(self.get_parameter('green_votes_needed').value):
                self.get_logger().info('light: STOPPED → GO (green)')
                self.light_state = 'GO'
                self.vote_green = 0
        elif self.light_state == 'GO':
            self.vote_red = self.vote_red + 1 if red else 0
            self.vote_green = 0
            if self.vote_red >= int(self.get_parameter('red_votes_needed_light').value):
                self.get_logger().info('light: GO → STOPPED (red)')
                self.light_state = 'STOPPED'
                self.vote_red = 0

    # ================================================================== #
    #  갈림길: 표지판 보이면 즉시 그 라인 hugging + 1회 stop-and-go
    # ================================================================== #
    def _pick_sign(self, msg: DetectionArray):
        """left/right 중 conf·크기 조건을 통과한 가장 큰(가까운) 표지판."""
        conf_min = float(self.get_parameter('sign_conf_min').value)
        area_min = float(self.get_parameter('sign_area_min').value)
        best = None
        for det in msg.detections:
            if det.class_name not in ('left', 'right'):
                continue
            if det.confidence < conf_min or det.area_ratio < area_min:
                continue
            if best is None or det.area_ratio > best.area_ratio:
                best = det
        return best

    def _update_sign_latch(self, msg: DetectionArray):
        """표지판이 보이면 즉시 그 방향으로 hugging 시작 (투표/상태머신 없음)."""
        best = self._pick_sign(msg)
        if best is None:
            return
        self.last_sign_time = self.get_clock().now()
        direction = 1 if best.class_name == 'left' else -1
        if direction != self.latched_dir:
            was_cruise = (self.latched_dir == 0)
            self.latched_dir = direction
            if was_cruise:
                self.stopped_done = False   # 새 갈림길 진입 → 정지 재무장
            self.get_logger().info(
                f'sign → hug {"LEFT" if direction == 1 else "RIGHT"} '
                f'(conf={best.confidence:.2f}, area={best.area_ratio:.3f})')

    def _release_hug_if_sign_lost(self):
        """표지판을 hug_hold_sec 이상 못 보면 hugging 해제 → 평소 양쪽 추종."""
        if self.latched_dir == 0 or self.last_sign_time is None:
            return
        gone = (self.get_clock().now() - self.last_sign_time).nanoseconds / 1e9
        if gone >= float(self.get_parameter('hug_hold_sec').value):
            self.get_logger().info('sign lost → release hug (cruise)')
            self.latched_dir = 0
            self.last_sign_time = None
            self.stopped_done = False
            self.stop_start = None

    def _fork_stopping(self):
        """갈림길 stop-and-go: 표지판 hugging 중 + 두 차선 확실할 때 1회 정지.
        현재 정지 중이면 True를 돌려 스로틀 0을 먹인다."""
        if not bool(self.get_parameter('enable_fork_mission').value):
            return False
        now = self.get_clock().now()
        # 정지 트리거 (조건 충족 시 이번 갈림길 1회)
        if (self.latched_dir != 0 and not self.stopped_done and self.stop_start is None
                and float(self.get_parameter('sign_stop_sec').value) > 0.0
                and self._lane_is_fresh() and self.lane_state.detected
                and self.lane_state.confidence
                >= float(self.get_parameter('stop_lane_conf_min').value)):
            self.stop_start = now
        # 정지 실행
        if self.stop_start is not None:
            elapsed = (now - self.stop_start).nanoseconds / 1e9
            if elapsed < float(self.get_parameter('sign_stop_sec').value):
                return True
            self.stop_start = None
            self.stopped_done = True
        return False

    # ================================================================== #
    #  동적 장애물: 빨간 구간 / 아루코 (OpenCV 직접 감지)
    # ================================================================== #
    def _detect_red(self, frame):
        h, w = frame.shape[:2]
        top = int(clamp(float(self.get_parameter('red_roi_top_frac').value), 0.0, 0.95) * h)
        roi = frame[top:, :]
        if roi.size == 0:
            return False, 0.0, (0, top, w, h)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_lo = int(self.get_parameter('red_h_lo').value)
        h_hi = int(self.get_parameter('red_h_hi').value)
        s_min = int(self.get_parameter('red_s_min').value)
        v_min = int(self.get_parameter('red_v_min').value)
        lower1 = np.array([0, s_min, v_min], dtype=np.uint8)
        upper1 = np.array([h_lo, 255, 255], dtype=np.uint8)
        lower2 = np.array([h_hi, s_min, v_min], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

        ratio = float((mask > 0).mean())
        present = ratio >= float(self.get_parameter('red_ratio_trigger').value)
        return present, ratio, (0, top, w, h)

    def _init_aruco(self):
        """아루코 딕셔너리 준비. DetectorParameters는 매 프레임 새로 만들어 live 튜닝.

        ⚠️ 시스템 cv2 4.5.4(apt)에 cv2.aruco가 있음 — pip opencv 금지(카메라 죽음).
        4.5.4 함수형 API, 4.7+ ArucoDetector 클래스 둘 다 대응.
        """
        self.enable_aruco = True
        self.aruco_dict = None
        self._clahe = None
        self._aruco_class_api = hasattr(cv2.aruco, 'ArucoDetector')
        if not hasattr(cv2, 'aruco'):
            self.get_logger().error(
                'cv2.aruco가 없다(pip opencv가 시스템 4.5.4를 가리는지 확인). '
                '아루코 정지 비활성화(빨강 감속만 동작).')
            self.enable_aruco = False
            return

        dict_name = str(self.get_parameter('aruco_dict').value)
        dict_id = getattr(cv2.aruco, f'DICT_{dict_name}', None)
        if dict_id is None:
            self.get_logger().warning(
                f'알 수 없는 aruco_dict={dict_name} → DICT_6X6_50 사용')
            dict_id = cv2.aruco.DICT_6X6_50

        if hasattr(cv2.aruco, 'getPredefinedDictionary'):
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        else:
            self.aruco_dict = cv2.aruco.Dictionary_get(dict_id)

        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        target = int(self.get_parameter('aruco_target_id').value)
        self.get_logger().info(
            f'aruco enabled: dict=DICT_{dict_name}, target_id={target}, '
            f'upscale={int(self.get_parameter("aruco_upscale").value)}')

    def _build_aruco_params(self):
        try:
            p = cv2.aruco.DetectorParameters()
        except AttributeError:
            p = cv2.aruco.DetectorParameters_create()
        p.minMarkerPerimeterRate = float(
            self.get_parameter('aruco_min_perimeter_rate').value)
        p.maxMarkerPerimeterRate = 4.0
        p.adaptiveThreshWinSizeMin = 3
        p.adaptiveThreshWinSizeMax = int(
            self.get_parameter('aruco_adaptive_win_max').value)
        p.adaptiveThreshWinSizeStep = 4
        p.polygonalApproxAccuracyRate = float(
            self.get_parameter('aruco_poly_accuracy').value)
        p.maxErroneousBitsInBorderRate = float(
            self.get_parameter('aruco_max_border_bits').value)
        p.errorCorrectionRate = float(
            self.get_parameter('aruco_error_correction').value)
        try:
            p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        except Exception:
            pass
        return p

    def _detect_aruco(self, frame):
        if not self.enable_aruco:
            return []
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if bool(self.get_parameter('aruco_clahe').value) and self._clahe is not None:
            gray = self._clahe.apply(gray)

        scale = max(1, int(self.get_parameter('aruco_upscale').value))
        det_gray = gray if scale == 1 else cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        params = self._build_aruco_params()
        if self._aruco_class_api:
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, params)
            corners, ids, _ = detector.detectMarkers(det_gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                det_gray, self.aruco_dict, parameters=params)

        boxes = []
        self.aruco_raw_ids = []
        if ids is None:
            return boxes
        self.aruco_raw_ids = [int(i) for i in ids.flatten()]
        target = int(self.get_parameter('aruco_target_id').value)
        frame_area = float(w * h)
        for corner, marker_id in zip(corners, ids.flatten()):
            marker_id = int(marker_id)
            if target >= 0 and marker_id != target:
                continue
            pts = corner.reshape(-1, 2) / float(scale)
            x1 = int(np.clip(pts[:, 0].min(), 0, w - 1))
            y1 = int(np.clip(pts[:, 1].min(), 0, h - 1))
            x2 = int(np.clip(pts[:, 0].max(), 0, w - 1))
            y2 = int(np.clip(pts[:, 1].max(), 0, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            area_ratio = float(cv2.contourArea(pts.astype(np.float32))) / frame_area
            boxes.append((x1, y1, x2, y2, area_ratio, marker_id))
        return boxes

    def _update_obstacle_time(self):
        """STOP→BOOST(마커 통과), BOOST→CRUISE(구간 이탈), APPROACH→CRUISE(빨강 오검출)."""
        now = self.get_clock().now()
        red_gone = (self.last_red_time is None or
                    (now - self.last_red_time).nanoseconds / 1e9
                    >= float(self.get_parameter('red_clear_time_sec').value))

        if self.obstacle_state == 'STOP':
            if self.last_aruco_time is not None:
                gone = (now - self.last_aruco_time).nanoseconds / 1e9
                if gone >= float(self.get_parameter('clear_time_sec').value):
                    self.get_logger().info('obstacle: STOP → BOOST (marker passed)')
                    self.obstacle_state = 'BOOST'
                    self.aruco_votes = 0
        elif self.obstacle_state == 'BOOST':
            if red_gone:
                self.get_logger().info('obstacle: BOOST → CRUISE (zone cleared)')
                self.obstacle_state = 'CRUISE'
                self.red_votes = 0
        elif self.obstacle_state == 'APPROACH':
            if red_gone:
                self.get_logger().info('obstacle: APPROACH → CRUISE (red lost)')
                self.obstacle_state = 'CRUISE'
                self.red_votes = 0

    # ================================================================== #
    #  조향 PD + 곡률 가변속도 (미션 공통)
    # ================================================================== #
    def _lane_is_fresh(self):
        if self.lane_state is None or self.lane_stamp is None:
            return False
        timeout = float(self.get_parameter('lane_timeout_sec').value)
        age = (self.get_clock().now() - self.lane_stamp).nanoseconds / 1e9
        return age <= timeout

    def _lane_pd_steer(self, trim):
        if not (self._lane_is_fresh() and self.lane_state.detected):
            return None
        kp = float(self.get_parameter('steer_kp').value)
        kd = float(self.get_parameter('steer_kd').value)
        ka = float(self.get_parameter('steer_ka').value)
        sign = float(self.get_parameter('steering_sign').value)
        offset = self.lane_state.offset
        angle = self.lane_state.angle
        derivative = offset - self.prev_offset
        self.prev_offset = offset
        return sign * (kp * offset + kd * derivative + ka * angle) + trim

    def _curve_throttle(self, base):
        """직선은 base로 빠르게, 코너는 미리 감속. angle(예측)+offset(보정) 기반."""
        if not (self._lane_is_fresh() and self.lane_state.detected):
            return base
        ka = float(self.get_parameter('speed_ka').value)
        ko = float(self.get_parameter('speed_ko').value)
        curve = ka * abs(self.lane_state.angle) + ko * abs(self.lane_state.offset)
        target = base * (1.0 - curve)
        return clamp(target, float(self.get_parameter('min_throttle').value), base)

    def _slew_throttle(self, target):
        """가속은 슬루 제한(완만), 감속은 즉시. (min_throttle 하한/정지는 control_loop에서 처리)"""
        max_up = float(self.get_parameter('throttle_accel_rate').value) * self.command_dt
        if target > self.prev_throttle + max_up:
            out = self.prev_throttle + max_up
        else:
            out = target
        self.prev_throttle = out
        return out

    # ================================================================== #
    #  스로틀 중재: cruise(가변속도)를 미션 목표들의 min으로 깎는다
    # ================================================================== #
    def _arbitrate_throttle(self, cruise, fork_stopping):
        target = cruise
        if bool(self.get_parameter('enable_light_mission').value):
            if self.light_state == 'STOPPED':
                target = min(target, 0.0)           # 초록 전 / 도착
        if bool(self.get_parameter('enable_obstacle_mission').value):
            if self.obstacle_state == 'STOP':
                target = min(target, 0.0)           # 아루코 정지
            elif self.obstacle_state == 'APPROACH':
                target = min(target, cruise * float(self.get_parameter('slow_factor').value))
            # CRUISE / BOOST → 제약 없음
        if fork_stopping:
            target = min(target, 0.0)               # 갈림길 stop-and-go
        return target

    def control_loop(self):
        # 시간 기반 상태 전이
        if bool(self.get_parameter('enable_fork_mission').value):
            self._release_hug_if_sign_lost()
        if bool(self.get_parameter('enable_obstacle_mission').value):
            self._update_obstacle_time()

        # 갈림길 hugging 힌트: latched_dir(±1) → lane_node가 그 라인만 hugging (0=평소)
        hint = (self.latched_dir
                if bool(self.get_parameter('enable_fork_mission').value) else 0)
        self.branch_hint_pub.publish(Int8(data=int(hint)))

        trim = float(self.get_parameter('steer_trim').value)
        if math.isnan(trim):
            trim = float(self.vehicle_config.get('STEER_TRIM', 0.0))

        control = Control()
        control.header.stamp = self.get_clock().now().to_msg()
        control.header.frame_id = 'decision'

        min_throttle = float(self.get_parameter('min_throttle').value)

        pd_steer = self._lane_pd_steer(trim)

        if pd_steer is None:
            # 차선을 잃으면 안전 정지. 재출발은 min_throttle부터(데드밴드 0~min 건너뜀).
            self.prev_offset = 0.0
            self.prev_throttle = min_throttle
            control.steering = float(clamp(trim, -1.0, 1.0))
            control.throttle = 0.0
            self.control_pub.publish(control)
            return

        control.steering = float(clamp(pd_steer, -1.0, 1.0))

        cruise = self._curve_throttle(float(self.get_parameter('base_throttle').value))
        target = self._arbitrate_throttle(cruise, self._fork_stopping())
        if target <= 0.0:
            # 완전 정지(정지 미션). 재출발은 min_throttle부터 → 데드밴드 구간 안 지나감.
            control.throttle = 0.0
            self.prev_throttle = min_throttle
        else:
            # 움직일 땐 min_throttle 이상 보장(모터 데드밴드 회피) + 가속 슬루 제한.
            # → 출력은 {0(완전정지)} ∪ [min_throttle, base] 범위로 보장됨.
            control.throttle = float(self._slew_throttle(max(target, min_throttle)))

        self.control_pub.publish(control)

    # ================================================================== #
    #  디버그 오버레이 (빨강 ROI/아루코 + 세 미션 상태) → /obstacle/debug
    # ================================================================== #
    def _publish_debug(self, frame, source_msg):
        overlay = frame.copy()
        if self.red_roi_box is not None:
            x0, y0, x1, y1 = self.red_roi_box
            cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 255), 1)
        for (x1, y1, x2, y2, area, mid) in self.aruco_boxes:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(overlay, f'id{mid} {area:.3f}', (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

        hug = {1: 'L', -1: 'R', 0: '-'}[self.latched_dir]
        text = (f'light:{self.light_state} hug:{hug} obs:{self.obstacle_state} '
                f'red={self.red_ratio:.3f} lanes={int(self.lanes_visible)} '
                f'aruco={len(self.aruco_boxes)} raw={self.aruco_raw_ids}')
        cv2.putText(overlay, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)

        ok, encoded = cv2.imencode(
            '.jpg', overlay, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return
        out = CompressedImage()
        out.header.stamp = source_msg.header.stamp
        out.header.frame_id = 'decision_debug'
        out.format = 'jpeg'
        out.data = encoded.tobytes()
        self.debug_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = DecisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
