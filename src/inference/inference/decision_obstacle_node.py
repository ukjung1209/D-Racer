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


def clamp(value, low, high):
    return max(low, min(high, value))


class DecisionObstacleNode(Node):
    """차선추종 + 동적 장애물(빨강 구간 감속 → 아루코 정지 → 통과 후 가속) 주행.

    ── 팀 병렬개발용 파생 노드 ──
    차선추종 PD 코어는 decision_node와 100% 동일하게 유지하고(steering만 담당),
    throttle에 '동적 장애물 상태머신'만 얹었다. 나중에 decision_node로 merge할 때는
    이 상태머신을 그대로 옮기고 control_loop의 throttle 우선순위만 합치면 된다.

    ── 동적 장애물 인식은 전부 이 노드 안에서 OpenCV로 직접 한다 ──
    object_node(YOLO)/`/object/detections`에 의존하지 않고 카메라 원본
    (camera/image/compressed)을 직접 구독해서:
      1) 빨간 트랙 배경 감지  : 하단 ROI의 빨강 픽셀 비율(HSV, H 0근방+180근방)
      2) 아루코 마커 감지     : cv2.aruco (바닥에 누웠다 올라오면 갑자기 검출됨)
    조향에 필요한 '양쪽 흰색 차선 보임'은 lane_node의 /lane/state.detected로 판단한다
    (그 자체도 흰색 OpenCV 마스크 결과라 로직은 전부 OpenCV로 닫혀 있다).

    상태머신 (조향은 항상 차선추종, throttle만 상태로 제어):
      CRUISE   : 원속도(base_throttle) 주행. 빨간 배경 + 양쪽 차선이
                 red_votes_needed만큼 연속으로 보이면 감속(→APPROACH).
      APPROACH : 원속도의 slow_factor(기본 70%)로 감속 주행. 아루코 마커가
                 aruco_votes_needed만큼 보이면 즉시 정지(→STOP).
                 (빨강이 red_clear_time_sec 이상 사라지면 오검출로 보고 →CRUISE)
      STOP     : 정지(throttle 0). 마커가 clear_time_sec(기본 1.5초) 이상 안 보이면
                 (=올라온 마커가 다시 내려가 통과) 원속도로 재출발(→BOOST).
      BOOST    : 원속도로 빠르게 통과. 빨강 구간을 벗어나면(빨강이
                 red_clear_time_sec 이상 안 보임) 재무장(→CRUISE). BOOST 동안은
                 아직 빨강이 보여도 다시 감속하지 않아 같은 구간에서 재정지하지 않는다.
    신호등/갈림길 미션은 이 노드 범위 밖(다른 파생 노드/merge에서 담당).
    """

    def __init__(self):
        super().__init__('decision_obstacle_node')

        self.declare_parameter('lane_topic', 'lane/state')
        self.declare_parameter('image_topic', 'camera/image/compressed')
        self.declare_parameter('control_topic', '/control')
        self.declare_parameter('debug_topic', 'obstacle/debug')
        self.declare_parameter('command_hz', 20.0)
        self.declare_parameter('vehicle_config_file', get_default_vehicle_config_path())
        self.declare_parameter('publish_debug', True)
        self.declare_parameter('jpeg_quality', 90)

        # --- 차선추종 PD 제어 파라미터 (decision_node와 동일) ---
        self.declare_parameter('steer_kp', 0.8)
        self.declare_parameter('steer_kd', 0.3)
        self.declare_parameter('steer_ka', 0.0)
        self.declare_parameter('steering_sign', -1.0)
        self.declare_parameter('steer_trim', float('nan'))
        self.declare_parameter('base_throttle', 0.18)
        self.declare_parameter('lane_timeout_sec', 0.5)

        # --- 동적 장애물 미션 파라미터 (live 튜닝 가능) ---
        self.declare_parameter('enable_obstacle_mission', True)
        self.declare_parameter('slow_factor', 0.7)          # 빨강 구간 감속 배율 (원속도 대비)

        # 빨간 트랙 감지: 하단 ROI(red_roi_top_frac~1.0, 전폭)의 빨강 픽셀 비율.
        # HSV 빨강은 H가 0근방과 180근방 두 구간으로 갈라져서 둘 다 OR한다.
        self.declare_parameter('red_roi_top_frac', 0.5)     # ROI 상단 시작(프레임 높이 비율)
        self.declare_parameter('red_h_lo', 8)               # H 0근방 상한 (0~red_h_lo)
        self.declare_parameter('red_h_hi', 172)             # H 180근방 하한 (red_h_hi~180)
        self.declare_parameter('red_s_min', 90)             # 채도 하한(낮은 채도 배경 배제)
        self.declare_parameter('red_v_min', 60)             # 명도 하한(너무 어두운 픽셀 배제)
        self.declare_parameter('red_ratio_trigger', 0.15)   # ROI에서 빨강이 이 비율↑이면 빨강구간
        self.declare_parameter('red_votes_needed', 3)       # 감속 확정 연속 프레임 수
        self.declare_parameter('red_clear_time_sec', 1.0)   # 빨강 사라짐 판정 시간

        # 아루코 마커(동적 장애물)
        # ⚠️ 이 마커는 6X6_50 / ID3. 카메라가 320x160로 저해상도라 6X6은 그냥은
        #    거의 안 잡히고 엉뚱한 ID로 오독된다. 실측 튜닝 결과:
        #      - 검출 전 2배 업스케일(aruco_upscale=2) → 6X6 검출률 급상승
        #      - 오류보정 억제(aruco_error_correction=0.4) → 오독을 valid ID로
        #        복원하지 못하게 해 false ID 제거
        #      - 사각형 후보 통과 완화(min_perimeter_rate↓, poly_accuracy↑,
        #        adaptive 창 확대) → 저대비/기울어진 마커도 후보로 잡음
        #      - aruco_target_id로 해당 ID만 정지 → 남은 오독까지 무시
        self.declare_parameter('aruco_dict', '6X6_50')       # cv2.aruco.DICT_<이 값>
        self.declare_parameter('aruco_target_id', 3)         # 이 ID만 정지 트리거(-1=아무 ID나)
        self.declare_parameter('aruco_upscale', 2)           # 검출 전 업스케일 배율(6X6 저해상도 보정)
        self.declare_parameter('aruco_clahe', True)          # 검출 전 국소대비 강화(빨강/저대비 배경 보정)
        self.declare_parameter('aruco_min_area', 0.0)        # 마커 area_ratio 하한(0=아무 크기나)
        self.declare_parameter('aruco_votes_needed', 1)      # 정지 확정 표수('바로 멈춤'=1)
        self.declare_parameter('clear_time_sec', 1.5)        # 마커 사라짐→재출발 판정 시간
        # --- 아루코 검출기 파라미터 (live 튜닝, /obstacle/debug의 aruco= 보며 조정) ---
        self.declare_parameter('aruco_min_perimeter_rate', 0.01)  # 작은 마커 허용(기본 0.03)
        self.declare_parameter('aruco_poly_accuracy', 0.08)       # 기울어진 사각형 허용(기본 0.03)
        self.declare_parameter('aruco_adaptive_win_max', 45)      # 적응임계 창 최대(기본 23)
        self.declare_parameter('aruco_error_correction', 0.4)     # ↓일수록 오독을 valid로 안 만듦(기본 0.6)
        self.declare_parameter('aruco_max_border_bits', 0.35)     # 테두리 비트 오류 허용율(기본 0.35)

        lane_topic = str(self.get_parameter('lane_topic').value)
        image_topic = str(self.get_parameter('image_topic').value)
        control_topic = str(self.get_parameter('control_topic').value)
        debug_topic = str(self.get_parameter('debug_topic').value)
        command_hz = float(self.get_parameter('command_hz').value)
        self.publish_debug = bool(self.get_parameter('publish_debug').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)

        self.vehicle_config = self._load_vehicle_config(
            os.path.expanduser(str(self.get_parameter('vehicle_config_file').value))
        )

        self.lane_state = None
        self.lane_stamp = None
        self.prev_offset = 0.0

        # --- 장애물 상태머신 상태 ---
        self.mission_state = 'CRUISE'    # CRUISE | APPROACH | STOP | BOOST
        self.red_votes = 0
        self.aruco_votes = 0
        self.last_red_time = None        # 빨간 구간을 마지막으로 본 시각
        self.last_aruco_time = None      # 아루코를 마지막으로 본 시각

        # --- 디버그(오버레이/로그)용 최근 관측값 ---
        self.red_ratio = 0.0
        self.red_roi_box = None
        self.aruco_boxes = []            # target_id 필터 통과한 마커 [(x1,y1,x2,y2,area,id), ...]
        self.aruco_raw_ids = []          # 필터 전 검출된 모든 ID (오독 진단용)
        self.lanes_visible = False

        self._init_aruco()

        self.create_subscription(LaneState, lane_topic, self.lane_callback, 10)
        self.create_subscription(
            CompressedImage, image_topic, self.image_callback, 10)
        self.control_pub = self.create_publisher(Control, control_topic, 10)
        self.debug_pub = self.create_publisher(CompressedImage, debug_topic, 10)

        self.timer = self.create_timer(1.0 / command_hz, self.control_loop)

        self.get_logger().info(
            f'decision_obstacle_node started: lane_topic={lane_topic}, '
            f'image_topic={image_topic}, control_topic={control_topic}, '
            f'command_hz={command_hz}, aruco={self.enable_aruco}'
        )

    # ------------------------------------------------------------------ #
    #  아루코 검출기 준비 (opencv 버전별 API 차이 흡수 — object_node와 동일)
    # ------------------------------------------------------------------ #
    def _init_aruco(self):
        """아루코 딕셔너리를 만든다. DetectorParameters는 live 튜닝을 위해
        매 프레임 _build_aruco_params()로 새로 만든다.

        ⚠️ 이 보드의 시스템 cv2 4.5.4(apt)에 이미 cv2.aruco가 들어 있으니 pip로
        opencv를 절대 깔지 마라(pip opencv는 GStreamer가 없어 camera_node가 죽는다).
        4.5.4는 detectMarkers 함수형 API, 4.7+는 ArucoDetector 클래스라 둘 다 대응.
        """
        self.enable_aruco = True
        self.aruco_dict = None
        self._clahe = None
        self._aruco_class_api = hasattr(cv2.aruco, 'ArucoDetector')  # opencv 4.7+
        if not hasattr(cv2, 'aruco'):
            self.get_logger().error(
                'cv2.aruco가 없다. 시스템 cv2(4.5.4)엔 있어야 정상 — pip opencv가 '
                '시스템 걸 가리고 있는지 확인하라(pip opencv는 카메라를 죽인다). '
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

        # 국소대비 강화기(빨강/저대비 배경에서 검은 테두리를 또렷하게)
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        target = int(self.get_parameter('aruco_target_id').value)
        self.get_logger().info(
            f'aruco enabled: dict=DICT_{dict_name}, target_id={target}, '
            f'upscale={int(self.get_parameter("aruco_upscale").value)}')

    def _build_aruco_params(self):
        """현재 ROS 파라미터로 DetectorParameters를 만든다(매 프레임 → live 튜닝).

        저해상도 6X6 마커 실측 튜닝값이 기본값이다: 사각형 후보 통과는 완화하되
        (min_perimeter↓, poly_accuracy↑, adaptive 창 확대) 비트 오류보정은 조여
        (error_correction↓) 오독을 valid ID로 복원하지 못하게 한다.
        """
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
        # 코너 서브픽셀 보정: 저해상도에서 코너를 정밀화해 비트 샘플링/ID 디코드 안정화
        try:
            p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        except Exception:
            pass
        return p

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

    def lane_callback(self, msg: LaneState):
        self.lane_state = msg
        self.lane_stamp = self.get_clock().now()

    # ------------------------------------------------------------------ #
    #  카메라 프레임 → 빨간 구간/아루코 감지 + 이벤트 기반 상태 전이
    # ------------------------------------------------------------------ #
    def image_callback(self, msg: CompressedImage):
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

        # 빨강이 보이면 마지막 관측 시각 갱신 (구간 이탈 판정 기준)
        if red_present:
            self.last_red_time = now

        # 감속 진입 조건: 빨간 배경 + 양쪽 차선이 함께 보임 (연속 표수로 확정)
        if red_present and self.lanes_visible:
            self.red_votes += 1
        else:
            self.red_votes = 0

        # 아루코 관측 (min_area 이상만 유효) → 표수/마지막 관측 시각 갱신
        min_area = float(self.get_parameter('aruco_min_area').value)
        aruco_seen = any(b[4] >= min_area for b in self.aruco_boxes)
        if aruco_seen:
            self.aruco_votes += 1
            self.last_aruco_time = now
        else:
            self.aruco_votes = 0

        # --- 이벤트 기반 전이 (시간 기반 전이는 control_loop에서) ---
        if self.mission_state == 'CRUISE':
            if self.red_votes >= int(self.get_parameter('red_votes_needed').value):
                self.get_logger().info(
                    f'obstacle: CRUISE → APPROACH (red_ratio={self.red_ratio:.3f})')
                self.mission_state = 'APPROACH'
                self.red_votes = 0
        elif self.mission_state == 'APPROACH':
            if self.aruco_votes >= int(self.get_parameter('aruco_votes_needed').value):
                mid = max((b[4] for b in self.aruco_boxes), default=0.0)
                self.get_logger().info(
                    f'obstacle: APPROACH → STOP (aruco area={mid:.3f})')
                self.mission_state = 'STOP'
                self.aruco_votes = 0

        if self.publish_debug:
            self._publish_debug(frame, msg)

    # ------------------------------------------------------------------ #
    #  빨간 트랙 배경 감지: 하단 ROI의 빨강 픽셀 비율
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    #  아루코 마커 검출 → [(x1,y1,x2,y2,area_ratio,id), ...]
    # ------------------------------------------------------------------ #
    def _detect_aruco(self, frame):
        if not self.enable_aruco:
            return []
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 국소대비 강화(CLAHE): 빨강/저대비 배경에서 검은 테두리를 또렷하게 만들어
        # 사각형 후보 검출률을 올린다. 저해상도 6X6에 특히 효과적.
        if bool(self.get_parameter('aruco_clahe').value) and self._clahe is not None:
            gray = self._clahe.apply(gray)

        # 6X6 저해상도 보정: 업스케일해서 검출 후 좌표는 원본 스케일로 되돌린다.
        scale = max(1, int(self.get_parameter('aruco_upscale').value))
        det_gray = gray if scale == 1 else cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        params = self._build_aruco_params()
        if self._aruco_class_api:                 # opencv 4.7+
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, params)
            corners, ids, _ = detector.detectMarkers(det_gray)
        else:                                     # 함수형 API (4.5.4)
            corners, ids, _ = cv2.aruco.detectMarkers(
                det_gray, self.aruco_dict, parameters=params)

        boxes = []
        self.aruco_raw_ids = []      # 필터 전 검출된 모든 ID (오독 진단용)
        if ids is None:
            return boxes
        self.aruco_raw_ids = [int(i) for i in ids.flatten()]
        target = int(self.get_parameter('aruco_target_id').value)
        frame_area = float(w * h)
        for corner, marker_id in zip(corners, ids.flatten()):
            marker_id = int(marker_id)
            # target_id가 -1이 아니면 그 ID만 정지 트리거로 인정(오독 무시)
            if target >= 0 and marker_id != target:
                continue
            pts = corner.reshape(-1, 2) / float(scale)   # 업스케일 좌표 → 원본 좌표
            x1 = int(np.clip(pts[:, 0].min(), 0, w - 1))
            y1 = int(np.clip(pts[:, 1].min(), 0, h - 1))
            x2 = int(np.clip(pts[:, 0].max(), 0, w - 1))
            y2 = int(np.clip(pts[:, 1].max(), 0, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            area_ratio = float(cv2.contourArea(pts.astype(np.float32))) / frame_area
            boxes.append((x1, y1, x2, y2, area_ratio, marker_id))
        return boxes

    # ------------------------------------------------------------------ #
    #  시간 기반 전이: STOP→BOOST(마커 통과), BOOST→CRUISE(구간 이탈),
    #                  APPROACH→CRUISE(빨강 오검출 복귀)
    # ------------------------------------------------------------------ #
    def _update_mission_time(self):
        now = self.get_clock().now()
        red_gone = (self.last_red_time is None or
                    (now - self.last_red_time).nanoseconds / 1e9
                    >= float(self.get_parameter('red_clear_time_sec').value))

        if self.mission_state == 'STOP':
            if self.last_aruco_time is not None:
                gone = (now - self.last_aruco_time).nanoseconds / 1e9
                if gone >= float(self.get_parameter('clear_time_sec').value):
                    self.get_logger().info('obstacle: STOP → BOOST (marker passed)')
                    self.mission_state = 'BOOST'
                    self.aruco_votes = 0
        elif self.mission_state == 'BOOST':
            if red_gone:
                self.get_logger().info('obstacle: BOOST → CRUISE (zone cleared)')
                self.mission_state = 'CRUISE'
                self.red_votes = 0
        elif self.mission_state == 'APPROACH':
            if red_gone:
                self.get_logger().info('obstacle: APPROACH → CRUISE (red lost)')
                self.mission_state = 'CRUISE'
                self.red_votes = 0

    def _lane_is_fresh(self):
        if self.lane_state is None or self.lane_stamp is None:
            return False
        timeout = float(self.get_parameter('lane_timeout_sec').value)
        age = (self.get_clock().now() - self.lane_stamp).nanoseconds / 1e9
        return age <= timeout

    def _lane_pd_steer(self, trim):
        """차선추종 PD 조향값. 차선이 없으면 None. (decision_node와 동일)"""
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

    def _state_throttle(self, base):
        """상태별 목표 throttle. STOP은 0, APPROACH는 감속, 나머지는 원속도."""
        if self.mission_state == 'STOP':
            return 0.0
        if self.mission_state == 'APPROACH':
            return base * float(self.get_parameter('slow_factor').value)
        return base   # CRUISE / BOOST

    def control_loop(self):
        if bool(self.get_parameter('enable_obstacle_mission').value):
            self._update_mission_time()

        trim = float(self.get_parameter('steer_trim').value)
        if math.isnan(trim):
            trim = float(self.vehicle_config.get('STEER_TRIM', 0.0))

        control = Control()
        control.header.stamp = self.get_clock().now().to_msg()
        control.header.frame_id = 'decision'

        pd_steer = self._lane_pd_steer(trim)
        base = float(self.get_parameter('base_throttle').value)

        if pd_steer is not None:
            control.steering = float(clamp(pd_steer, -1.0, 1.0))
            # 조향은 항상 차선추종, throttle만 장애물 상태로 제어
            control.throttle = float(self._state_throttle(base))
        else:
            # 차선을 잃으면 안전 정지
            self.prev_offset = 0.0
            control.steering = float(clamp(trim, -1.0, 1.0))
            control.throttle = 0.0

        self.control_pub.publish(control)

    # ------------------------------------------------------------------ #
    #  디버그 오버레이: 빨강 ROI/비율 + 아루코 박스 + 상태 → /obstacle/debug
    # ------------------------------------------------------------------ #
    def _publish_debug(self, frame, source_msg):
        overlay = frame.copy()
        if self.red_roi_box is not None:
            x0, y0, x1, y1 = self.red_roi_box
            cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 255), 1)
        for (x1, y1, x2, y2, area, mid) in self.aruco_boxes:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(overlay, f'id{mid} {area:.3f}', (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

        # aruco=필터통과 수, raw=필터전 검출ID(오독 진단: 마커는 보이나 ID가 틀림)
        text = (f'{self.mission_state} red={self.red_ratio:.3f} '
                f'lanes={int(self.lanes_visible)} aruco={len(self.aruco_boxes)} '
                f'raw={self.aruco_raw_ids}')
        cv2.putText(overlay, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)

        ok, encoded = cv2.imencode(
            '.jpg', overlay, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return
        out = CompressedImage()
        out.header.stamp = source_msg.header.stamp
        out.header.frame_id = 'obstacle_debug'
        out.format = 'jpeg'
        out.data = encoded.tobytes()
        self.debug_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = DecisionObstacleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
