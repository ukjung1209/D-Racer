import math
import os
from pathlib import Path

import rclpy
from rclpy.node import Node

from std_msgs.msg import Int8
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


class DecisionArrowNode(Node):
    """차선(/lane/state)과 물체(/object/detections)를 종합해 최종 주행 명령(/control)을 낸다.

    기본은 lane_state offset/angle 기반 PD 차선추종 + 곡률 가변속도.
    갈림길은 단순화: YOLO left/right 표지판이 보이면(conf/area 충족) 즉시 그 방향으로
    latch 해서 lane_node에 branch_hint(±1)를 보내 그 라인만 hugging 하게 한다
    (반대 라인 무시, 그 라인 ±bias를 중점으로 폐루프 추종). 표지판이 보이고 두 차선도
    확실히 보일 때(lane conf 충족) 갈림길당 한 번 sign_stop_sec만큼 정지 후 출발
    (stop-and-go)한다. 표지판을 hug_hold_sec
    이상 놓치면 hugging 해제 → 평소 양쪽 차선 추종으로 복귀. 투표/ARMED/area 트리거
    같은 방어 로직은 없다. 신호등/동적장애물 미션은 아직 TODO.
    """

    def __init__(self):
        super().__init__('decision_arrow_node')

        self.declare_parameter('lane_topic', 'lane/state')
        self.declare_parameter('detection_topic', 'object/detections')
        self.declare_parameter('control_topic', '/control')
        self.declare_parameter('command_hz', 20.0)
        self.declare_parameter('vehicle_config_file', get_default_vehicle_config_path())

        # --- 차선추종 PD 제어 파라미터 (live 튜닝 가능) ---
        self.declare_parameter('steer_kp', 0.8)          # offset 비례
        self.declare_parameter('steer_kd', 0.3)          # offset 변화율
        self.declare_parameter('steer_ka', 0.0)          # angle 전방주시 피드포워드
        self.declare_parameter('steering_sign', -1.0)    # 방향 반대면 부호 뒤집기
        self.declare_parameter('steer_trim', float('nan'))  # NaN이면 vehicle_config STEER_TRIM
        self.declare_parameter('base_throttle', 0.15)
        self.declare_parameter('lane_timeout_sec', 0.5)  # 이 시간 넘게 lane 없으면 정지

        # --- 곡률 기반 가변속도 (직선 빠르게 / 코너 미리 감속) ---
        # target = base × (1 − speed_ka·|angle| − speed_ko·|offset|), [min_throttle, base]로 클램프.
        # angle(먼 밴드 곡률)은 코너 진입 전 예측 감속, offset(현재 이탈)은 보정. steer가 아니라
        # angle을 쓰는 이유: S자 변곡점에서 steer는 0을 지나며 속도가 튀지만 angle은 안 그렇다.
        self.declare_parameter('speed_ka', 0.6)          # |angle| 예측 감속 gain
        self.declare_parameter('speed_ko', 0.4)          # |offset| 보정 감속 gain
        self.declare_parameter('min_throttle', 0.10)     # 감속 하한(너무 느려 멈추지 않게)
        # 슬루 제한: 가속은 초당 이 값까지만 올림(완만), 감속은 즉시. S자에서 순간 가속 튐 방지.
        self.declare_parameter('throttle_accel_rate', 0.5)

        # --- 좌/우 갈림길 파라미터 (단순화: 표지판 보이면 즉시 그 라인 hugging) ---
        self.declare_parameter('enable_fork_mission', True)
        self.declare_parameter('sign_conf_min', 0.5)      # 표지판 최소 confidence
        self.declare_parameter('sign_area_min', 0.01)     # 이 크기 이상이어야 방향 인정(멀면 무시)
        self.declare_parameter('branch_hint_topic', 'lane/branch_hint')  # lane_node에 hugging 힌트 발행
        # 표지판을 이 시간 이상 못 보면 hugging 해제 → 평소 양쪽 차선 추종 복귀
        self.declare_parameter('hug_hold_sec', 1.0)
        # 표지판 인식 시 이 시간만큼 정지 후 출발 (stop-and-go). 0이면 정지 안 함.
        self.declare_parameter('sign_stop_sec', 0.7)
        # 정지는 '두 차선이 확실히 보일 때'만 (lane confidence 이 값 이상). 불안정한 순간 정지 방지.
        self.declare_parameter('stop_lane_conf_min', 0.5)

        lane_topic = str(self.get_parameter('lane_topic').value)
        detection_topic = str(self.get_parameter('detection_topic').value)
        control_topic = str(self.get_parameter('control_topic').value)
        command_hz = float(self.get_parameter('command_hz').value)
        self.command_dt = 1.0 / command_hz   # 슬루 제한용 tick 간격(초)

        self.vehicle_config = self._load_vehicle_config(
            os.path.expanduser(str(self.get_parameter('vehicle_config_file').value))
        )

        self.lane_state = None
        self.lane_stamp = None       # 마지막 lane_state 수신 시각 (Time)
        self.detections = None
        self.prev_offset = 0.0
        self.prev_throttle = 0.0     # 직전 tick throttle (가속 슬루 제한 기준)

        # --- 갈림길 상태 (단순화) ---
        self.latched_dir = 0         # +1 = left, -1 = right, 0 = 평소 양쪽 추종
        self.last_sign_time = None   # 표지판을 마지막으로 본 시각 (Time)
        self.stop_start = None       # 정지 시작 시각 (Time), None=정지 안 함
        self.stopped_done = False    # 이번 갈림길 정지 완료 여부 (한 갈림길당 1회)

        branch_hint_topic = str(self.get_parameter('branch_hint_topic').value)

        self.create_subscription(LaneState, lane_topic, self.lane_callback, 10)
        self.create_subscription(
            DetectionArray, detection_topic, self.detection_callback, 10)
        self.control_pub = self.create_publisher(Control, control_topic, 10)
        # FORK 중 lane_node가 확정 방향 라인만 hugging 하도록 힌트를 준다
        self.branch_hint_pub = self.create_publisher(Int8, branch_hint_topic, 10)

        self.timer = self.create_timer(1.0 / command_hz, self.control_loop)

        self.get_logger().info(
            f'decision_arrow_node started: lane_topic={lane_topic}, '
            f'detection_topic={detection_topic}, control_topic={control_topic}, '
            f'command_hz={command_hz}'
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

    def lane_callback(self, msg: LaneState):
        self.lane_state = msg
        self.lane_stamp = self.get_clock().now()

    def detection_callback(self, msg: DetectionArray):
        self.detections = msg
        if not bool(self.get_parameter('enable_fork_mission').value):
            return
        self._update_sign_latch(msg)

    # ------------------------------------------------------------------ #
    #  갈림길: 표지판 검출로 방향 투표/확정 및 FORK 트리거
    # ------------------------------------------------------------------ #
    def _pick_sign(self, msg: DetectionArray):
        """left/right 중 confidence·크기 조건을 통과한 가장 큰(가까운) 표지판을 고른다."""
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
                self.stopped_done = False   # 새 갈림길 진입 → 정지 재무장(조건은 control_loop에서)
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
            self.stopped_done = False   # 다음 갈림길에서 다시 정지하도록 리셋
            self.stop_start = None

    def _lane_is_fresh(self):
        if self.lane_state is None or self.lane_stamp is None:
            return False
        timeout = float(self.get_parameter('lane_timeout_sec').value)
        age = (self.get_clock().now() - self.lane_stamp).nanoseconds / 1e9
        return age <= timeout

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
        """가속은 슬루 제한(완만), 감속은 즉시. S자 변곡점 순간 가속 튐 방지."""
        max_up = float(self.get_parameter('throttle_accel_rate').value) * self.command_dt
        if target > self.prev_throttle + max_up:
            out = self.prev_throttle + max_up   # 가속은 조금씩만
        else:
            out = target                        # 감속(및 소폭 증가)은 즉시
        self.prev_throttle = out
        return out

    def _lane_pd_steer(self, trim):
        """차선추종 PD 조향값. 차선이 없으면 None."""
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

    def control_loop(self):
        # TODO: 나머지 미션 (출발신호 / 동적장애물 / 도착)
        if bool(self.get_parameter('enable_fork_mission').value):
            self._release_hug_if_sign_lost()

        # latched_dir(±1)이면 lane_node가 그 방향 라인만 hugging (0=평소 양쪽 중심)
        self.branch_hint_pub.publish(Int8(data=int(self.latched_dir)))

        trim = float(self.get_parameter('steer_trim').value)
        if math.isnan(trim):
            trim = float(self.vehicle_config.get('STEER_TRIM', 0.0))

        control = Control()
        control.header.stamp = self.get_clock().now().to_msg()
        control.header.frame_id = 'decision'

        # 정지 트리거: 표지판 인식(hugging)됐고 + 두 차선이 확실히 보일 때(conf 충족)
        # 이번 갈림길에 한 번 정지. 불안정한 순간엔 안 서고 조건 충족될 때까지 기다린다.
        if (self.latched_dir != 0 and not self.stopped_done and self.stop_start is None
                and float(self.get_parameter('sign_stop_sec').value) > 0.0
                and self._lane_is_fresh() and self.lane_state.detected
                and self.lane_state.confidence
                >= float(self.get_parameter('stop_lane_conf_min').value)):
            self.stop_start = self.get_clock().now()

        # 정지 실행 (stop-and-go)
        stopping = False
        if self.stop_start is not None:
            elapsed = (self.get_clock().now() - self.stop_start).nanoseconds / 1e9
            if elapsed < float(self.get_parameter('sign_stop_sec').value):
                stopping = True
            else:
                self.stop_start = None       # 정지 종료 → 출발
                self.stopped_done = True      # 이번 갈림길 정지 완료

        pd_steer = self._lane_pd_steer(trim)

        if stopping:
            # 정지 중에도 조향은 유지해 갈래 방향을 미리 겨눈다. 재출발은 0부터 슬루 상승.
            control.steering = float(clamp(
                pd_steer if pd_steer is not None else trim, -1.0, 1.0))
            control.throttle = 0.0
            self.prev_throttle = 0.0
        elif pd_steer is not None:
            # 차선추종 PD. hugging 중엔 lane_node가 이미 그 라인을 +bias 중점으로 잡아주므로
            # 여기선 bias 없이 같은 PD를 그대로 쓴다.
            control.steering = float(clamp(pd_steer, -1.0, 1.0))
            # 속도는 항상 곡률 기반 가변속도 (hugging 여부와 무관)
            target_throttle = self._curve_throttle(
                float(self.get_parameter('base_throttle').value))
            control.throttle = float(self._slew_throttle(target_throttle))
        else:
            # 차선을 잃으면 안전 정지
            self.prev_offset = 0.0
            self.prev_throttle = 0.0   # 재출발 시 0부터 슬루 상승
            control.steering = float(clamp(trim, -1.0, 1.0))
            control.throttle = 0.0

        self.control_pub.publish(control)


def main(args=None):
    rclpy.init(args=args)
    node = DecisionArrowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
