import math
import os
from pathlib import Path

import rclpy
from rclpy.node import Node

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

    차선추종(lane_state offset/angle PD)에 더해, 좌/우 갈림길 미션 상태머신을 돌린다.
    트랙은 Y자(분기점 정면 안쪽에 좌/우 표지판)라, 표지판이 멀리 작게 보일 때
    YOLO left/right를 여러 프레임 투표로 방향을 확정(CRUISE→ARMED)하고, 접근하며
    표지판이 충분히 커지면(=분기점 코앞) 확정 방향으로 조향 bias를 줘 갈래로 진입
    (ARMED→FORK), 일정 시간 뒤 차선추종으로 복귀(FORK→CRUISE)한다.
    신호등/동적장애물 미션은 아직 TODO.
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

        # --- 좌/우 갈림길 미션 파라미터 (live 튜닝 가능) ---
        self.declare_parameter('enable_fork_mission', True)
        self.declare_parameter('sign_conf_min', 0.5)      # 표지판 최소 confidence
        self.declare_parameter('sign_area_min', 0.01)     # 방향 판독 시작(멀리 작게 보임) 최소 area_ratio
        self.declare_parameter('sign_votes_needed', 5)    # 방향 확정에 필요한 연속 검출 표수
        self.declare_parameter('fork_area_trigger', 0.06)  # 이보다 커지면 분기점 코앞 → bias 시작
        self.declare_parameter('arm_timeout_sec', 3.0)    # ARMED에서 표지판 이 시간 놓치면 안전 리셋
        # left일 때 +fork_bias, right일 때 -fork_bias를 최종 steering에 더한다.
        # 실제 좌/우가 반대로 꺾이면 fork_bias를 음수로 주면 된다.
        self.declare_parameter('fork_bias', 0.5)
        self.declare_parameter('fork_duration_sec', 2.0)  # bias 주행 유지 시간
        self.declare_parameter('fork_throttle', 0.13)     # 갈림길 통과 시 저속

        lane_topic = str(self.get_parameter('lane_topic').value)
        detection_topic = str(self.get_parameter('detection_topic').value)
        control_topic = str(self.get_parameter('control_topic').value)
        command_hz = float(self.get_parameter('command_hz').value)

        self.vehicle_config = self._load_vehicle_config(
            os.path.expanduser(str(self.get_parameter('vehicle_config_file').value))
        )

        self.lane_state = None
        self.lane_stamp = None       # 마지막 lane_state 수신 시각 (Time)
        self.detections = None
        self.prev_offset = 0.0

        # --- 갈림길 상태머신 상태 ---
        self.fork_state = 'CRUISE'   # CRUISE | ARMED | FORK
        self.latched_dir = 0         # +1 = left, -1 = right
        self.vote_left = 0
        self.vote_right = 0
        self.last_sign_time = None   # 표지판을 마지막으로 본 시각 (Time)
        self.fork_start_time = None  # FORK 진입 시각 (Time)

        self.create_subscription(LaneState, lane_topic, self.lane_callback, 10)
        self.create_subscription(
            DetectionArray, detection_topic, self.detection_callback, 10)
        self.control_pub = self.create_publisher(Control, control_topic, 10)

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
        self._update_fork_vote(msg)

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

    def _update_fork_vote(self, msg: DetectionArray):
        best = self._pick_sign(msg)
        if best is None:
            return  # 이번 프레임에 유효 표지판 없음 (ARMED 타임아웃은 control_loop에서 처리)

        self.last_sign_time = self.get_clock().now()
        direction = 1 if best.class_name == 'left' else -1

        if self.fork_state == 'CRUISE':
            # 같은 방향은 표를 쌓고 반대 방향 표는 리셋 → 오검출 흔들림 방지
            if direction == 1:
                self.vote_left += 1
                self.vote_right = 0
            else:
                self.vote_right += 1
                self.vote_left = 0
            votes_needed = int(self.get_parameter('sign_votes_needed').value)
            if max(self.vote_left, self.vote_right) >= votes_needed:
                self.latched_dir = direction
                self.fork_state = 'ARMED'
                self.get_logger().info(
                    f'fork: direction latched = {"LEFT" if direction == 1 else "RIGHT"}')

        elif self.fork_state == 'ARMED':
            # 확정 방향의 표지판이 충분히 커지면(분기점 코앞) bias 주행 시작
            trigger = float(self.get_parameter('fork_area_trigger').value)
            if direction == self.latched_dir and best.area_ratio >= trigger:
                self.fork_state = 'FORK'
                self.fork_start_time = self.get_clock().now()
                self.get_logger().info(
                    f'fork: entering FORK (area={best.area_ratio:.3f})')

    def _reset_fork(self):
        self.fork_state = 'CRUISE'
        self.latched_dir = 0
        self.vote_left = 0
        self.vote_right = 0
        self.last_sign_time = None
        self.fork_start_time = None

    def _lane_is_fresh(self):
        if self.lane_state is None or self.lane_stamp is None:
            return False
        timeout = float(self.get_parameter('lane_timeout_sec').value)
        age = (self.get_clock().now() - self.lane_stamp).nanoseconds / 1e9
        return age <= timeout

    def _update_fork_state(self):
        """시간 기반 전이: FORK 종료(타이머), ARMED 표지판 놓침(안전 리셋)."""
        now = self.get_clock().now()
        if self.fork_state == 'FORK':
            elapsed = (now - self.fork_start_time).nanoseconds / 1e9
            if elapsed >= float(self.get_parameter('fork_duration_sec').value):
                self.get_logger().info('fork: FORK done → CRUISE')
                self._reset_fork()
        elif self.fork_state == 'ARMED' and self.last_sign_time is not None:
            gone = (now - self.last_sign_time).nanoseconds / 1e9
            if gone >= float(self.get_parameter('arm_timeout_sec').value):
                self.get_logger().warning('fork: ARMED lost sign → reset CRUISE')
                self._reset_fork()

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
        # TODO: 나머지 미션 상태머신 (출발신호 / 동적장애물 / 도착)
        if bool(self.get_parameter('enable_fork_mission').value):
            self._update_fork_state()

        trim = float(self.get_parameter('steer_trim').value)
        if math.isnan(trim):
            trim = float(self.vehicle_config.get('STEER_TRIM', 0.0))

        control = Control()
        control.header.stamp = self.get_clock().now().to_msg()
        control.header.frame_id = 'decision'

        pd_steer = self._lane_pd_steer(trim)

        if self.fork_state == 'FORK':
            # 갈림길 통과: 확정 방향으로 bias를 준다. 차선이 잠깐 안 잡혀도 주행 유지.
            fork_bias = float(self.get_parameter('fork_bias').value)
            base = pd_steer if pd_steer is not None else trim
            steer = base + self.latched_dir * fork_bias
            control.steering = float(clamp(steer, -1.0, 1.0))
            control.throttle = float(self.get_parameter('fork_throttle').value)
        elif pd_steer is not None:
            # 평상시 차선추종 (CRUISE / ARMED 모두 표지판에 다가가며 직진 추종)
            control.steering = float(clamp(pd_steer, -1.0, 1.0))
            control.throttle = float(self.get_parameter('base_throttle').value)
        else:
            # 차선을 잃으면 안전 정지
            self.prev_offset = 0.0
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
