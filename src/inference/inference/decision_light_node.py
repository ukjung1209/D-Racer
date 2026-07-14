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


class DecisionLightNode(Node):
    """차선(/lane/state)추종 + 신호등(/object/detections) 출발/정지 주행.

    ── 신호등 미션 로직 (아주 단순) ──
    조향은 항상 /lane/state PD 차선추종, throttle만 신호등 상태로 켜고 끈다.
      STOPPED : 정지. 초록불(green)을 green_votes_needed프레임 연속 보면 출발(→GO).
      GO      : 차선추종 주행. 빨간불(red)을 red_votes_needed프레임 연속 보면 정지(→STOPPED).
    → 처음엔 STOPPED로 시작(가만히 있다가 초록불 3프레임 보면 출발).
    → STOPPED에서 다시 초록불 3프레임 보면 재출발한다.

    초록/빨강은 object_node의 YOLO(best_320.onnx)가 내는 /object/detections의
    class_name('green'/'red')으로 판단한다. 크기(area) 조건 없이 '연속 프레임 수'만 본다.
    """

    def __init__(self):
        super().__init__('decision_light_node')

        self.declare_parameter('lane_topic', 'lane/state')
        self.declare_parameter('detection_topic', 'object/detections')
        self.declare_parameter('control_topic', '/control')
        self.declare_parameter('command_hz', 20.0)
        self.declare_parameter('vehicle_config_file', get_default_vehicle_config_path())

        # --- 차선추종 PD 제어 파라미터 (decision_node와 동일) ---
        self.declare_parameter('steer_kp', 0.8)
        self.declare_parameter('steer_kd', 0.4)
        self.declare_parameter('steer_ka', 0.0)
        self.declare_parameter('steering_sign', -1.0)
        self.declare_parameter('steer_trim', float('nan'))
        self.declare_parameter('base_throttle', 0.20)
        self.declare_parameter('lane_timeout_sec', 0.5)

        # --- 신호등 미션 파라미터 (실시간 튜닝 가능) ---
        self.declare_parameter('enable_light_mission', True)
        self.declare_parameter('light_conf_min', 0.5)     # 신호등 최소 confidence
        self.declare_parameter('green_votes_needed', 3)   # 초록불 이만큼 연속 보면 출발
        self.declare_parameter('red_votes_needed', 3)     # 빨간불 이만큼 연속 보면 정지

        lane_topic = str(self.get_parameter('lane_topic').value)
        detection_topic = str(self.get_parameter('detection_topic').value)
        control_topic = str(self.get_parameter('control_topic').value)
        command_hz = float(self.get_parameter('command_hz').value)

        self.vehicle_config = self._load_vehicle_config(
            os.path.expanduser(str(self.get_parameter('vehicle_config_file').value))
        )

        self.lane_state = None
        self.lane_stamp = None
        self.prev_offset = 0.0

        # --- 신호등 상태머신 상태 ---
        self.light_state = 'STOPPED'   # STOPPED | GO
        self.vote_green = 0
        self.vote_red = 0

        self.create_subscription(LaneState, lane_topic, self.lane_callback, 10)
        self.create_subscription(
            DetectionArray, detection_topic, self.detection_callback, 10)
        self.control_pub = self.create_publisher(Control, control_topic, 10)

        self.timer = self.create_timer(1.0 / command_hz, self.control_loop)

        self.get_logger().info(
            f'decision_light_node started: lane_topic={lane_topic}, '
            f'detection_topic={detection_topic}, control_topic={control_topic}, '
            f'command_hz={command_hz}, start_state={self.light_state}'
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
        if not bool(self.get_parameter('enable_light_mission').value):
            return
        self._update_light(msg)

    # ------------------------------------------------------------------ #
    #  신호등: green/red 연속 프레임 수로 출발/정지 전이
    # ------------------------------------------------------------------ #
    def _has_light(self, msg: DetectionArray, class_name):
        """해당 색 신호등이 confidence 넘겨 검출됐는지."""
        conf_min = float(self.get_parameter('light_conf_min').value)
        for det in msg.detections:
            if det.class_name == class_name and det.confidence >= conf_min:
                return True
        return False

    def _update_light(self, msg: DetectionArray):
        green = self._has_light(msg, 'green')
        red = self._has_light(msg, 'red')

        if self.light_state == 'STOPPED':
            # 초록불을 연속으로 보면 출발
            self.vote_green = self.vote_green + 1 if green else 0
            self.vote_red = 0
            if self.vote_green >= int(self.get_parameter('green_votes_needed').value):
                self.get_logger().info('light: STOPPED → GO (green)')
                self.light_state = 'GO'
                self.vote_green = 0

        elif self.light_state == 'GO':
            # 빨간불을 연속으로 보면 정지
            self.vote_red = self.vote_red + 1 if red else 0
            self.vote_green = 0
            if self.vote_red >= int(self.get_parameter('red_votes_needed').value):
                self.get_logger().info('light: GO → STOPPED (red)')
                self.light_state = 'STOPPED'
                self.vote_red = 0

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

    def control_loop(self):
        trim = float(self.get_parameter('steer_trim').value)
        if math.isnan(trim):
            trim = float(self.vehicle_config.get('STEER_TRIM', 0.0))

        control = Control()
        control.header.stamp = self.get_clock().now().to_msg()
        control.header.frame_id = 'decision'

        pd_steer = self._lane_pd_steer(trim)
        going = self.light_state == 'GO'

        if pd_steer is not None:
            control.steering = float(clamp(pd_steer, -1.0, 1.0))
            # GO면 주행, STOPPED면 조향은 유지하되 정지
            control.throttle = float(self.get_parameter('base_throttle').value) if going else 0.0
        else:
            # 차선을 잃으면 안전 정지
            self.prev_offset = 0.0
            control.steering = float(clamp(trim, -1.0, 1.0))
            control.throttle = 0.0

        self.control_pub.publish(control)


def main(args=None):
    rclpy.init(args=args)
    node = DecisionLightNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
