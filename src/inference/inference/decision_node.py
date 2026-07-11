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


class DecisionNode(Node):
    """차선(/lane/state)과 물체(/object/detections)를 종합해 최종 주행 명령(/control)을 낸다.

    지금은 차선추종만 한다: lane_state의 offset/angle을 PD 제어해 steering을 만들고
    차선이 보이면 throttle을 준다. 신호등/표지판/장애물 미션 상태머신은 아직 TODO.
    """

    def __init__(self):
        super().__init__('decision_node')

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

        self.create_subscription(LaneState, lane_topic, self.lane_callback, 10)
        self.create_subscription(
            DetectionArray, detection_topic, self.detection_callback, 10)
        self.control_pub = self.create_publisher(Control, control_topic, 10)

        self.timer = self.create_timer(1.0 / command_hz, self.control_loop)

        self.get_logger().info(
            f'decision_node started: lane_topic={lane_topic}, '
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

    def _lane_is_fresh(self):
        if self.lane_state is None or self.lane_stamp is None:
            return False
        timeout = float(self.get_parameter('lane_timeout_sec').value)
        age = (self.get_clock().now() - self.lane_stamp).nanoseconds / 1e9
        return age <= timeout

    def control_loop(self):
        # TODO: 미션 상태머신 (출발신호 -> S자 -> 갈림길 -> 장애물 -> 도착)
        #       detections(신호등/표지판/ArUco)로 stop/branch를 오버라이드할 자리.
        trim = float(self.get_parameter('steer_trim').value)
        if math.isnan(trim):
            trim = float(self.vehicle_config.get('STEER_TRIM', 0.0))

        control = Control()
        control.header.stamp = self.get_clock().now().to_msg()
        control.header.frame_id = 'decision'

        # 차선이 신선하고 검출됐을 때만 주행, 아니면 안전 정지
        if self._lane_is_fresh() and self.lane_state.detected:
            kp = float(self.get_parameter('steer_kp').value)
            kd = float(self.get_parameter('steer_kd').value)
            ka = float(self.get_parameter('steer_ka').value)
            sign = float(self.get_parameter('steering_sign').value)
            throttle = float(self.get_parameter('base_throttle').value)

            offset = self.lane_state.offset
            angle = self.lane_state.angle
            derivative = offset - self.prev_offset
            self.prev_offset = offset

            steer = sign * (kp * offset + kd * derivative + ka * angle) + trim
            control.steering = float(clamp(steer, -1.0, 1.0))
            control.throttle = float(throttle)
        else:
            self.prev_offset = 0.0
            control.steering = float(clamp(trim, -1.0, 1.0))
            control.throttle = 0.0

        self.control_pub.publish(control)


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
