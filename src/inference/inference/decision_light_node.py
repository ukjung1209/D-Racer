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
    """차선(/lane/state)과 신호등(/object/detections green/red)으로 주행 명령(/control)을 낸다.

    ── 팀 병렬개발용 파생 노드 ──
    차선추종 PD 코어는 decision_node와 100% 동일하게 유지하고, 갈림길 대신
    '신호등 출발+정지' 상태머신만 얹었다. 나중에 decision_node로 merge할 때는
    이 상태머신을 그대로 옮기고 control_loop의 throttle 우선순위만 합치면 된다.

    신호등 상태머신 (조향은 항상 차선추종, throttle만 상태로 제어):
      HOLD  : 초기 정지. green을 votes만큼 연속 검출하면 출발(→GO).
              start_require_green=False면 처음부터 GO로 시작(신호 없이 주행 테스트).
      GO    : 차선추종 주행. red가 충분히 크게(=정지선 코앞, area≥stop_area_trigger)
              votes만큼 보이면 정지(→STOP).
      STOP  : 정지선 정지(throttle 0). green을 votes만큼 다시 보면 재출발(→GO).
    갈림길/동적장애물 미션은 이 노드 범위 밖(다른 파생 노드/merge에서 담당).
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
        self.declare_parameter('steer_kd', 0.3)
        self.declare_parameter('steer_ka', 0.0)
        self.declare_parameter('steering_sign', -1.0)
        self.declare_parameter('steer_trim', float('nan'))
        self.declare_parameter('base_throttle', 0.15)
        self.declare_parameter('lane_timeout_sec', 0.5)

        # --- 신호등 미션 파라미터 (live 튜닝 가능) ---
        self.declare_parameter('enable_light_mission', True)
        self.declare_parameter('start_require_green', True)  # 출발 시 green 신호 대기
        self.declare_parameter('light_conf_min', 0.5)        # 신호등 최소 confidence
        self.declare_parameter('green_votes_needed', 3)      # 출발/재출발 확정 표수
        self.declare_parameter('red_votes_needed', 3)        # 정지 확정 표수
        self.declare_parameter('stop_area_trigger', 0.04)    # red 이보다 크면 정지선 코앞
        self.declare_parameter('go_throttle', float('nan'))  # NaN이면 base_throttle 사용

        lane_topic = str(self.get_parameter('lane_topic').value)
        detection_topic = str(self.get_parameter('detection_topic').value)
        control_topic = str(self.get_parameter('control_topic').value)
        command_hz = float(self.get_parameter('command_hz').value)

        self.vehicle_config = self._load_vehicle_config(
            os.path.expanduser(str(self.get_parameter('vehicle_config_file').value))
        )

        self.lane_state = None
        self.lane_stamp = None
        self.detections = None
        self.prev_offset = 0.0

        # --- 신호등 상태머신 상태 ---
        # start_require_green=False면 바로 GO로 시작
        self.light_state = 'HOLD' if bool(
            self.get_parameter('start_require_green').value) else 'GO'
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
        self.detections = msg
        if not bool(self.get_parameter('enable_light_mission').value):
            return
        self._update_light(msg)

    # ------------------------------------------------------------------ #
    #  신호등: green/red 검출로 출발/정지 상태 전이
    # ------------------------------------------------------------------ #
    def _pick_light(self, msg: DetectionArray, class_name):
        """해당 색의 신호등 중 confidence를 통과한 가장 큰(가까운) 것을 고른다."""
        conf_min = float(self.get_parameter('light_conf_min').value)
        best = None
        for det in msg.detections:
            if det.class_name != class_name:
                continue
            if det.confidence < conf_min:
                continue
            if best is None or det.area_ratio > best.area_ratio:
                best = det
        return best

    def _update_light(self, msg: DetectionArray):
        green = self._pick_light(msg, 'green')
        red = self._pick_light(msg, 'red')

        if self.light_state in ('HOLD', 'STOP'):
            # green을 연속으로 보면 출발/재출발
            if green is not None:
                self.vote_green += 1
            else:
                self.vote_green = 0
            self.vote_red = 0
            if self.vote_green >= int(self.get_parameter('green_votes_needed').value):
                self.get_logger().info(
                    f'light: {self.light_state} → GO (green)')
                self.light_state = 'GO'
                self.vote_green = 0

        elif self.light_state == 'GO':
            # red가 충분히 크게(정지선 코앞) 연속으로 보이면 정지
            trigger = float(self.get_parameter('stop_area_trigger').value)
            if red is not None and red.area_ratio >= trigger:
                self.vote_red += 1
            else:
                self.vote_red = 0
            self.vote_green = 0
            if self.vote_red >= int(self.get_parameter('red_votes_needed').value):
                self.get_logger().info('light: GO → STOP (red)')
                self.light_state = 'STOP'
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
            # HOLD/STOP이면 조향은 유지하되 정지, GO면 주행
            go_throttle = float(self.get_parameter('go_throttle').value)
            if math.isnan(go_throttle):
                go_throttle = float(self.get_parameter('base_throttle').value)
            control.throttle = go_throttle if going else 0.0
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
