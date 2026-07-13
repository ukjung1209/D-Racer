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


class DecisionObstacleNode(Node):
    """차선(/lane/state)과 동적 장애물(/object/detections의 아루코 'obstacle')로 주행한다.

    ── 팀 병렬개발용 파생 노드 ──
    차선추종 PD 코어는 decision_node와 100% 동일하게 유지하고, 갈림길 대신
    '동적 장애물 정지-대기 재출발' 상태머신만 얹었다. 나중에 decision_node로
    merge할 때는 이 상태머신을 그대로 옮기고 control_loop의 throttle 우선순위만
    합치면 된다.

    동적 장애물은 아루코 마커로 나타나며, object_node(enable_aruco=True)가
    /object/detections에 class_name='obstacle'(area_ratio=근접도)로 함께 발행한다.

    장애물 상태머신 (조향은 항상 차선추종, throttle만 상태로 제어):
      CRUISE     : 차선추종 주행. 전방 마커가 충분히 크게(area≥obstacle_area_trigger)
                   votes만큼 보이면 정지(→STOP_WAIT).
      STOP_WAIT  : 정지(throttle 0). 마커가 clear_time_sec 이상 안 보이면
                   (=지나감) 재출발(→CRUISE).
    신호등/갈림길 미션은 이 노드 범위 밖(다른 파생 노드/merge에서 담당).
    """

    def __init__(self):
        super().__init__('decision_obstacle_node')

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

        # --- 동적 장애물 미션 파라미터 (live 튜닝 가능) ---
        self.declare_parameter('enable_obstacle_mission', True)
        self.declare_parameter('obstacle_area_trigger', 0.02)  # 마커 이보다 크면 정지
        self.declare_parameter('obstacle_votes_needed', 3)     # 정지 확정 표수
        self.declare_parameter('clear_time_sec', 1.0)          # 마커 사라짐 판정 시간

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

        # --- 장애물 상태머신 상태 ---
        self.obstacle_state = 'CRUISE'   # CRUISE | STOP_WAIT
        self.vote_obstacle = 0
        self.last_obstacle_time = None   # 마커를 마지막으로 본 시각 (Time)

        self.create_subscription(LaneState, lane_topic, self.lane_callback, 10)
        self.create_subscription(
            DetectionArray, detection_topic, self.detection_callback, 10)
        self.control_pub = self.create_publisher(Control, control_topic, 10)

        self.timer = self.create_timer(1.0 / command_hz, self.control_loop)

        self.get_logger().info(
            f'decision_obstacle_node started: lane_topic={lane_topic}, '
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
        if not bool(self.get_parameter('enable_obstacle_mission').value):
            return
        self._update_obstacle(msg)

    # ------------------------------------------------------------------ #
    #  동적 장애물: 아루코 마커 검출로 정지 트리거
    # ------------------------------------------------------------------ #
    def _pick_obstacle(self, msg: DetectionArray):
        """'obstacle'(아루코) 중 가장 큰(가까운) 것을 고른다."""
        best = None
        for det in msg.detections:
            if det.class_name != 'obstacle':
                continue
            if best is None or det.area_ratio > best.area_ratio:
                best = det
        return best

    def _update_obstacle(self, msg: DetectionArray):
        best = self._pick_obstacle(msg)
        if best is None:
            self.vote_obstacle = 0
            return  # 이번 프레임에 마커 없음 (STOP_WAIT 해제는 control_loop에서 시간처리)

        # 마커가 보이면(크기 무관) 마지막 관측 시각 갱신 → clear 타임아웃 기준
        self.last_obstacle_time = self.get_clock().now()

        if self.obstacle_state == 'CRUISE':
            trigger = float(self.get_parameter('obstacle_area_trigger').value)
            if best.area_ratio >= trigger:
                self.vote_obstacle += 1
            else:
                self.vote_obstacle = 0
            if self.vote_obstacle >= int(
                    self.get_parameter('obstacle_votes_needed').value):
                self.get_logger().info(
                    f'obstacle: CRUISE → STOP_WAIT (area={best.area_ratio:.3f}, '
                    f'id={best.class_id})')
                self.obstacle_state = 'STOP_WAIT'
                self.vote_obstacle = 0

    def _update_obstacle_state(self):
        """시간 기반 전이: 마커가 clear_time_sec 이상 안 보이면 재출발."""
        if self.obstacle_state != 'STOP_WAIT' or self.last_obstacle_time is None:
            return
        gone = (self.get_clock().now()
                - self.last_obstacle_time).nanoseconds / 1e9
        if gone >= float(self.get_parameter('clear_time_sec').value):
            self.get_logger().info('obstacle: STOP_WAIT → CRUISE (cleared)')
            self.obstacle_state = 'CRUISE'
            self.vote_obstacle = 0

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
        if bool(self.get_parameter('enable_obstacle_mission').value):
            self._update_obstacle_state()

        trim = float(self.get_parameter('steer_trim').value)
        if math.isnan(trim):
            trim = float(self.vehicle_config.get('STEER_TRIM', 0.0))

        control = Control()
        control.header.stamp = self.get_clock().now().to_msg()
        control.header.frame_id = 'decision'

        pd_steer = self._lane_pd_steer(trim)
        stopped = self.obstacle_state == 'STOP_WAIT'

        if pd_steer is not None:
            control.steering = float(clamp(pd_steer, -1.0, 1.0))
            # 장애물 앞이면 조향은 유지하되 정지, 아니면 주행
            control.throttle = 0.0 if stopped else float(
                self.get_parameter('base_throttle').value)
        else:
            # 차선을 잃으면 안전 정지
            self.prev_offset = 0.0
            control.steering = float(clamp(trim, -1.0, 1.0))
            control.throttle = 0.0

        self.control_pub.publish(control)


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
