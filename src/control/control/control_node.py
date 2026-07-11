import os
from pathlib import Path

import rclpy
from rclpy.node import Node
import yaml

from control_msgs.msg import Control
from joystick_msgs.msg import Joystick
from topst_utils.d3racer import D3Racer


def get_default_vehicle_config_path():
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'src' / 'config' / 'vehicle_config.yaml'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/D-Racer/src/config/vehicle_config.yaml'


class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')

        # ROS parameters
        self.declare_parameter('i2c_bus', 3)
        self.declare_parameter('pca9685_addr', 0x40)
        self.declare_parameter('steering_channel', 0)
        self.declare_parameter('throttle_channel', 1)
        self.declare_parameter('vehicle_config_file', get_default_vehicle_config_path())
        self.declare_parameter('use_joystick_control', False)
        self.declare_parameter('joystick_topic', 'joystick')
        self.declare_parameter('control_topic', '/control')
        self.declare_parameter('command_hz', 10.0)
        # 후진 세기. 조이스틱 후진 값은 accel_ratio 때문에 매우 약해(-0.17 수준)
        # 이 ESC가 후진으로 인식하지 못한다. 스틱을 아래로 내리면 이 세기(양수, 0~1)로
        # 후진 신호를 내보낸다. 실측상 0.5에서 가장 안정적으로 후진이 걸렸다.
        self.declare_parameter('reverse_speed', 0.5)
        # 후진 더블탭 자동화. 이 ESC는 후진 진입 시 "후진(브레이크) -> 중립 -> 후진"
        # 순서를 거쳐야만 실제로 후진한다(손으로 두 번 내리던 동작). 스틱을 내리면
        # 아래 시간만큼 후진 탭 -> 중립을 자동으로 넣은 뒤 후진을 유지한다.
        # command_hz=10이면 타이머가 0.1초 간격이라, 이 값이 0.1이면 각 단계가 1사이클만
        # 나가 타이밍이 밀리면 중립이 건너뛰어져 arming이 불안정하다. 0.2 이상 권장.
        self.declare_parameter('reverse_tap_time', 0.2)     # 첫 후진 탭 지속(초)
        self.declare_parameter('reverse_neutral_time', 0.2)  # 탭 뒤 중립 지속(초)

        i2c_bus = int(self.get_parameter('i2c_bus').value)
        pca9685_addr = int(self.get_parameter('pca9685_addr').value)
        steering_channel = int(self.get_parameter('steering_channel').value)
        throttle_channel = int(self.get_parameter('throttle_channel').value)
        self.vehicle_config_file = os.path.expanduser(
            str(self.get_parameter('vehicle_config_file').value)
        )
        self.use_joystick_control = bool(self.get_parameter('use_joystick_control').value)
        joystick_topic = str(self.get_parameter('joystick_topic').value)
        control_topic = str(self.get_parameter('control_topic').value)
        command_hz = float(self.get_parameter('command_hz').value)
        if command_hz <= 0.0:
            raise ValueError('command_hz must be greater than 0')

        self.command_hz = command_hz
        self.reverse_speed = abs(float(self.get_parameter('reverse_speed').value))
        self.reverse_tap_time = float(self.get_parameter('reverse_tap_time').value)
        self.reverse_neutral_time = float(self.get_parameter('reverse_neutral_time').value)
        self.steer_trim = self.load_steer_trim()

        self.d3_racer = D3Racer(
            i2c_bus=i2c_bus,
            pca9685_addr=pca9685_addr,
            steering_channel=steering_channel,
            throttle_channel=throttle_channel,
        )

        self.get_logger().info(
            'd3_racer configured:\n'
            f'  i2c_bus={i2c_bus}\n'
            f'  pca9685_addr=0x{pca9685_addr:02X}\n'
            f'  steering_channel={steering_channel}\n'
            f'  throttle_channel={throttle_channel}\n'
            f'  steer_trim={self.steer_trim}\n'
            f'  use_joystick_control={self.use_joystick_control}\n'
            f'  joystick_topic={joystick_topic}\n'
            f'  control_topic={control_topic}\n'
            f'  command_hz={self.command_hz}\n'
            f'  vehicle_config_file={self.vehicle_config_file}'
        )

        self.throttle = 0.0
        self.steering = self.steer_trim
        self.e_stop_active = False

        # 후진 진입 시각(더블탭 자동화용). 스틱이 중립/전진이면 None.
        self.reverse_since = None

        # Control inputs
        self.create_subscription(
            Joystick,
            joystick_topic,
            self.joystick_callback,
            10,
        )
        self.create_subscription(
            Control,
            control_topic,
            self.control_callback,
            10,
        )

        # Command output loop
        self.timer = self.create_timer(1.0 / self.command_hz, self.timer_callback)

    def timer_callback(self):
        if self.e_stop_active:
            self.apply_actuation(self.steering, 0.0)
            return

        self.apply_actuation(self.steering, self.resolve_reverse(self.throttle))

    def resolve_reverse(self, throttle):
        """스틱을 아래로 내리면(throttle < 0) 후진 더블탭을 자동으로 넣어 출력한다.

        이 ESC는 후진 진입 시 "후진(브레이크) -> 중립 -> 후진" 순서를 거쳐야만
        실제로 후진한다. 스틱을 내린 순간부터의 경과 시간으로 단계를 나눈다:
          0 ~ tap                : 후진 탭(-reverse_speed)
          tap ~ tap+neutral      : 중립(0)
          그 이후                 : 후진 유지(-reverse_speed)
        조이스틱 후진 값은 너무 약하므로 실제 세기는 reverse_speed로 대체한다.
        전진/중립으로 돌아오면 초기화되어 다음 후진 때 다시 더블탭을 넣는다.
        """
        if throttle >= 0.0:
            self.reverse_since = None
            return throttle

        now = self.get_clock().now().nanoseconds / 1e9
        if self.reverse_since is None:
            self.reverse_since = now
        elapsed = now - self.reverse_since

        if self.reverse_tap_time <= elapsed < self.reverse_tap_time + self.reverse_neutral_time:
            return 0.0
        return -self.reverse_speed

    def apply_actuation(self, steering, throttle):
        self.d3_racer.set_steering_percent(float(steering))
        self.d3_racer.set_throttle_percent(float(throttle))

    def joystick_callback(self, msg: Joystick):
        if bool(msg.e_stop_en):
            self.engage_e_stop()
            return

        self.release_e_stop()

        if not self.use_joystick_control:
            return

        self.steering = float(msg.control_msg.steering)
        self.throttle = float(msg.control_msg.throttle)

    def control_callback(self, msg: Control):
        if self.e_stop_active or self.use_joystick_control:
            return

        self.steering = float(msg.steering)
        self.throttle = float(msg.throttle)

    def engage_e_stop(self):
        if self.e_stop_active:
            return

        self.e_stop_active = True
        self.throttle = 0.0
        self.apply_actuation(self.steering, 0.0)
        self.get_logger().warning('E-STOP engaged. Ignoring incoming throttle commands.')

    def release_e_stop(self):
        if not self.e_stop_active:
            return

        self.e_stop_active = False
        self.throttle = 0.0
        self.apply_actuation(self.steering, 0.0)
        self.get_logger().warning('E-STOP released. Resuming throttle commands.')

    def load_steer_trim(self):
        if not os.path.exists(self.vehicle_config_file):
            return 0.0

        try:
            with open(self.vehicle_config_file, 'r', encoding='utf-8') as config_stream:
                config_data = yaml.safe_load(config_stream) or {}
        except Exception as exc:
            self.get_logger().warning(
                f'Failed to read vehicle config file {self.vehicle_config_file}: {exc}'
            )
            return 0.0

        return float(config_data.get('STEER_TRIM', 0.0))

    def destroy_node(self):
        try:
            if hasattr(self, 'd3_racer') and self.d3_racer is not None:
                self.apply_actuation(self.steer_trim, 0.0)
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt. Shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()
