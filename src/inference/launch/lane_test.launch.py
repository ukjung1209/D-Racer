"""lane_node 튜닝/차선추종 테스트용 런치.

파이프라인:  camera → lane_node(/lane/state) → decision_node(/control) → control_node → PWM

기본:  camera_node + lane_node + monitor_node  (바퀴 안 움직임, 검출만)
  → 대시보드(grayscale 패널)에 /lane/debug/compressed 오버레이가 뜬다.
  → `ros2 param set /lane_node <param> <value>` 로 검출 실시간 튜닝.

차선추종 주행:  `ros2 launch inference lane_test.launch.py drive:=true`
  → decision_node + control_node + joystick_node(E-STOP)를 추가로 띄운다.
  → lane_node는 /lane/state만 발행하고, decision_node가 PD로 /control을 만든다.
  → ⚠️ 바퀴를 먼저 띄우고 확인할 것. 반대로 꺾이면
     `ros2 param set /decision_node steering_sign 1.0`.
  → 조향 게인 튜닝: `ros2 param set /decision_node steer_kp 1.0` 등.
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def get_vehicle_config_path():
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'src' / 'config' / 'vehicle_config.yaml'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/D-Racer/src/config/vehicle_config.yaml'


def generate_launch_description():
    vehicle_config_path = get_vehicle_config_path()
    drive = LaunchConfiguration('drive')
    monitor = LaunchConfiguration('monitor')

    return LaunchDescription([
        DeclareLaunchArgument(
            'drive',
            default_value='true',
            description='true면 decision_node + control_node를 띄워 실제 차선추종 주행 '
                        '(검출만 하려면 drive:=false)',
        ),
        DeclareLaunchArgument(
            'monitor',
            default_value='true',
            description='true면 monitor_node를 띄워 대시보드에서 오버레이 확인',
        ),

        Node(
            package='camera',
            executable='camera_node',
            name='camera_node',
            output='screen',
            parameters=[
                {'vehicle_config_file': vehicle_config_path},
            ],
        ),

        # 차선 검출 → /lane/state 발행 (+ /lane/debug/compressed 오버레이)
        Node(
            package='inference',
            executable='lane_node',
            name='lane_node',
            output='screen',
            parameters=[
                {
                    'vehicle_config_file': vehicle_config_path,
                    # --- 회색 트랙 오렌지 라인 튜닝값 (2026-07-03 실측) ---
                    'lane_color': 'orange',
                    'hsv_lower': [8.0, 90.0, 90.0],
                    'hsv_upper': [26.0, 255.0, 255.0],
                    'roi_top_px': 45,
                    'num_bands': 10,
                    'line_split_gap_px': 40,
                    'lane_half_width_px': 90,
                    'publish_debug': True,
                },
            ],
        ),

        # /lane/state → PD 차선추종 → /control (주행 모드에서만)
        Node(
            package='inference',
            executable='decision_node',
            name='decision_node',
            output='screen',
            condition=IfCondition(drive),
            parameters=[
                {
                    'vehicle_config_file': vehicle_config_path,
                    'steer_kp': 1.5,
                    'steer_kd': 0.3,
                    'steer_ka': 0.0,
                    'steering_sign': -1.0,
                    'base_throttle': 0.15,
                },
            ],
        ),

        # /control → 실제 PWM 구동 (주행 모드에서만)
        Node(
            package='control',
            executable='control_node',
            name='control_node',
            output='screen',
            condition=IfCondition(drive),
            parameters=[
                {
                    'use_joystick_control': False,
                    'vehicle_config_file': vehicle_config_path,
                },
            ],
        ),

        # E-STOP(조이스틱 X)용 — 주행 모드에서만
        Node(
            package='joystick',
            executable='joystick_node',
            name='joystick_node',
            output='screen',
            condition=IfCondition(drive),
            parameters=[
                {
                    'calibration_mode': False,
                    'vehicle_config_file': vehicle_config_path,
                },
            ],
        ),

        # 대시보드 — grayscale 패널을 lane 오버레이로 리맵해 트랙 검출 상태 확인
        Node(
            package='monitor',
            executable='monitor_node',
            name='monitor_node',
            output='screen',
            condition=IfCondition(monitor),
            parameters=[
                {
                    'vehicle_config_file': vehicle_config_path,
                    'debug_image': True,
                    'opencv_grayscale_topic': '/lane/debug/raw',   # 원본+사다리꼴
                    'opencv_blur_topic': '/lane/debug/bev',        # 펼친 BEV+검출점
                },
            ],
        ),
    ])
