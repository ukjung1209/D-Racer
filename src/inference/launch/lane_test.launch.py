"""lane_node 튜닝/차선추종 테스트용 런치.

파이프라인:  camera → lane_node(/lane/state) → decision_node(/control) → control_node → PWM

기본:  camera_node + lane_node + monitor_node  (바퀴 안 움직임, 검출만)
  → 대시보드 GRAYSCALE=/lane/debug/raw(원본+사다리꼴), BLUR=/lane/debug/bev(펼친 BEV).
  → `ros2 param set /lane_node <param> <value>` 로 검출 실시간 튜닝.

차선추종 주행:  `ros2 launch inference lane_test.launch.py drive:=true`
  → decision_arrow_node + control_node + joystick_node(E-STOP)를 추가로 띄운다.
  → lane_node는 /lane/state만 발행하고, decision_arrow_node가 PD로 /control을 만든다.
    (여기선 object_node가 없어 갈림길 미션은 꺼두고 순수 차선추종만 한다.)
  → ⚠️ 바퀴를 먼저 띄우고 확인할 것. 반대로 꺾이면
     `ros2 param set /decision_arrow_node steering_sign 1.0`.
  → 조향 게인 튜닝: `ros2 param set /decision_arrow_node steer_kp 1.0` 등.
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

        # 차선 검출 → /lane/state 발행 (+ /lane/debug/raw, /lane/debug/bev 오버레이)
        Node(
            package='inference',
            executable='lane_node',
            name='lane_node',
            output='screen',
            parameters=[
                {
                    'vehicle_config_file': vehicle_config_path,
                    # --- 흰색 차선 모드 (white_s_max/white_v_min으로 튜닝) ---
                    'lane_color': 'white',
                    'white_s_max': 20,     # 채도 상한↓ = 회색/색깔 배제 (안 잡히면 ↑, 잡티 많으면 ↓). 40→20: 푸르스름한 트랙 무늬(반사) 배제
                    'white_v_min': 120,    # 명도 하한↑ = 완전 흰색만 (노출40에서 흰선 밝기 실측 ~130 → 120)
                    # 빛반사(넓은 밝은 덩어리) 제거: 커널보다 작은 밝은 구조(얇은 차선)만 남김
                    'white_tophat_ksize': 21,  # 글레어 남으면 ↓, 차선 끊기면 ↑ (0=끔)
                    'white_tophat_min': 18,    # top-hat 대비 하한 (글레어 남으면 ↑, 차선 끊기면 ↓)
                    'roi_top_px': 45,
                    'num_bands': 10,
                    'line_split_gap_px': 40,
                    'lane_half_width_px': 90,
                    # --- BEV 사다리꼴 (실트랙 튜닝값) ---
                    'bev_top_left': 0.1,
                    'bev_top_right': 0.9,
                    'bev_top_y': 0.32,
                    'bev_bottom_left': 0.0,
                    'bev_bottom_right': 1.0,
                    'publish_debug': True,
                },
            ],
        ),

        # /lane/state → PD 차선추종 → /control (주행 모드에서만)
        Node(
            package='inference',
            executable='decision_arrow_node',
            name='decision_arrow_node',
            output='screen',
            condition=IfCondition(drive),
            parameters=[
                {
                    'vehicle_config_file': vehicle_config_path,
                    'steer_kp': 0.8,
                    'steer_kd': 0.45,
                    'steer_ka': 0.0,
                    'steering_sign': -1.0,
                    'base_throttle': 0.31,
                    # 가변속도: 직선 빠르게 / 코너 미리 감속 (angle 예측 + offset 보정)
                    'speed_ka': 0.4,
                    'speed_ko': 0.4,
                    'min_throttle': 0.10,
                    'throttle_accel_rate': 0.5,
                    'enable_fork_mission': False,  # object_node 없음 → 순수 차선추종
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
                    'use_joystick_control': True,  # 시작은 수동(안전). A버튼으로 자동 전환
                    'mode_toggle_enable': True,    # A버튼 수동/자동 토글 허용
                    'vehicle_config_file': vehicle_config_path,
                },
            ],
        ),

        # E-STOP(X)/모드토글(A)용 — 주행 모드에서만
        Node(
            package='joystick',
            executable='joystick_node',
            name='joystick_node',
            output='screen',
            condition=IfCondition(drive),
            parameters=[
                {
                    'calibration_mode': False,
                    'manual_mode_start': True,    # 시작은 수동(안전). A로 자동 전환
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
