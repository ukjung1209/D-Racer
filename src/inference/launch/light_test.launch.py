"""신호등 미션 테스트 런치 (차선주행 + 신호등 출발/정지).

── 팀 병렬개발용 (decision_light_node) ──
파이프라인:
  camera ─┬─ lane_node   (/lane/state)          ─┐
          └─ object_node (/object/detections)   ─┴─ decision_light_node (/control) → control_node → PWM

기본 동작: 차선추종 위에 신호등 상태머신을 얹은 자율주행.
  → decision_light_node가 /lane/state로 PD 차선추종을 하면서, /object/detections의
    green/red 신호등으로 출발(HOLD→GO)·정지(GO→STOP)·재출발(STOP→GO)한다.
  → 안전상 **수동으로 시작**. 조이스틱 A: 수동↔자동 토글, X: E-STOP.
  → 반대로 꺾이면 `ros2 param set /decision_light_node steering_sign 1.0`.

신호등 튜닝 (실시간 `ros2 param set /decision_light_node <param> <value>`):
  → 신호등 크기 확인:      `ros2 topic echo /object/detections` (area_ratio 읽기)
  → 정지선 앞 정지 트리거:   stop_area_trigger   (기본 0.04)
  → 출발/정지 확정 표수:     green_votes_needed / red_votes_needed (기본 3)
  → 신호 없이 바로 출발 테스트: `ros2 param set /decision_light_node start_require_green false`
  → 검출 confidence 임계:    `ros2 param set /object_node conf_threshold 0.5`

monitor:=false 로 대시보드 없이 토픽만 확인 가능.
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


def get_model_path():
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'best_320.onnx'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/D-Racer/best_320.onnx'


def generate_launch_description():
    vehicle_config_path = get_vehicle_config_path()
    model_path = get_model_path()
    monitor = LaunchConfiguration('monitor')

    return LaunchDescription([
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

        # 차선 검출 → /lane/state (+ /lane/debug/raw, /lane/debug/bev 오버레이)
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
                    # --- BEV 사다리꼴 (실트랙 튜닝값) ---
                    'bev_top_left': 0.25,
                    'bev_top_right': 0.75,
                    'bev_top_y': 0.32,
                    'bev_bottom_left': 0.05,
                    'bev_bottom_right': 0.95,
                    'publish_debug': True,
                },
            ],
        ),

        # 신호등/표지판 검출 → /object/detections (+ /object/debug 오버레이)
        Node(
            package='inference',
            executable='object_node',
            name='object_node',
            output='screen',
            parameters=[
                {
                    'model_file': model_path,
                    'conf_threshold': 0.35,
                    'publish_debug': True,
                },
            ],
        ),

        # /lane/state + /object/detections → 차선추종 + 신호등 상태머신 → /control
        Node(
            package='inference',
            executable='decision_light_node',
            name='decision_light_node',
            output='screen',
            parameters=[
                {
                    'vehicle_config_file': vehicle_config_path,
                    # --- 차선추종 PD ---
                    'steer_kp': 0.8,
                    'steer_kd': 0.4,
                    'steer_ka': 0.0,
                    'steering_sign': -1.0,
                    'base_throttle': 0.15,
                    # --- 신호등 미션 (실트랙에서 area_ratio 실측 후 튜닝) ---
                    'enable_light_mission': True,
                    'start_require_green': True,
                    'light_conf_min': 0.5,
                    'green_votes_needed': 3,
                    'red_votes_needed': 3,
                    'stop_area_trigger': 0.04,
                },
            ],
        ),

        # /control → 실제 PWM 구동. 시작은 수동(안전), A버튼으로 자동 전환
        Node(
            package='control',
            executable='control_node',
            name='control_node',
            output='screen',
            parameters=[
                {
                    'use_joystick_control': True,  # 시작은 수동(안전). A버튼으로 자동 전환
                    'mode_toggle_enable': True,    # A버튼 수동/자동 토글 허용
                    'vehicle_config_file': vehicle_config_path,
                },
            ],
        ),

        # E-STOP(X)/모드토글(A)용
        Node(
            package='joystick',
            executable='joystick_node',
            name='joystick_node',
            output='screen',
            parameters=[
                {
                    'calibration_mode': False,
                    'manual_mode_start': True,    # 시작은 수동(안전). A로 자동 전환
                    'vehicle_config_file': vehicle_config_path,
                },
            ],
        ),

        # 대시보드 — GRAYSCALE=신호등 검출 박스(area_ratio 확인), BLUR=차선 BEV
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
                    'opencv_grayscale_topic': '/object/debug',     # 신호등 박스+라벨
                    'opencv_blur_topic': '/lane/debug/bev',        # 펼친 BEV+검출점
                },
            ],
        ),
    ])
