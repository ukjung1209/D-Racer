"""표지판 갈림길 미션 테스트 런치 (차선주행 + 좌/우 화살표 갈림길 제어).

── 팀 병렬개발용 (decision_arrow_node) ──
파이프라인:
  camera ─┬─ lane_node   (/lane/state)          ─┐
          └─ object_node (/object/detections)   ─┴─ decision_arrow_node (/control) → control_node → PWM

기본 동작: 차선추종(lane) 위에 좌/우 표지판 갈림길 상태머신을 얹은 풀 자율주행.
  → decision_arrow_node가 /lane/state로 PD 차선추종을 하면서, /object/detections의
    left/right 표지판을 투표·확정해 갈림길에서 확정 방향으로 조향 bias를 준다.

수동/자동 전환: 조이스틱으로 제어 (안전상 **수동으로 시작**).
  → A 버튼: 수동(스틱) ↔ 자동(decision_arrow_node) 토글.  X 버튼: E-STOP.
  → ⚠️ 바퀴를 먼저 띄우고 확인. 반대로 꺾이면
     `ros2 param set /decision_arrow_node steering_sign 1.0`.

갈림길 튜닝 (실시간 `ros2 param set /decision_arrow_node <param> <value>`):
  → 표지판 크기 확인:   `ros2 topic echo /object/detections` (area_ratio 읽기)
  → 멀리서 방향 판독 시작:  sign_area_min      (기본 0.01)
  → 코앞에서 꺾기 시작:     fork_area_trigger  (기본 0.06)
  → 꺾는 세기(좌우 반대면 음수): fork_bias      (기본 0.5)
  → 검출 confidence 임계:   `ros2 param set /object_node conf_threshold 0.5`

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

        # /lane/state + /object/detections → 차선추종 + 갈림길 상태머신 → /control
        Node(
            package='inference',
            executable='decision_arrow_node',
            name='decision_arrow_node',
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
                    # --- 좌/우 갈림길 미션 (실트랙에서 area_ratio 실측 후 튜닝) ---
                    'enable_fork_mission': True,
                    'sign_conf_min': 0.5,
                    'sign_area_min': 0.01,
                    'sign_votes_needed': 5,
                    'fork_area_trigger': 0.06,
                    'fork_bias': 0.5,
                    'fork_duration_sec': 2.0,
                    'fork_throttle': 0.13,
                    'arm_timeout_sec': 3.0,
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

        # 대시보드 — GRAYSCALE=표지판 검출 박스(area_ratio 확인), BLUR=차선 BEV
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
                    'opencv_grayscale_topic': '/object/debug',     # 표지판 박스+라벨
                    'opencv_blur_topic': '/lane/debug/bev',        # 펼친 BEV+검출점
                },
            ],
        ),
    ])
