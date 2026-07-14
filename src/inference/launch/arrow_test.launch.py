"""표지판 갈림길 미션 테스트 런치 (차선주행 + 좌/우 화살표 갈림길 제어).

── 팀 병렬개발용 (decision_arrow_node) ──
파이프라인:
  camera ─┬─ lane_node   (/lane/state)          ─┐
          └─ object_node (/object/detections)   ─┴─ decision_arrow_node (/control) → control_node → PWM

기본 동작: 차선추종(lane) 위에 좌/우 표지판 갈림길 상태머신을 얹은 풀 자율주행.
  → decision_arrow_node가 /lane/state로 PD 차선추종을 하면서, /object/detections의
    left/right 표지판을 투표해 방향을 확정하고, 표지판이 코앞이 되면 lane_node에
    branch_hint를 보내 확정 방향 라인만 hugging(반대 라인 무시)해 갈래로 진입한다.

수동/자동 전환: 조이스틱으로 제어 (안전상 **수동으로 시작**).
  → A 버튼: 수동(스틱) ↔ 자동(decision_arrow_node) 토글.  X 버튼: E-STOP.
  → ⚠️ 바퀴를 먼저 띄우고 확인. 반대로 꺾이면
     `ros2 param set /decision_arrow_node steering_sign 1.0`.

갈림길 튜닝 (실시간 `ros2 param set /decision_arrow_node <param> <value>`):
  → 표지판 크기 확인:   `ros2 topic echo /object/detections` (area_ratio 읽기)
  → 멀리서 방향 판독 시작:  sign_area_min      (기본 0.01)
  → hugging 시작 시점:     fork_area_trigger  (기본 0.06, 표지판이 이보다 크면 시작)
  → 라인에서 얼마나 붙나:  `ros2 param set /lane_node hug_bias_px 60` (hugging bias, BEV px)
  → 통과 후 복귀 지연:     hug_hold_sec       (기본 1.0, 표지판 놓치고 이 시간 뒤 평소)
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
                    # --- 흰색 차선 모드 (white_s_max/white_v_min으로 튜닝) ---
                    'lane_color': 'white',
                    'white_s_max': 40,     # 채도 상한↓ = 회색/색깔 배제 (안 잡히면 ↑, 잡티 많으면 ↓)
                    'white_v_min': 120,    # 명도 하한↑ = 완전 흰색만 (노출40에서 흰선 밝기 실측 ~130 → 120)
                    # 빛반사(넓은 밝은 덩어리) 제거: 커널보다 작은 밝은 구조(얇은 차선)만 남김
                    'white_tophat_ksize': 21,  # 글레어 남으면 ↓, 차선 끊기면 ↑ (0=끔)
                    'white_tophat_min': 18,    # top-hat 대비 하한 (글레어 남으면 ↑, 차선 끊기면 ↓)
                    'roi_top_px': 45,
                    'num_bands': 10,
                    'line_split_gap_px': 40,
                    'lane_half_width_px': 90,
                    'hug_bias_px': 75,     # 갈림길 hugging 시 라인에서 떨어지는 정도(BEV px)
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
                    # 반사광 억제로 노출 낮추면 어두워 YOLO가 놓침 → YOLO 입력만 밝기 복원.
                    # 어두우면 낮춰가며 튜닝: ros2 param set /object_node yolo_gamma 0.5
                    'yolo_gamma': 0.4,
                    'yolo_clahe': False,   # 반사광/조명 편차 심하면 True
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
                    'base_throttle': 0.28,       # 직선 최고속 (가변속도가 코너에서만 깎아냄)
                    # 가변속도: 직선 빠르게 / 코너 미리 감속 (angle 예측 + offset 보정)
                    'speed_ka': 0.4,             # |angle| 예측 감속 gain
                    'speed_ko': 0.4,             # |offset| 보정 감속 gain
                    'min_throttle': 0.20,        # 감속 하한
                    'throttle_accel_rate': 0.5,  # 가속 슬루 제한(급가속 튐 방지)
                    # --- 좌/우 갈림길 (단순화: 표지판 보이면 즉시 그 라인 hugging) ---
                    'enable_fork_mission': True,
                    'sign_conf_min': 0.5,
                    'sign_area_min': 0.01,       # 이 크기 이상이어야 방향 인정 (너무 일찍 틀면 ↑)
                    'hug_hold_sec': 1.0,         # 표지판 놓치고 이 시간 뒤 평소 복귀
                    'sign_stop_sec': 0.7,        # 표지판 인식+두 차선 보일 때 정지 시간(stop-and-go), 0=끔
                    'stop_lane_conf_min': 0.5,   # 이 lane confidence 이상일 때만 정지(두 차선 확실)
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
                    # 2패널만: YOLO 박스 + 차선 BEV. 나머지는 ''로 꺼서 모니터 부하↓
                    'image_topic': '',                             # 카메라 원본 패널 숨김
                    'opencv_grayscale_topic': '/object/debug',     # 표지판 박스+라벨
                    'opencv_blur_topic': '/lane/debug/bev',        # 펼친 BEV+검출점
                    'opencv_edge_topic': '',                       # lane raw 패널 숨김
                },
            ],
        ),
    ])
