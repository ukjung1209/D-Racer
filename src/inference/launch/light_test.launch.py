"""신호등 미션 테스트 런치 (차선주행 + 신호등 출발/정지).

── 팀 병렬개발용 (decision_light_node) ──
파이프라인:
  camera ─┬─ lane_node   (/lane/state)          ─┐
          └─ object_node (/object/detections)   ─┴─ decision_light_node (/control) → control_node → PWM

기본 동작: 차선추종 위에 신호등 상태머신을 얹은 자율주행.
  → decision_light_node가 /lane/state로 PD 차선추종을 하면서, /object/detections의
    green/red 신호등으로 출발(STOPPED→GO)·정지(GO→STOPPED)한다.
      STOPPED : 정지. 초록불을 green_votes_needed프레임 연속 보면 출발.
      GO      : 차선추종 주행. 빨간불을 red_votes_needed프레임 연속 보면 정지.
  → 처음엔 가만히 있다가 초록불 3프레임 보면 출발, 빨간불 3프레임 보면 정지.
  → 안전상 **수동으로 시작**. 조이스틱 A: 수동↔자동 토글, X: E-STOP.
  → 반대로 꺾이면 `ros2 param set /decision_light_node steering_sign 1.0`.

차선(검정 트랙 + 흰 사이드라인):
  → 흰색(lane_color='white') 모드. 채도 낮고 명도 높은 픽셀만 흰선으로 잡는다.
  → 안 잡히면 대시보드 BLUR(BEV)을 보며 명도 하한을 내린다:
      `ros2 param set /lane_node white_v_min 160` (잡티 많으면 반대로 올림)
      색깔·회색이 섞여 잡히면  `ros2 param set /lane_node white_s_max 30` 으로 낮춤.

신호등 튜닝 (실시간 `ros2 param set /decision_light_node <param> <value>`):
  → 출발 확정 표수:   green_votes_needed (기본 3, 초록불 3프레임 연속이면 출발)
  → 정지 확정 표수:   red_votes_needed   (기본 3, 빨간불 3프레임 연속이면 정지)
  → 검출 confidence:  light_conf_min     (기본 0.5) / `ros2 param set /object_node conf_threshold 0.35`

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
                {
                    'vehicle_config_file': vehicle_config_path,
                    'debug_log': False,   # 매 프레임 'Published frame' 로그 끔(터미널 렉 방지)
                },
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
                    # 어두운 트랙 + 흰 사이드라인. 노출40으로 낮춰 흰선이 어두워져
                    # v_min을 200→175로 내림. 안 잡히면 더 내리고, 잡티 많으면 올린다.
                    #   ros2 param set /lane_node white_v_min 160
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
                    # --- BEV 사다리꼴 (오렌지 원형 트랙 튜닝값) ---
                    'bev_top_left': 0.1,
                    'bev_top_right': 0.9,
                    'bev_top_y': 0.32,
                    'bev_bottom_left': 0.0,
                    'bev_bottom_right': 1.0,
                    'publish_debug': True,
                },
            ],
        ),

        # 신호등 검출 → /object/detections (+ /object/debug 오버레이)
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
                    'steer_kp': 1.3,
                    'steer_kd': 0.5,
                    'steer_ka': 0.3,
                    'steering_sign': -1.0,
                    'base_throttle': 0.16,
                    # --- 신호등 미션 ---
                    'enable_light_mission': True,
                    'light_conf_min': 0.5,
                    'green_votes_needed': 3,    # 초록불 3프레임 연속이면 출발
                    'red_votes_needed': 2,      # 빨간불 2프레임 연속이면 정지
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
                    'debug_log_enable': False,    # '[Joystick DBG]' 5Hz 로그 끔(터미널 렉 방지)
                    'vehicle_config_file': vehicle_config_path,
                },
            ],
        ),

        # 대시보드 — GRAYSCALE=신호등 검출 박스, BLUR=차선 BEV, EDGE=원본+사다리꼴
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
                    'opencv_edge_topic': '/lane/debug/raw',        # 원본 ROI+BEV 사다리꼴(초록)
                },
            ],
        ),
    ])
