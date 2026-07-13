"""동적 장애물 미션 테스트 런치 (차선주행 + 아루코 마커 정지/재출발).

── 팀 병렬개발용 (decision_obstacle_node) ──
파이프라인:
  camera ─┬─ lane_node   (/lane/state)                    ─┐
          └─ object_node (/object/detections, aruco 포함)  ─┴─ decision_obstacle_node (/control) → control_node → PWM

동적 장애물은 아루코 마커로 나타난다. object_node를 enable_aruco:=True로 띄워
YOLO 신호등/표지판과 함께 아루코 마커를 검출해 같은 /object/detections에
class_name='obstacle'(area_ratio=근접도)로 발행한다.
  → decision_obstacle_node가 /lane/state로 PD 차선추종을 하면서, 전방 마커가
    충분히 커지면 정지(CRUISE→STOP_WAIT), 마커가 지나가면 재출발(STOP_WAIT→CRUISE).
  → 안전상 **수동으로 시작**. 조이스틱 A: 수동↔자동 토글, X: E-STOP.
  → 반대로 꺾이면 `ros2 param set /decision_obstacle_node steering_sign 1.0`.

⚠️ 아루코는 opencv-contrib에 있음. cv2.aruco 없다는 에러가 나면 보드에서
   `pip install opencv-contrib-python` 후 다시 실행.

장애물 튜닝 (실시간 `ros2 param set /decision_obstacle_node <param> <value>`):
  → 마커 크기 확인:        `ros2 topic echo /object/detections` (obstacle의 area_ratio)
  → 정지 트리거 크기:       obstacle_area_trigger  (기본 0.02, 클수록 더 가까이서 정지)
  → 정지 확정 표수:         obstacle_votes_needed  (기본 3)
  → 지나감 판정 시간:       clear_time_sec         (기본 1.0초)
  → 아루코 딕셔너리 변경:    `ros2 param set /object_node aruco_dict 5X5_100` (재기동 필요)

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

        # 신호등/표지판(YOLO) + 아루코 마커 검출 → /object/detections (+ /object/debug)
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
                    # 아루코 마커(동적 장애물)를 켠다 → object/detections에 obstacle 추가
                    'enable_aruco': True,
                    'aruco_dict': '4X4_50',
                },
            ],
        ),

        # /lane/state + /object/detections → 차선추종 + 장애물 정지/재출발 → /control
        Node(
            package='inference',
            executable='decision_obstacle_node',
            name='decision_obstacle_node',
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
                    # --- 동적 장애물 미션 (실트랙에서 area_ratio 실측 후 튜닝) ---
                    'enable_obstacle_mission': True,
                    'obstacle_area_trigger': 0.02,
                    'obstacle_votes_needed': 3,
                    'clear_time_sec': 1.0,
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

        # 대시보드 — GRAYSCALE=검출 박스(아루코 마커 area_ratio 확인), BLUR=차선 BEV
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
                    'opencv_grayscale_topic': '/object/debug',     # YOLO+아루코 박스
                    'opencv_blur_topic': '/lane/debug/bev',        # 펼친 BEV+검출점
                },
            ],
        ),
    ])
