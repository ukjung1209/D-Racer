"""대회 주행용 통합 런치 — 차선추종 + 신호등 + 갈림길 + 동적장애물 (전 미션).

── decision_arrow/light/obstacle_node를 합친 decision_node 하나로 주행 ──
파이프라인:
  camera ─┬─ lane_node   → /lane/state          ─┐
          ├─ object_node → /object/detections    ─┼─ decision_node → /control → control
          └─ camera/image/compressed ────────────┘   └→ /lane/branch_hint

decision_node가 조향(항상 lane PD)과 스로틀(미션별 목표의 min 중재)을 한 곳에서
결정하므로 노드끼리 /control을 두고 싸우지 않는다. 세 미션은 decision.yaml의
enable_*_mission으로 켜고 끈다 → 튜닝 땐 한 미션만 켜면 이 런치로 단일 미션 테스트.

파라미터(kp/kd/ka, base_throttle, 미션 임계값, BEV 등)는 전부 config/decision.yaml.
주행마다 그 yaml만 고친다(이 파이썬 런치는 안 건드림). 트랙별 프리셋은:
  ros2 launch inference race.launch.py params_file:=/경로/decision_track_A.yaml

수동/자동: 안전상 **수동 시작**. 조이스틱 A=수동↔자동 토글, X=E-STOP.
  → 반대로 꺾이면 decision.yaml의 steering_sign을 1.0으로, 또는
    ros2 param set /decision_node steering_sign 1.0

monitor:=false 로 대시보드 없이 토픽만 확인 가능.

⚠️ 아루코는 시스템 cv2 4.5.4(apt)에 있음. pip로 opencv 깔지 말 것(카메라가 죽음).
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _find_in_tree(rel_path, fallback):
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / rel_path
        if candidate.exists():
            return str(candidate)
    return fallback


def get_vehicle_config_path():
    return _find_in_tree(
        'src/config/vehicle_config.yaml',
        '/home/topst/D-Racer/src/config/vehicle_config.yaml')


def get_default_params_path():
    return _find_in_tree(
        'src/config/decision.yaml',
        '/home/topst/D-Racer/src/config/decision.yaml')


def get_model_path():
    return _find_in_tree('best_320.onnx', '/home/topst/D-Racer/best_320.onnx')


def generate_launch_description():
    vehicle_config_path = get_vehicle_config_path()
    model_path = get_model_path()
    monitor = LaunchConfiguration('monitor')
    params_file = LaunchConfiguration('params_file')

    return LaunchDescription([
        DeclareLaunchArgument(
            'monitor',
            default_value='true',
            description='true면 monitor_node를 띄워 대시보드에서 오버레이 확인',
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value=get_default_params_path(),
            description='decision_node/lane_node/object_node 튜닝 파라미터 yaml '
                        '(트랙별 프리셋으로 교체 가능)',
        ),

        Node(
            package='camera',
            executable='camera_node',
            name='camera_node',
            output='screen',
            parameters=[
                {'vehicle_config_file': vehicle_config_path, 'debug_log': False},
            ],
        ),

        # 차선 검출 → /lane/state (+ /lane/debug/raw, /lane/debug/bev 오버레이)
        Node(
            package='inference',
            executable='lane_node',
            name='lane_node',
            output='screen',
            parameters=[
                {'vehicle_config_file': vehicle_config_path},
                params_file,   # lane_node: 섹션 (BEV/흰색 마스크 튜닝값)
            ],
        ),

        # 신호등/표지판 검출 → /object/detections (+ /object/debug 오버레이)
        Node(
            package='inference',
            executable='object_node',
            name='object_node',
            output='screen',
            parameters=[
                {'model_file': model_path},
                params_file,   # object_node: 섹션 (conf/gamma)
            ],
        ),

        # 통합 결정: /lane/state + /object/detections + camera/image → /control
        Node(
            package='inference',
            executable='decision_node',
            name='decision_node',
            output='screen',
            parameters=[
                {'vehicle_config_file': vehicle_config_path},
                params_file,   # decision_node: 섹션 (PD + 세 미션)
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
                    'use_joystick_control': True,
                    'mode_toggle_enable': True,
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
                    'manual_mode_start': True,
                    'debug_log_enable': False,
                    'throttle_scale': 0.25,        # 조이스틱 기본 throttle(=accel_ratio 시작값). 버튼으로 0.12~0.4 조절
                    'vehicle_config_file': vehicle_config_path,
                },
            ],
        ),

        # 대시보드 4패널 매핑
        #   Image Status = 차선 인지(/lane/debug/bev)
        #   Gray Scale   = YOLO(/object/debug: 신호등/표지판 박스)
        #   BLUR         = 아루코(/obstacle/debug: 빨강 ROI/마커/미션 상태)
        #   EDGE         = BEV 사다리꼴(/lane/debug/raw: 원본 ROI + 초록 변환영역)
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
                    'image_topic': '/lane/debug/bev',              # Image Status = 차선 인지
                    'opencv_grayscale_topic': '/object/debug',     # Gray Scale = YOLO
                    'opencv_blur_topic': '/obstacle/debug',        # BLUR = 아루코
                    'opencv_edge_topic': '/lane/debug/raw',        # EDGE = BEV 사다리꼴
                },
            ],
        ),
    ])
