"""동적 장애물 미션 테스트 런치 (차선주행 + 빨강 구간 감속 + 아루코 정지/재출발).

── 팀 병렬개발용 (decision_obstacle_node) ──
파이프라인:
  camera ─┬─ lane_node (/lane/state)          ─┐
          └────────(camera/image/compressed)──┴─ decision_obstacle_node (/control) → control_node → PWM

동적 장애물 인식은 전부 decision_obstacle_node 안에서 OpenCV로 직접 한다
(object_node/YOLO 불필요 → 이 런치에서 뺐다). decision_obstacle_node가:
  1) /lane/state로 PD 차선추종(steering),
  2) camera/image/compressed에서 빨간 트랙 배경 + 아루코 마커를 직접 감지(throttle)
하며 상태머신을 돌린다:
  CRUISE(원속도) → [빨강배경+양쪽차선] → APPROACH(70% 감속)
    → [아루코 보임] → STOP(정지) → [아루코 1.5초↑ 사라짐] → BOOST(원속도 통과)
    → [빨강 구간 이탈] → CRUISE 재무장
  → 안전상 **수동으로 시작**. 조이스틱 A: 수동↔자동 토글, X: E-STOP.
  → 반대로 꺾이면 `ros2 param set /decision_obstacle_node steering_sign 1.0`.

⚠️ 아루코는 이 보드의 시스템 cv2 4.5.4(apt)에 이미 들어있음. **pip로 opencv 깔지 말 것**
   (pip opencv는 GStreamer가 없어 camera_node가 죽음). cv2.aruco 없다는 에러가 나면
   pip opencv가 시스템 걸 가리는 것이니 `pip3 uninstall opencv-python opencv-contrib-python`.

튜닝 (실시간 `ros2 param set /decision_obstacle_node <param> <value>`,
      값은 /obstacle/debug 오버레이의 red=/aruco= 숫자를 보며 맞춘다):
  → 감속 진입 빨강 비율:    red_ratio_trigger  (기본 0.15, ROI에서 빨강 비율↑이면 진입)
  → 빨강 감지 ROI 상단:     red_roi_top_frac   (기본 0.5, 하단 절반만 봄)
  → 감속 확정 표수:         red_votes_needed   (기본 3)
  → 감속 배율:              slow_factor        (기본 0.7 = 원속도의 70%)
  → 정지 확정 표수:         aruco_votes_needed (기본 1 = 보이면 바로 정지)
  → 마커 통과 판정 시간:     clear_time_sec     (기본 1.5초)
  → 빨강 이탈 판정 시간:     red_clear_time_sec (기본 1.0초)

아루코 (이 마커 = 6X6_50 / ID3. 실측 튜닝 완료):
  → 320x160 저해상도라 6X6은 그냥은 거의 안 잡히고 엉뚱한 ID로 오독됨. 그래서:
    · aruco_upscale=2          검출 전 2배 업스케일 (6X6 검출률 급상승)
    · aruco_error_correction=0.4  오독을 valid ID로 복원 못하게 (false ID 제거)
    · aruco_target_id=3        ID3만 정지 트리거 (남은 오독 무시)
  → 마커가 화면에서 충분히 커야(대략 area_ratio 0.1↑) 안정적으로 읽힘. 실제
    미션은 마커가 코앞에서 올라오니 크게 보여 OK. 멀리서 안 잡히면 거리/해상도 탓.
  → 딕셔너리/타깃 변경:  `ros2 param set /decision_obstacle_node aruco_dict 6X6_50`,
                        `ros2 param set /decision_obstacle_node aruco_target_id 3`

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


def generate_launch_description():
    vehicle_config_path = get_vehicle_config_path()
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

        # /lane/state(조향) + camera/image(빨강·아루코 OpenCV 감지) → /control (+ /obstacle/debug)
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
                    'base_throttle': 0.18,
                    # --- 동적 장애물 미션 (실트랙에서 /obstacle/debug 보며 튜닝) ---
                    'enable_obstacle_mission': True,
                    'slow_factor': 0.7,          # 빨강 구간 감속 = 원속도의 70%
                    # 빨간 트랙 감지 (하단 절반 ROI의 빨강 픽셀 비율)
                    'red_roi_top_frac': 0.5,
                    'red_ratio_trigger': 0.15,
                    'red_votes_needed': 3,
                    'red_clear_time_sec': 1.0,
                    # 아루코 마커 (보이면 바로 정지, 1.5초 사라지면 재출발)
                    # 이 마커는 6X6_50 / ID3. 320x160 저해상도라 2x 업스케일 +
                    # 오류보정 억제로 실측 튜닝함(아래 검출기 파라미터 참고).
                    'aruco_dict': '6X6_50',
                    'aruco_target_id': 3,        # 이 ID만 정지(-1=아무 ID나). 오독 무시
                    'aruco_upscale': 2,          # 주행 중 기울기/흔들림 여유용(정지 마커는 1로도 100%). 불안정하면 3
                    'aruco_clahe': False,        # 흰 여백 두른 뒤로는 불필요(대비 충분) → CPU 절약. 조명 나쁘면 True
                    'aruco_votes_needed': 1,
                    'clear_time_sec': 1.5,
                    # 아루코 검출기 튜닝 (저대비/저해상도 6X6 실측값)
                    'aruco_min_perimeter_rate': 0.01,
                    'aruco_poly_accuracy': 0.08,
                    'aruco_adaptive_win_max': 45,
                    'aruco_error_correction': 0.4,   # ↓일수록 오독을 valid ID로 안 만듦
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

        # 대시보드 — GRAYSCALE=장애물 오버레이(빨강 ROI 비율+아루코 박스+상태), BLUR=차선 BEV
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
                    'opencv_grayscale_topic': '/obstacle/debug',   # 빨강/아루코/상태 오버레이
                    'opencv_blur_topic': '/lane/debug/bev',        # 펼친 BEV+검출점
                },
            ],
        ),
    ])
