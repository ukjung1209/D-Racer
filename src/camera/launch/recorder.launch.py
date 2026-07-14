from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    fps_arg = DeclareLaunchArgument(
        'fps', default_value='30.0',
        description='mp4 에 기록할 fps. 카메라 publish_hz 와 맞추면 실시간 속도.')
    fourcc_arg = DeclareLaunchArgument(
        'fourcc', default_value='mp4v',
        description='코덱. mp4v 가 안 열리면 XVID 로 바꾸고 확장자는 코드에서 .mp4 유지.')
    topic_arg = DeclareLaunchArgument(
        'subscribe_topic', default_value='camera/image/compressed',
        description='구독할 카메라 압축 이미지 토픽.')
    gamma_arg = DeclareLaunchArgument(
        'yolo_gamma', default_value='0.4',
        description='밝게 버전 감마 (decision.yaml 과 동일 기본값 0.4). <1 이면 밝아짐.')
    clahe_arg = DeclareLaunchArgument(
        'yolo_clahe', default_value='false',
        description='밝게 버전에 CLAHE 대비 평활화 적용 여부.')
    save_raw_arg = DeclareLaunchArgument(
        'save_raw', default_value='true',
        description='밝기 안 올린 원본 mp4 저장 여부.')
    save_bright_arg = DeclareLaunchArgument(
        'save_bright', default_value='true',
        description='밝기 올린(YOLO 입력용) mp4 저장 여부.')

    # 카메라 노드를 함께 켜서 recorder.launch 하나로 "카메라+녹화" 가 되게 한다.
    # 주의: 다른 launch 에서 camera_node 를 또 켜면 v4l2 장치 충돌이 나므로
    # 이 launch 를 쓸 때는 camera_node 를 중복 실행하지 말 것.
    camera_node = Node(
        package='camera',
        executable='camera_node',
        name='camera_node',
        output='screen',
        parameters=[{
            'publish_topic': LaunchConfiguration('subscribe_topic'),
        }],
    )

    recorder_node = Node(
        package='camera',
        executable='recorder_node',
        name='recorder_node',
        output='screen',
        parameters=[{
            'subscribe_topic': LaunchConfiguration('subscribe_topic'),
            'fps': LaunchConfiguration('fps'),
            'fourcc': LaunchConfiguration('fourcc'),
            'yolo_gamma': LaunchConfiguration('yolo_gamma'),
            'yolo_clahe': LaunchConfiguration('yolo_clahe'),
            'save_raw': LaunchConfiguration('save_raw'),
            'save_bright': LaunchConfiguration('save_bright'),
        }],
    )

    return LaunchDescription([
        fps_arg,
        fourcc_arg,
        topic_arg,
        gamma_arg,
        clahe_arg,
        save_raw_arg,
        save_bright_arg,
        camera_node,
        recorder_node,
    ])
