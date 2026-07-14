import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage


def get_default_output_dir():
    for base_path in Path(__file__).resolve().parents:
        if (base_path / 'src').is_dir():
            return str(base_path / 'recordings')
    return '/home/topst/D-Racer/recordings'


class RecorderNode(Node):
    """카메라 압축 이미지 토픽을 구독해 mp4 영상으로 저장하는 노드.

    데이터셋 수집용. 실행하면 녹화 시작, Ctrl+C 로 정지하며 파일을 닫는다.
    원본 프레임을 그대로 담아두고, 필요한 jpg 프레임은 PC 에서
    extract_frames.py 로 뽑아 쓰는 흐름을 전제로 한다.
    """

    def __init__(self):
        super().__init__('recorder_node')

        # ROS parameters
        self.declare_parameter('subscribe_topic', 'camera/image/compressed')
        self.declare_parameter('output_dir', get_default_output_dir())
        # VideoWriter 에 기록되는 fps. 카메라 publish_hz 와 맞추면 실시간 속도로 재생된다.
        self.declare_parameter('fps', 30.0)
        # mp4v(=MPEG-4) 는 대부분의 OpenCV 빌드에 내장돼 있어 임베디드에서 안전.
        # 코덱이 없으면 XVID + .avi 로 바꿔볼 것.
        self.declare_parameter('fourcc', 'mp4v')
        self.declare_parameter('debug_log', True)
        # --- YOLO 밝기 버전 동시 녹화 ---
        # 카메라는 반사광 억제로 어둡게 찍고, object_node 는 YOLO 입력에만 감마로
        # 밝기를 올린다. 데이터셋도 두 조건을 다 확보하려고, 같은 프레임에서
        # raw(안 올림)와 bright(감마 적용) mp4 를 동시에 저장한다.
        # object_node._normalize_brightness 와 동일 로직(decision.yaml 기본 gamma=0.4).
        self.declare_parameter('save_raw', True)
        self.declare_parameter('save_bright', True)
        self.declare_parameter('yolo_gamma', 0.4)
        self.declare_parameter('yolo_clahe', False)

        self.subscribe_topic = str(self.get_parameter('subscribe_topic').value)
        self.output_dir = os.path.expanduser(str(self.get_parameter('output_dir').value))
        self.fps = float(self.get_parameter('fps').value)
        if self.fps <= 0.0:
            raise ValueError('fps must be greater than 0')
        self.fourcc_str = str(self.get_parameter('fourcc').value)
        self.debug_log = bool(self.get_parameter('debug_log').value)
        self.save_raw = bool(self.get_parameter('save_raw').value)
        self.save_bright = bool(self.get_parameter('save_bright').value)
        self.yolo_gamma = float(self.get_parameter('yolo_gamma').value)
        self.yolo_clahe = bool(self.get_parameter('yolo_clahe').value)
        if not self.save_raw and not self.save_bright:
            raise ValueError('save_raw 와 save_bright 가 둘 다 꺼져 있어 저장할 게 없습니다.')

        # 밝기 감마 LUT 는 한 번만 만들어 재사용. CLAHE 객체도 처음 쓸 때 생성.
        self._gamma_lut = None
        if abs(self.yolo_gamma - 1.0) > 1e-3 and self.yolo_gamma > 0.0:
            self._gamma_lut = np.array(
                [((i / 255.0) ** self.yolo_gamma) * 255 for i in range(256)],
                dtype=np.uint8)
        self._clahe = None

        os.makedirs(self.output_dir, exist_ok=True)

        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # (writer, output_path, transform) 3-튜플 목록. transform 이 None 이면 원본.
        self.channels = []
        if self.save_raw:
            self.channels.append({
                'writer': None,
                'path': os.path.join(self.output_dir, f'drive_{stamp}_raw.mp4'),
                'bright': False,
            })
        if self.save_bright:
            self.channels.append({
                'writer': None,
                'path': os.path.join(self.output_dir, f'drive_{stamp}_bright.mp4'),
                'bright': True,
            })
        self.frame_size = None
        self.frame_count = 0

        # QoS compatible with camera_node publisher.
        self.image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.subscription = self.create_subscription(
            CompressedImage,
            self.subscribe_topic,
            self.image_callback,
            self.image_qos,
        )

        channel_paths = ', '.join(
            f"{'bright' if ch['bright'] else 'raw'}={ch['path']}"
            for ch in self.channels)
        self.get_logger().info('\n'
            f'[Recorder Node] : subscribe={self.subscribe_topic} \n'
            f'[outputs] : {channel_paths} \n'
            f'[fps] : {self.fps} \n'
            f'[fourcc] : {self.fourcc_str} \n'
            f'[yolo_gamma] : {self.yolo_gamma}, [yolo_clahe] : {self.yolo_clahe} \n'
            'Recording... press Ctrl+C to stop and finalize the file(s).\n'
        )

    def _normalize_brightness(self, frame):
        # object_node._normalize_brightness 와 동일하게 감마 LUT + (옵션) CLAHE 적용.
        if self._gamma_lut is not None:
            frame = cv2.LUT(frame, self._gamma_lut)
        if self.yolo_clahe:
            if self._clahe is None:
                self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            l_ch = self._clahe.apply(l_ch)
            frame = cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
        return frame

    def image_callback(self, msg):
        frame = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warning('Failed to decode incoming frame')
            return

        if self.frame_size is None:
            height, width = frame.shape[:2]
            self.frame_size = (width, height)
            fourcc = cv2.VideoWriter_fourcc(*self.fourcc_str)
            for ch in self.channels:
                writer = cv2.VideoWriter(
                    ch['path'], fourcc, self.fps, self.frame_size)
                if not writer.isOpened():
                    self.get_logger().error(
                        f'Failed to open VideoWriter (fourcc={self.fourcc_str}, '
                        f'size={width}x{height}). Try fourcc=XVID with an .avi output.')
                    self.frame_size = None
                    return
                ch['writer'] = writer
            self.get_logger().info(
                f'VideoWriter opened: {width}x{height} @ {self.fps}fps '
                f'({len(self.channels)} file(s))')

        # 첫 프레임과 해상도가 다르면 VideoWriter 가 프레임을 버리므로 맞춰준다.
        if (frame.shape[1], frame.shape[0]) != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)

        for ch in self.channels:
            out = self._normalize_brightness(frame) if ch['bright'] else frame
            ch['writer'].write(out)
        self.frame_count += 1
        if self.debug_log and self.frame_count % 30 == 0:
            self.get_logger().info(f'Recorded {self.frame_count} frames')

    def destroy_node(self):
        try:
            for ch in self.channels:
                if ch['writer'] is not None:
                    ch['writer'].release()
                    ch['writer'] = None
                    self.get_logger().info(
                        f'Saved {self.frame_count} frames -> {ch["path"]}')
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
