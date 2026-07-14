#!/usr/bin/env python3
"""mp4 녹화 영상에서 jpg 프레임을 뽑아 데이터셋을 만드는 스크립트 (PC 에서 실행).

차량에서 recorder_node 로 녹화한 mp4 를 scp 로 PC 에 옮긴 뒤 실행한다.
YOLO 라벨링에 바로 넣을 수 있는 frame_000001.jpg 형태로 저장한다.

예시:
    # 파일 하나에서 초당 5장 추출
    python3 extract_frames.py drive_20260714_120000.mp4 --fps 5

    # 폴더 안 모든 mp4 를 한 번에, 초당 2장
    python3 extract_frames.py ./recordings --fps 2 --out ./dataset

    # 모든 프레임 추출
    python3 extract_frames.py drive.mp4 --fps 0
"""
import argparse
import sys
from pathlib import Path

import cv2


def iter_video_files(input_path):
    path = Path(input_path)
    if path.is_dir():
        for ext in ('*.mp4', '*.avi', '*.mov', '*.mkv'):
            yield from sorted(path.glob(ext))
    elif path.is_file():
        yield path
    else:
        print(f'입력 경로를 찾을 수 없음: {input_path}', file=sys.stderr)


def extract_one(video_path, out_dir, target_fps, quality, start_index):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f'  열기 실패, 건너뜀: {video_path}', file=sys.stderr)
        return start_index

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if target_fps and target_fps > 0 and src_fps > 0:
        step = max(1, round(src_fps / target_fps))
    else:
        step = 1  # 모든 프레임 저장

    stem = video_path.stem
    saved = 0
    frame_idx = 0
    index = start_index
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            out_path = out_dir / f'{stem}_{index:06d}.jpg'
            cv2.imwrite(str(out_path), frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            index += 1
            saved += 1
        frame_idx += 1

    cap.release()
    print(f'  {video_path.name}: {saved}장 저장 (원본 {src_fps:.1f}fps, '
          f'{step}프레임마다 1장)')
    return index


def main():
    parser = argparse.ArgumentParser(
        description='mp4 영상에서 jpg 프레임을 추출한다.')
    parser.add_argument('input', help='mp4 파일 또는 mp4 들이 들어있는 폴더')
    parser.add_argument('--out', default='dataset',
                        help='jpg 저장 폴더 (기본: dataset)')
    parser.add_argument('--fps', type=float, default=5.0,
                        help='초당 뽑을 장수 (0 이면 모든 프레임, 기본: 5)')
    parser.add_argument('--quality', type=int, default=95,
                        help='jpg 품질 0~100 (기본: 95)')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = list(iter_video_files(args.input))
    if not videos:
        print('추출할 영상이 없음.', file=sys.stderr)
        sys.exit(1)

    print(f'{len(videos)}개 영상 -> {out_dir}/ 에 추출 시작 '
          f'(목표 {args.fps}fps)')
    index = 0
    for video in videos:
        index = extract_one(video, out_dir, args.fps, args.quality, index)

    print(f'완료: 총 {index}장 저장됨 -> {out_dir}/')


if __name__ == '__main__':
    main()
