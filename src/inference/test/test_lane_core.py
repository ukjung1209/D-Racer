"""lane_core 순수 함수 pytest (rclpy/inference_msgs 불필요).

합성 마스크로 밴드 분석·앵커·재획득 로직을 결정적으로 검증한다. ROS 없이 돌도록
inference 패키지 경로만 잡고 lane_core를 직접 import 한다(패키지 __init__은 비어 있음).

실행:  cd src/inference && python3 -m pytest test/test_lane_core.py -v
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from inference import lane_core   # noqa: E402


# 공통 파라미터 (decision.yaml 기본값과 동일 규모)
W = 320
H = 240
NUM_BANDS = 10
MIN_PIXELS = 12
SPLIT_GAP = 40
HALF = 90
MAX_WIDTH = 60
MIN_CLUSTER_PIX = 30
GATE_RATIO = 0.4


def _analyze(mask, seed_left, seed_right, last_known_center=None,
             cluster_min_pixels=MIN_CLUSTER_PIX):
    """analyze_bands를 기본 파라미터로 호출하는 헬퍼(branch_hint=0, 평상 경로)."""
    return lane_core.analyze_bands(
        mask, W, NUM_BANDS, MIN_PIXELS, SPLIT_GAP, HALF,
        seed_center=W / 2.0, seed_left=seed_left, seed_right=seed_right,
        branch_hint=0, hug_bias=45, hug_line_seed=None, hug_track_tol=40,
        cluster_max_width_px=MAX_WIDTH, cluster_min_pixels=cluster_min_pixels,
        anchor_gate_ratio=GATE_RATIO,
        anchor_alpha_band=1.0, anchor_max_step_band_px=1e9,
        last_known_center=last_known_center, reacquire_margin=0.3)


def _vertical_lines(x_left, x_right, thickness=4):
    mask = np.zeros((H, W), np.uint8)
    mask[:, x_left:x_left + thickness] = 255
    mask[:, x_right:x_right + thickness] = 255
    return mask


# ------------------------------------------------------------------ #
#  (a) 폭 100px 가로 띠(십자)가 섞여도 앵커가 오염되지 않음
# ------------------------------------------------------------------ #
def test_cross_bar_does_not_pollute_anchors():
    # 두 세로 차선을 half_width(90)에 맞춰 배치: 좌=70, 우=250 → 중심 ~161.5, 반폭 90.
    mask = _vertical_lines(70, 250)
    # 한 밴드에 폭 90px 가로 띠(십자 가로획) 삽입 → cluster_max_width_px(60) 초과로 기각돼야.
    mask[120:126, 116:206] = 255
    bands, _ = _analyze(mask, seed_left=72.0, seed_right=252.0)
    valid = [b for b in bands if b['valid']]
    assert len(valid) >= 8, f'유효 밴드가 너무 적음: {len(valid)}'
    centers = [b['center'] for b in valid]
    # 가로획이 배정되면 center가 크게 흔들린다. 필터가 먹으면 전부 ~161 근처.
    assert max(centers) - min(centers) < 10, f'center 오염: {centers}'
    for b in valid:
        assert 155 <= b['center'] <= 168, f'center 이탈: {b["center"]}'


# ------------------------------------------------------------------ #
#  (b) 게이트 밖 클러스터만 있는 밴드는 valid=False
# ------------------------------------------------------------------ #
def test_out_of_gate_cluster_yields_invalid_band():
    # 라인 하나만 x=20에 있는데 앵커는 좌=200/우=290 → 게이트(90*0.4=36) 밖.
    mask = np.zeros((H, W), np.uint8)
    mask[:, 18:22] = 255
    bands, _ = _analyze(mask, seed_left=200.0, seed_right=290.0)
    valid = [b for b in bands if b['valid']]
    assert valid == [], f'게이트 밖 클러스터가 배정됨: {valid}'


# ------------------------------------------------------------------ #
#  (c) 한쪽 선 + last_known_center prior로 양가설이 올바른 쪽 선택
# ------------------------------------------------------------------ #
def test_single_line_two_hypothesis_uses_prior():
    # 세로 라인 하나만 x=120. 앵커는 없음(재획득 상황). prior에 따라 좌/우가 갈린다.
    mask = np.zeros((H, W), np.uint8)
    mask[:, 118:122] = 255

    # prior가 오른쪽(240)이면: 이 선(120)은 '왼쪽 차선' → center=120+90=210이 prior에 더 가까움
    bands_l, _ = _analyze(mask, seed_left=None, seed_right=None, last_known_center=240.0)
    valid_l = [b for b in bands_l if b['valid']]
    assert valid_l, 'prior=240에서 한쪽 라인 판정 실패'
    assert valid_l[0]['left'] is not None and valid_l[0]['right'] is None
    assert abs(valid_l[0]['center'] - 210.0) < 1.0

    # prior가 왼쪽(30)이면: 이 선(120)은 '오른쪽 차선' → center=120-90=30이 prior와 일치
    bands_r, _ = _analyze(mask, seed_left=None, seed_right=None, last_known_center=30.0)
    valid_r = [b for b in bands_r if b['valid']]
    assert valid_r, 'prior=30에서 한쪽 라인 판정 실패'
    assert valid_r[0]['right'] is not None and valid_r[0]['left'] is None
    assert abs(valid_r[0]['center'] - 30.0) < 1.0

    # prior가 정확히 가운데(120)면 양가설 점수 차 < margin → 판정 보류(무효)
    bands_amb, _ = _analyze(mask, seed_left=None, seed_right=None, last_known_center=120.0)
    assert [b for b in bands_amb if b['valid']] == [], '모호한 prior인데 배정됨'
