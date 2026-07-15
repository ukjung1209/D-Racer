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
SPLIT_GAP = 40
HALF = 90
MAX_WIDTH = 60
MIN_CLUSTER_PIX = 30
GATE_RATIO = 0.4


def _analyze(mask, seed_left, seed_right, last_known_center=None,
             cluster_min_pixels=MIN_CLUSTER_PIX):
    """analyze_bands를 기본 파라미터로 호출하는 헬퍼(branch_hint=0, 평상 경로)."""
    return lane_core.analyze_bands(
        mask, W, NUM_BANDS, SPLIT_GAP, HALF,
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


# ================================================================== #
#  hugging (A-1~A-3): 분리섬 V자에 끌리지 않고 바깥 차선 모서리 유지
# ================================================================== #
def _analyze_hug(mask, branch_hint, hug_line_seed):
    """analyze_bands를 hugging 경로로 호출하는 헬퍼(seed_left/right는 hug에서 미사용)."""
    return lane_core.analyze_bands(
        mask, W, NUM_BANDS, SPLIT_GAP, HALF,
        seed_center=W / 2.0, seed_left=None, seed_right=None,
        branch_hint=branch_hint, hug_bias=45, hug_line_seed=hug_line_seed,
        hug_track_tol=40,
        cluster_max_width_px=MAX_WIDTH, cluster_min_pixels=MIN_CLUSTER_PIX,
        anchor_gate_ratio=GATE_RATIO,
        anchor_alpha_band=1.0, anchor_max_step_band_px=1e9,
        last_known_center=None, reacquire_margin=0.3)


def _fork_mask(lane_left_x=58, island_top_x=150, island_bottom_x=93,
               island_w=30, right_x=250, thickness=4):
    """두 세로 차선 + '아래로 갈수록 좌 차선에 접근하는 분리섬 빗변' 합성 마스크.

    분리섬 왼쪽 모서리는 위(먼 곳)에선 island_top_x, 아래(가까운 곳)에선
    island_bottom_x. 아래 밴드에서 좌 차선(≈lane_left_x)과 split_gap 이내로 붙어
    하나의 뚱뚱한 클러스터로 병합된다(cx는 안쪽으로 끌리지만 x_min은 차선 위에 남음).
    """
    mask = np.zeros((H, W), np.uint8)
    mask[:, lane_left_x:lane_left_x + thickness] = 255
    mask[:, right_x:right_x + thickness] = 255
    for y in range(H):
        frac = y / (H - 1)          # 0=위(먼 곳), 1=아래(가까운 곳)
        edge = int(round(island_top_x + (island_bottom_x - island_top_x) * frac))
        mask[y, edge:edge + island_w] = 255
    return mask


def test_extract_clusters_exposes_outer_edges():
    """A-1: 병합된 바닥 밴드에서 x_min은 차선 모서리에 남고 cx는 섬 쪽으로 끌린다."""
    band = _fork_mask()[H - (H // NUM_BANDS):, :]      # 가장 아래(병합) 밴드
    clusters = lane_core.extract_clusters(band, SPLIT_GAP)
    left = min(clusters, key=lambda c: c['x_min'])
    assert 'x_min' in left and 'x_max' in left
    assert left['x_min'] <= 60, f'x_min이 차선을 벗어남: {left["x_min"]}'
    # 좌 차선과 분리섬이 한 클러스터로 병합 → cx는 x_min보다 한참 안쪽으로 끌려 있다.
    assert left['cx'] - left['x_min'] > 20, f'병합이 안 됨(cx≈x_min): {left}'
    assert left['x_max'] > 90


def test_hug_left_stays_on_outer_edge_when_island_merges():
    """A-2/A-3: 좌-hug에서 추종선이 분리섬(안쪽)에 끌리지 않고 좌 차선 모서리(~58) 유지."""
    mask = _fork_mask()
    bands, hug_line = _analyze_hug(mask, branch_hint=1, hug_line_seed=58.0)
    valid = [b for b in bands if b['valid']]
    assert len(valid) >= 8, f'유효 밴드가 너무 적음: {len(valid)}'
    assert abs(hug_line - 58.0) <= 3.0, f'추종선이 섬으로 끌림: {hug_line}'
    for b in valid:
        assert b['left'] is not None
        # cx를 썼다면 병합 밴드에서 left≈90+로 끌렸을 것. x_min이므로 차선 모서리에 머문다.
        assert b['left'] <= 66.0, f'left가 섬 쪽으로 끌림: {b["left"]}'


def test_hug_right_uses_x_max_edge():
    """A-2: 우-hug는 x_max(오른쪽 모서리)를 대표값으로 추종한다."""
    # 우 차선 250, 분리섬이 아래로 갈수록 오른쪽에서 접근(우 차선의 x_max로 방어).
    mask = _fork_mask(island_top_x=150, island_bottom_x=210, island_w=30)
    bands, hug_line = _analyze_hug(mask, branch_hint=-1, hug_line_seed=253.0)
    valid = [b for b in bands if b['valid']]
    assert len(valid) >= 8
    # 우 차선 모서리 x_max ≈ 253 유지(섬이 왼쪽에서 붙어도 안쪽으로 안 끌림).
    assert abs(hug_line - 253.0) <= 3.0, f'우-hug 추종선 오염: {hug_line}'
    for b in valid:
        assert b['right'] is not None and b['right'] >= 248.0


def test_hug_left_frame_sequence_no_inward_drift():
    """A-3: 분리섬이 프레임마다 더 접근해도 추종선은 바깥 차선 모서리에 머문다(드리프트 X).

    첫 프레임 시드는 평시 좌 앵커(B-1 상당)에서 온 58. 이후는 직전 추종선을 시드로 잇는다.
    """
    hug_line = 58.0
    lines = []
    for island_bottom in (120, 108, 96, 90, 86):   # 섬이 점점 좌 차선에 접근
        mask = _fork_mask(island_bottom_x=island_bottom)
        _, hug_line = _analyze_hug(mask, branch_hint=1, hug_line_seed=hug_line)
        lines.append(hug_line)
    for hl in lines:
        assert abs(hl - 58.0) <= 4.0, f'추종선 안쪽 드리프트: {lines}'


def test_hug_ignores_out_of_tol_jump():
    """A-3a: 추종 라인이 사라진 밴드(섬만 남음)에서 tol 밖 점프는 무시하고 위치 유지."""
    # 좌 차선을 아래 절반만 그리고, 섬은 x=180 부근(추종선 58에서 tol 40 밖).
    mask = np.zeros((H, W), np.uint8)
    mask[H // 2:, 58:62] = 255                      # 좌 차선(아래 절반만)
    mask[:, 180:210] = 255                          # 섬(세로, 항상 존재)
    _, hug_line = _analyze_hug(mask, branch_hint=1, hug_line_seed=58.0)
    # 위쪽 밴드엔 섬(180)만 있지만 |180-58|>40 → 갱신 무시, 추종선은 58 근처 유지.
    assert abs(hug_line - 58.0) <= 3.0, f'tol 밖 점프에 끌림: {hug_line}'


# ================================================================== #
#  B-2: hugging 해제(앵커 None) 후 히스토그램 재획득
# ================================================================== #
def test_reacquire_after_hug_release():
    """hint 1→0 전환으로 앵커를 비운 다음 프레임, 히스토그램 재획득이 두 라인을
    다시 잡아 정상 검출로 복귀한다. (동결된 낡은 앵커면 게이트에 전부 막히는 것도 확인.)"""
    mask = _vertical_lines(70, 250)                 # 정상 두 라인(좌 70, 우 250)

    # (1) hugging 동안 동결된 낡은 앵커(중앙 근처)면 실제 라인이 게이트 밖 → 검출 불능.
    bands_stale, _ = _analyze(mask, seed_left=160.0, seed_right=200.0)
    assert [b for b in bands_stale if b['valid']] == [], '낡은 앵커인데 검출됨'

    # (2) B-2: 앵커를 None으로 리셋 → 노드가 histogram_peaks로 재획득해 시드로 넣는다.
    peaks = lane_core.histogram_peaks(mask, HALF, 0.25)
    assert peaks is not None, '히스토그램 재획득 실패'
    left0, right0 = peaks
    assert abs(left0 - 70.0) <= 6.0 and abs(right0 - 250.0) <= 6.0
    bands_reacq, _ = _analyze(mask, seed_left=left0, seed_right=right0)
    valid = [b for b in bands_reacq if b['valid']]
    assert len(valid) >= 8, '재획득 후에도 검출 불능'
    assert abs(valid[0]['center'] - 160.0) <= 12.0
