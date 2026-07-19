"""lane_core hugging(branch_hint != 0) 경로 pytest (rclpy/inference_msgs 불필요).

hugging은 슬라이딩 윈도우 체인 도입 후에도 동작 불변이다(분리섬 V자에 끌리지 않고
바깥 차선 모서리 x_min/x_max를 추종). branch_hint==0 평소 경로는 체인 로직으로 교체돼
test_lane_core_windows.py가 따로 검증한다(앵커/게이트/양가설/재획득 테스트는 함께 삭제).

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


# ================================================================== #
#  hugging (A-1~A-3): 분리섬 V자에 끌리지 않고 바깥 차선 모서리 유지
# ================================================================== #
def _analyze_hug(mask, branch_hint, hug_line_seed):
    """analyze_bands를 hugging 경로로 호출하는 헬퍼(신규 hugging-only 시그니처)."""
    return lane_core.analyze_bands(
        mask, W, NUM_BANDS, SPLIT_GAP, HALF,
        branch_hint=branch_hint, hug_bias=45, hug_line_seed=hug_line_seed,
        hug_track_tol=40, cluster_min_pixels=MIN_CLUSTER_PIX)


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
