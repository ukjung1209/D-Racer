"""슬라이딩 윈도우 체인 + coast (branch_hint==0) pytest — 기존 실패-모드 재현 필수.

밴드별 좌/우 앵커 경쟁·유령·유도를 전부 걷어내고, 하단 베이스에서 위로 따라 올라가는
체인으로 교체했다. 차선을 갑자기 잃으면 추측하지 않고 직전 방향을 짧게 유지(coast)한다.
lane_core 순수 함수만 쓴다(rclpy 불필요). LaneSim은 lane_node의 프레임 상태기계를
그대로 미러링해 프레임 시퀀스(코너·소실·복귀)를 결정적으로 검증한다.

실행:  cd src/inference && python3 -m pytest test/test_lane_core_windows.py -v
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from inference import lane_core   # noqa: E402


# 합성 마스크 규모 (decision.yaml lane_node 기본값과 동일 규모)
W = 320
H = 115
NUM_BANDS = 10
HALF = 90                # lane_half_width_px (반폭 초기값/base_half)
MARGIN = 40              # window_margin_px
MIN_PIX = 15             # cluster_min_pixels (체인 윈도우 점유 하한)
MAX_GAP = 2              # chain_max_gap
COAST_MAX = 8            # coast_max_frames
MATCH_TOL = 45           # base_match_tol_px
HIST_TOL = 0.25          # hist_width_tol
HALF_ALPHA = 0.2         # half_width_est_alpha
ANGLE_MAX = 0.3          # angle_max_delta


def _draw_line(mask, x_bottom, x_top=None, thickness=8, rows=None):
    """(x_bottom→x_top) 경로를 따라 폭 ~thickness 세로 라인. rows=(y0,y1)이면 그 행만.

    y=0(위=먼 곳)에서 x_top, y=H-1(아래=가까운 곳)에서 x_bottom. x_top 생략 시 수직선.
    """
    if x_top is None:
        x_top = x_bottom
    y_lo, y_hi = (0, H) if rows is None else rows
    for y in range(y_lo, y_hi):
        frac = y / (H - 1)                       # 0=위, 1=아래
        x = int(round(x_top + (x_bottom - x_top) * frac))
        x0 = max(0, min(W - thickness, x))
        mask[y, x0:x0 + thickness] = 255


def _cx(x0, thickness=8):
    return x0 + (thickness - 1) / 2.0


class LaneSim:
    """lane_node의 branch_hint==0 프레임 상태기계 미러(순수). step(mask)→발행 dict.

    프레임 간 상태는 명시된 5종뿐: prev_base_left/right, half_est, coast_count,
    직전 발행 offset/angle(+슬루용 prev_pub_angle). 그 외 상태 없음.
    """

    def __init__(self, base_left=None, base_right=None, half_est=HALF):
        self.prev_base_left = base_left
        self.prev_base_right = base_right
        self.half_est = float(half_est)
        self.coast_count = 0
        self.last_pub_offset = None
        self.last_pub_angle = None
        self.last_pub_conf = 0.0
        self.prev_pub_angle = None

    def step(self, mask):
        base_left, base_right = lane_core.find_bases(
            mask, self.half_est, HIST_TOL,
            self.prev_base_left, self.prev_base_right, MATCH_TOL)
        bands, centers, new_half, cbl, cbr = [], [], self.half_est, None, None
        if base_left is not None or base_right is not None:
            bands, centers, new_half, cbl, cbr = lane_core.analyze_chains(
                mask, W, NUM_BANDS, base_left, base_right,
                MARGIN, MIN_PIX, MAX_GAP, self.half_est, float(HALF), HALF_ALPHA)

        out = {'bands': bands, 'centers': centers, 'base': (base_left, base_right)}
        if len(centers) >= 2:
            ys = [c[0] for c in centers]
            xs = [c[1] for c in centers]
            ws = [c[2] for c in centers]
            offset, angle = lane_core.fit_lane_line(ys, xs, ws, W)
            offset = float(np.clip(offset, -1.0, 1.0))
            angle = float(np.clip(angle, -1.0, 1.0))
            angle, slewed = lane_core.slew_limit(angle, self.prev_pub_angle, ANGLE_MAX)
            conf = float(len(centers) / NUM_BANDS)
            self.half_est = new_half
            if cbl is not None:
                self.prev_base_left = cbl
            if cbr is not None:
                self.prev_base_right = cbr
            self.coast_count = 0
            self.last_pub_offset = offset
            self.last_pub_angle = angle
            self.last_pub_conf = conf
            self.prev_pub_angle = angle
            out.update(detected=True, offset=offset, angle=angle,
                       confidence=conf, state='TRACKING', slewed=slewed)
            return out

        # 베이스 없음 또는 유효 center < 2 → coast.
        self.coast_count += 1
        det, off, ang, conf, expired = lane_core.coast_decision(
            self.coast_count, COAST_MAX, self.last_pub_offset,
            self.last_pub_angle, self.last_pub_conf)
        if expired:
            self.coast_count = 0
            self.prev_base_left = None
            self.prev_base_right = None
            self.last_pub_offset = None
            self.last_pub_angle = None
            self.last_pub_conf = 0.0
            self.prev_pub_angle = None
            out.update(detected=False, offset=0.0, angle=0.0,
                       confidence=0.0, state='LOST')
        else:
            self.prev_pub_angle = ang
            out.update(detected=det, offset=off, angle=ang,
                       confidence=conf, state='COAST%d' % self.coast_count)
        return out


# ================================================================== #
#  1. 직선 2라인: 베이스 2개, 두 체인 완주, center 수직, offset≈0, angle≈0
# ================================================================== #
def test_two_straight_lines():
    mask = np.zeros((H, W), np.uint8)
    _draw_line(mask, 76)                         # 좌 cx≈79.5
    _draw_line(mask, 236)                        # 우 cx≈239.5
    sim = LaneSim()
    out = sim.step(mask)
    assert out['detected'] and out['state'] == 'TRACKING'
    assert out['base'][0] is not None and out['base'][1] is not None, '베이스 2개 실패'
    valid = [b for b in out['bands'] if b['valid']]
    assert len(valid) >= 9, f'두 체인 완주 실패: {len(valid)}'
    for b in valid:
        assert b['left'] is not None and b['right'] is not None, f'단측 밴드: {b}'
        assert abs(b['center'] - 159.5) < 3.0, f'center 수직 이탈: {b["center"]}'
    assert abs(out['offset']) < 0.05, f'offset≈0 실패: {out["offset"]}'
    assert abs(out['angle']) < 0.05, f'angle≈0 실패: {out["angle"]}'


# ================================================================== #
#  2. 급커브 단측 (핵심 — 기존 플립 재현 조건): 좌 라인만 +20px/프레임, 중앙 넘어감
# ================================================================== #
def test_single_side_sweep_across_center_no_flip():
    """좌 라인 하나가 프레임마다 +20px씩 이동해 화면 중앙(160)을 넘어가는 10프레임.

    앵커 경쟁이 없으므로 우 체인은 한 번도 생기지 않고(우측 점 0건), side 반전 0건,
    center = 좌 체인 ± half_est로만 만들어진다.
    """
    sim = LaneSim(base_left=100.0, base_right=None)   # 좌만 이미 획득한 상태로 진입
    for f in range(10):
        line_x0 = 100 + 20 * f                        # 100 … 280 (160 넘어감)
        mask = np.zeros((H, W), np.uint8)
        _draw_line(mask, line_x0)
        out = sim.step(mask)
        assert out['detected'], f'frame {f}: 검출 실패'
        assert out['base'][1] is None, f'frame {f}: 우 베이스 생성(플립 씨앗) {out["base"]}'
        valid = [b for b in out['bands'] if b['valid']]
        assert valid, f'frame {f}: 유효 밴드 없음'
        for b in valid:
            assert b['right'] is None, f'frame {f}: 우 체인 점 발생(플립) {b}'
            assert b['left'] is not None
        # center = clip(좌 체인 + half_est, 0, W) (단측 공식; 경계에서 클립)
        b0 = valid[0]
        expect = float(np.clip(b0['left'] + sim.half_est, 0.0, W))
        assert abs(b0['center'] - expect) < 1.5, \
            f'frame {f}: 단측 center 공식 이탈 {b0}'


# ================================================================== #
#  3. 상단 잡음 (기존 쌍안정 재현 조건): 체인 창 밖 잡음은 어떤 체인에도 기여 안 함
# ================================================================== #
def test_top_noise_ignored_by_chains():
    mask = np.zeros((H, W), np.uint8)
    _draw_line(mask, 76)                         # 좌 라인 cx≈79.5
    _draw_line(mask, 236)                        # 우 라인 cx≈239.5
    mask[0:33, 10:30] = 255                      # 상단 잡음 덩어리(좌 체인 창 [39,120] 밖)
    sim = LaneSim()
    out = sim.step(mask)
    valid = [b for b in out['bands'] if b['valid']]
    assert len(valid) >= 9
    for b in valid:
        # 좌 체인은 잡음(≈18)에 끌리지 않고 차선(≈79.5)에 머문다.
        assert b['left'] is None or abs(b['left'] - 79.5) < 6.0, f'좌 체인 잡음 오염: {b}'
        assert b['right'] is None or abs(b['right'] - 239.5) < 6.0, f'우 체인 오염: {b}'
        assert abs(b['center'] - 159.5) < 4.0, f'center 오염: {b["center"]}'
    # 잡음 x(≈18)이 어떤 체인 점에도 등장하지 않는다.
    assert not any(b['valid'] and b['left'] is not None and b['left'] < 40
                   for b in out['bands']), '잡음이 좌 체인에 배정됨'


# ================================================================== #
#  4. 돌연 소실 → coast → 복귀: 빈 구간 직전값 발행, 복귀 시 정상 재개·coast 리셋
# ================================================================== #
def test_loss_then_coast_then_recover():
    def straight():
        m = np.zeros((H, W), np.uint8)
        _draw_line(m, 76)
        _draw_line(m, 236)
        return m

    sim = LaneSim()
    good = sim.step(straight())
    assert good['detected'] and sim.coast_count == 0
    off0, ang0 = good['offset'], good['angle']

    # 3프레임 빈 마스크 → coast: 직전 offset/angle 그대로, detected=True, conf 감쇠.
    empty = np.zeros((H, W), np.uint8)
    prev_conf = good['confidence']
    for n in (1, 2, 3):
        out = sim.step(empty)
        assert out['detected'], f'coast {n}: detected=False(정지)'
        assert out['state'] == 'COAST%d' % n
        assert abs(out['offset'] - off0) < 1e-9, 'coast offset이 직전값과 다름'
        assert abs(out['angle'] - ang0) < 1e-9, 'coast angle이 직전값과 다름'
        assert out['confidence'] < prev_conf, 'coast confidence 감쇠 안 됨'
        prev_conf = out['confidence']
        assert sim.coast_count == n

    # 복귀 프레임: 정상 재개, coast_count 리셋.
    rec = sim.step(straight())
    assert rec['detected'] and rec['state'] == 'TRACKING'
    assert sim.coast_count == 0, 'coast_count 리셋 실패'


def test_coast_decision_pure():
    CD = lane_core.coast_decision
    # 유지: coast_count ≤ max, 직전값 있음 → 직전값 발행 + 0.7^n 감쇠.
    det, off, ang, conf, expired = CD(1, 8, 0.4, -0.2, 1.0)
    assert det and not expired and off == 0.4 and ang == -0.2
    assert abs(conf - 0.7) < 1e-9
    _, _, _, conf3, _ = CD(3, 8, 0.4, -0.2, 1.0)
    assert abs(conf3 - 0.7 ** 3) < 1e-9
    # 만료: coast_count > max → 미검출.
    assert CD(9, 8, 0.4, -0.2, 1.0) == (False, 0.0, 0.0, 0.0, True)
    # 직전값 없음(cold) → 즉시 만료(추측 금지).
    assert CD(1, 8, None, None, 0.0) == (False, 0.0, 0.0, 0.0, True)


# ================================================================== #
#  5. coast 만료: 빈 마스크 coast_max+1 → 만료부터 detected=False·상태 리셋
# ================================================================== #
def test_coast_expires_then_only_cold_start_reacquires():
    def straight():
        m = np.zeros((H, W), np.uint8)
        _draw_line(m, 76)
        _draw_line(m, 236)
        return m

    sim = LaneSim()
    sim.step(straight())                          # 정상 1프레임
    empty = np.zeros((H, W), np.uint8)
    for n in range(1, COAST_MAX + 1):             # coast_max 프레임까지 유지
        out = sim.step(empty)
        assert out['detected'], f'coast {n}: 조기 정지'
    # coast_max+1 프레임: 만료 → 미검출, 상태 전면 리셋.
    exp = sim.step(empty)
    assert not exp['detected'] and exp['state'] == 'LOST'
    assert sim.prev_base_left is None and sim.prev_base_right is None
    assert sim.last_pub_offset is None and sim.coast_count == 0

    # 만료 후 단측 라인만 오는 프레임 → 직전 베이스가 없어 재획득 실패(coast 만료 유지).
    single = np.zeros((H, W), np.uint8)
    _draw_line(single, 120)
    lost = sim.step(single)
    assert not lost['detected'], '단측 콜드스타트가 재획득됨(피크 2개만 허용 위반)'
    # 피크 2개 콜드스타트 프레임 → 재획득 성공.
    reacq = sim.step(straight())
    assert reacq['detected'] and reacq['state'] == 'TRACKING'


# ================================================================== #
#  6. 베이스 1피크 매칭: 직전 베이스 이내 → 진행, 밖 → coast
# ================================================================== #
def test_base_single_peak_matching():
    # (a) 직전 좌 베이스 100, 단일 피크가 118(|Δ|=18<45) → 좌로 매칭, 체인 진행.
    mask = np.zeros((H, W), np.uint8)
    _draw_line(mask, 115)                         # 라인 열 115..122 (히스토그램 argmax=115)
    bl, br = lane_core.find_bases(mask, HALF, HIST_TOL, 100.0, None, MATCH_TOL)
    assert bl is not None and br is None, f'좌 매칭 실패: {(bl, br)}'
    assert 113.0 <= bl <= 123.0, f'베이스가 라인 위가 아님: {bl}'

    # (b) 직전 좌 베이스 40, 같은 피크(118, |Δ|=78>45) → 매칭 실패 → 좌/우 모두 None.
    bl2, br2 = lane_core.find_bases(mask, HALF, HIST_TOL, 40.0, None, MATCH_TOL)
    assert bl2 is None and br2 is None, f'허용치 밖인데 매칭됨: {(bl2, br2)}'

    # (c) 직전 우 베이스가 더 가까우면 우로 매칭.
    bl3, br3 = lane_core.find_bases(mask, HALF, HIST_TOL, 20.0, 130.0, MATCH_TOL)
    assert bl3 is None and br3 is not None, f'우 매칭 실패: {(bl3, br3)}'


# ================================================================== #
#  7. 체인 교차 안전핀: 교차 높이 이후 두 체인 모두 점 없음
# ================================================================== #
def test_chain_cross_guard():
    # 좌 라인이 위로 갈수록 오른쪽으로 크게 휘어 우 라인(직선 240)을 넘어간다.
    mask = np.zeros((H, W), np.uint8)
    _draw_line(mask, 100, 280)                    # 좌: 아래 100 → 위 280 (우 라인 넘어감)
    _draw_line(mask, 240, 240)                    # 우: 직선 240
    bands, centers, _, _, _ = lane_core.analyze_chains(
        mask, W, NUM_BANDS, 100.0, 240.0, MARGIN, MIN_PIX, MAX_GAP,
        HALF, float(HALF), HALF_ALPHA)
    # 교차가 살아남은 밴드(both-visible인데 left>=right)는 없어야 한다.
    for b in bands:
        if b['valid'] and b['left'] is not None and b['right'] is not None:
            assert b['left'] < b['right'], f'교차 밴드 잔존: {b}'
    # 교차 이후 상단 밴드는 잘려 유효 밴드 수가 num_bands보다 작다(조기 종료).
    valid = [b for b in bands if b['valid']]
    assert len(valid) < NUM_BANDS, '교차 안전핀이 상단을 안 잘랐음'


# ================================================================== #
#  8. hugging byte-identical 회귀: 교체 전 스냅샷과 동일 출력
# ================================================================== #
def _analyze_hug(mask, hint, seed):
    return lane_core.analyze_bands(
        mask, W, NUM_BANDS, 40, HALF,
        branch_hint=hint, hug_bias=45, hug_line_seed=seed,
        hug_track_tol=40, cluster_min_pixels=30)


def test_hugging_byte_identical_snapshot():
    # 시나리오 A: 직선 2라인 (좌 58 / 우 250).
    mA = np.zeros((H, W), np.uint8)
    mA[:, 58:62] = 255
    mA[:, 250:254] = 255
    bands, hug = _analyze_hug(mA, 1, 58.0)
    assert hug == 58.0
    assert all(b['left'] == 58.0 and b['right'] is None and b['center'] == 103.0
               and b['weight'] == 0 and b['rejected'] == [] for b in bands)
    bands, hug = _analyze_hug(mA, -1, 253.0)
    assert hug == 253.0
    assert all(b['right'] == 253.0 and b['left'] is None and b['center'] == 208.0
               for b in bands)

    # 시나리오 B: 좌 곡선(아래58→위120) + 우 직선250 — 밴드별 값 스냅샷.
    mB = np.zeros((H, W), np.uint8)
    _draw_line(mB, 58, 120)
    _draw_line(mB, 250, 250)
    expect_left = [58.0, 61.0, 65.5, 70.75, 76.375, 82.1875,
                   88.09375, 94.046875, 100.0234375, 106.01171875]
    bands, hug = _analyze_hug(mB, 1, 58.0)
    assert abs(hug - 106.01171875) < 1e-6
    for b, ex in zip(bands, expect_left):
        assert abs(b['left'] - ex) < 1e-6, f'B 좌-hug 스냅샷 불일치: {b["left"]} != {ex}'
        assert abs(b['center'] - (ex + 45.0)) < 1e-6

    expect_right = [255.0, 256.0, 256.5, 256.75, 256.875, 256.9375,
                    256.96875, 256.984375, 256.9921875, 256.99609375]
    bands, hug = _analyze_hug(mB, -1, 253.0)
    assert abs(hug - 256.99609375) < 1e-6
    for b, ex in zip(bands, expect_right):
        assert abs(b['right'] - ex) < 1e-6, f'B 우-hug 스냅샷 불일치: {b["right"]} != {ex}'


# ================================================================== #
#  9. half_est: 실측 갱신·클램프
# ================================================================== #
def test_half_est_update_and_clamp():
    U = lane_core.update_half_est
    base = 90.0
    lo, hi = base * 0.7, base * 1.3               # 63, 117
    # 실측 반폭 150(> hi) → hi로 클램프 (alpha=1로 즉시 반영).
    assert U(90.0, 0.0, 300.0, base, 1.0) == hi
    # 실측 반폭 10(< lo) → lo로 클램프.
    assert U(90.0, 140.0, 160.0, base, 1.0) == lo
    # 정상 범위(90)는 그대로.
    assert abs(U(90.0, 70.0, 250.0, base, 0.2) - 90.0) < 1e-6

    # analyze_chains 경유: 좁은 두 라인 → half_est가 실측(≈20) 쪽으로 감소, 클램프 준수.
    mask = np.zeros((H, W), np.uint8)
    _draw_line(mask, 150)
    _draw_line(mask, 190)                         # 간격 40 → 실측 반폭 ≈20 (< lo)
    _, _, new_half, _, _ = lane_core.analyze_chains(
        mask, W, NUM_BANDS, 150.0, 190.0, MARGIN, MIN_PIX, MAX_GAP,
        90.0, base, HALF_ALPHA)
    assert new_half < 90.0, f'half_est 감소 실패: {new_half}'
    assert new_half >= lo - 1e-6, f'half_est 클램프 위반: {new_half}'


# ================================================================== #
#  10. hugging 종료 웜스타트: hug_line 시드 베이스로 즉시 재획득 (콜드스타트 회피)
# ================================================================== #
def test_hug_warmstart_bases():
    WB = lane_core.hug_warmstart_bases
    # 좌 hugging(+1): 좌=hug_line, 우=hug_line+2×half.
    bl, br = WB(1, 120.0, 90.0)
    assert bl == 120.0 and br == 120.0 + 180.0
    # 우 hugging(-1): 거울상.
    bl, br = WB(-1, 200.0, 90.0)
    assert br == 200.0 and bl == 200.0 - 180.0
    # hug_line None → 시드 없음(콜드스타트 폴백).
    assert WB(1, None, 90.0) == (None, None)


def test_warmstart_reacquires_without_cold_start():
    """hugging 종료 첫 프레임: 콜드스타트(피크 2개) 없이 hug_line 시드로 단측 라인을 즉시
    1피크 매칭해 TRACKING으로 잇는다(seed 없으면 같은 마스크가 coast 만료로 LOST)."""
    # 좌 라인 하나만 있는 갈림길 직후 프레임(우 차선 아직 안 보임).
    mask = np.zeros((H, W), np.uint8)
    _draw_line(mask, 118)                         # 좌 라인 cx≈121.5

    # (a) 콜드(prev_base 없음): 단측 1피크 + 직전 베이스 없음 → 재획득 실패 → LOST.
    cold = LaneSim(base_left=None, base_right=None)
    out_cold = cold.step(mask)
    assert not out_cold['detected'], '콜드스타트에서 단측 라인이 재획득됨(전제 어긋남)'

    # (b) 웜스타트: hint 종료 시 hug_line(120, 좌 hugging)으로 시드된 베이스.
    bl, br = lane_core.hug_warmstart_bases(1, 120.0, HALF)
    warm = LaneSim(base_left=bl, base_right=br)   # coast_count=0에서 시작
    out_warm = warm.step(mask)
    assert out_warm['detected'] and out_warm['state'] == 'TRACKING', \
        f'웜스타트 재획득 실패: {out_warm["state"]}'
    assert out_warm['base'][0] is not None, '좌 베이스 1피크 매칭 실패'
    assert warm.coast_count == 0
