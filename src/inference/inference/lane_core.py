"""lane_node의 순수 차선-분석 로직 (ROS 의존 없음, numpy만).

lane_node.py에서 분리한 이유: rclpy/inference_msgs 없이 pytest로 밴드 분석·앵커
로직을 직접 검증하기 위해서다. 여기 함수들은 상태를 self에 두지 않고 전부 인자로만
받는 순수 함수라, 합성 마스크를 넣어 결정적으로 테스트할 수 있다.

lane_node.py의 _analyze_bands는 이 모듈 함수를 얇게 감싸 ROS 파라미터만 읽어 넘긴다.
알고리즘(밴드+앵커 구조, 갈림길 hugging)은 그대로 유지하고, 아래 강건화만 얹었다.
"""

import numpy as np


# ------------------------------------------------------------------ #
#  앵커 갱신: EMA + 이동량 상한
# ------------------------------------------------------------------ #
def ema_step(anchor, meas, alpha, max_step):
    """앵커를 meas 쪽으로 EMA 갱신하되 한 스텝 이동량을 max_step으로 제한.

    커브에서 라인이 빠르게 움직여도 앵커가 한 번에 튀지 않게 하고(오검출 1회로 앵커가
    날아가는 것 방지), 그러면서도 alpha만큼은 실제 관측을 따라가게 한다.
    anchor가 None(첫 관측)이면 meas를 그대로 채택한다.
    """
    if anchor is None:
        return float(meas)
    step = float(np.clip(meas - anchor, -max_step, max_step))
    return float(anchor + alpha * step)


# ------------------------------------------------------------------ #
#  가로 밴드 → 라인 클러스터 추출
# ------------------------------------------------------------------ #
def extract_clusters(band, split_gap):
    """가로 밴드(2D 마스크 슬라이스)를 split_gap 틈으로 나눠 라인 클러스터 목록 반환.

    각 클러스터 dict:
      'cx'     : 멤버 열의 평균 x
      'width'  : 열 폭(마지막 열 − 첫 열)  → 십자 가로획(넓음) 판별용
      'pixels' : 실제 흰 픽셀 수(열별 nonzero 합, '열 수'가 아님) → 잡음 조각 판별용
    필터는 여기서 하지 않는다(hugging은 원본 클러스터가 필요) — 호출 측에서 거른다.
    """
    col_counts = np.count_nonzero(band, axis=0)      # 열별 픽셀 수
    xs = np.nonzero(col_counts)[0]                    # 픽셀이 있는 열(오름차순)
    if xs.size == 0:
        return []
    diffs = np.diff(xs)
    cuts = np.nonzero(diffs > split_gap)[0] if diffs.size > 0 else np.empty(0, int)
    bounds = [0] + [int(c) + 1 for c in cuts] + [xs.size]
    clusters = []
    for k in range(len(bounds) - 1):
        cols = xs[bounds[k]:bounds[k + 1]]
        clusters.append({
            'cx': float(cols.mean()),
            'width': float(cols[-1] - cols[0]),
            'pixels': int(col_counts[cols].sum()),
        })
    return clusters


def valid_clusters(clusters, max_width_px, min_pixels):
    """배정 전 필터: 너무 넓은(십자 가로획) / 픽셀 적은(잡음) 클러스터 제거."""
    return [c for c in clusters
            if c['width'] <= max_width_px and c['pixels'] >= min_pixels]


# ------------------------------------------------------------------ #
#  재획득 (mod 4): 앵커를 잃었을 때 좌/우를 다시 잡는다
# ------------------------------------------------------------------ #
def pick_side_hypothesis(x, half_width, width, last_known_center, margin):
    """한쪽 라인 x가 좌/우 어느 쪽인지 last_known_center prior로 판정.

    가설 L: 이 선이 왼쪽 → 차선중심 = x + half_width
    가설 R: 이 선이 오른쪽 → 차선중심 = x − half_width
    점수 = 0.4·(중심이 [0,width] 안:1/0) + 0.6·exp(−|중심 − prior| / half_width)
    두 가설 점수 차가 margin 이상일 때만 'left'/'right' 확정, 아니면 None(판정 보류).
    prior가 없으면(진짜 cold start) 판정 불가 → None.
    """
    if last_known_center is None:
        return None
    half = max(1.0, float(half_width))
    cl = x + half_width   # 가설 L의 차선중심
    cr = x - half_width   # 가설 R의 차선중심

    def score(c):
        in_bounds = 1.0 if 0.0 <= c <= width else 0.0
        return 0.4 * in_bounds + 0.6 * float(np.exp(-abs(c - last_known_center) / half))

    sl, sr = score(cl), score(cr)
    if abs(sl - sr) < margin:
        return None
    return 'left' if sl > sr else 'right'


def histogram_peaks(mask, half_width, width_tol):
    """마스크 하단 1/3 열 히스토그램에서 두 라인 피크(좌, 우)를 추정한다.

    최대 피크 + 그로부터 half_width 이상 떨어진 차순위 피크를 잡고, 두 피크 간격이
    차선폭 가정 2·half_width·(1±width_tol) 안일 때만 (left, right)를 반환한다.
    아니면 None(확정 보류) — 잘못된 앵커로 재출발하느니 한 프레임 더 기다린다.
    """
    h = mask.shape[0]
    band = mask[(2 * h) // 3:, :]
    hist = np.count_nonzero(band, axis=0).astype(np.float64)
    if hist.max() <= 0:
        return None
    p1 = int(np.argmax(hist))
    masked = hist.copy()
    lo = max(0, p1 - int(half_width))
    hi = min(len(masked), p1 + int(half_width) + 1)
    masked[lo:hi] = 0.0                        # 첫 피크 주변을 지우고 둘째 피크 탐색
    if masked.max() <= 0:
        return None
    p2 = int(np.argmax(masked))
    left, right = sorted((p1, p2))
    gap = right - left
    lo_gap = 2.0 * half_width * (1.0 - width_tol)
    hi_gap = 2.0 * half_width * (1.0 + width_tol)
    if lo_gap <= gap <= hi_gap:
        return float(left), float(right)
    return None


# ------------------------------------------------------------------ #
#  가로 밴드별로 좌/우 라인 분리 → 차선 중심 추정 (아래→위 전파)
# ------------------------------------------------------------------ #
def analyze_bands(mask, width, num_bands, min_pixels, split_gap,
                  half_width, seed_center, seed_left, seed_right,
                  branch_hint, hug_bias, hug_line_seed, hug_track_tol,
                  cluster_max_width_px, cluster_min_pixels, anchor_gate_ratio,
                  anchor_alpha_band=1.0, anchor_max_step_band_px=1e9,
                  last_known_center=None, reacquire_margin=0.3):
    """밴드마다 좌/우 라인을 나눠 차선 중심을 잡는다. (mask/width는 분석영상 좌표계)

    per-line 앵커(seed_left/seed_right)는 직전 프레임의 좌/우 라인 x다. 한쪽 라인만
    보여도 '중심 대비 좌/우'가 아니라 '어느 앵커에 더 가까운가'로 판정하므로, 커브에서
    라인이 화면중앙을 넘어와도 같은 라인으로 계속 인식된다.

    강건화(mod 1):
      · 클러스터 폭/픽셀수 필터로 십자 가로획·잡음 조각을 배정 전에 걸러낸다.
      · 앵커 게이트: 최근접이라도 |cx − anchor| > half_width·gate_ratio면 배정 안 함.
      · 좌/우가 같은 클러스터거나 left ≥ right 역전이면 '최대 틈 분리 폴백'(오염 통로)
        대신 그 밴드를 무효 처리한다. "잘못 배정하느니 밴드를 버린다."
    강건화(mod 2):
      · 밴드 간 앵커 전파를 raw 대입이 아니라 EMA+이동량 상한(ema_step)으로 한다.
        오검출 1회로 앵커가 튀지 않으면서 커브는 alpha만큼 따라간다.
    강건화(mod 4):
      · 앵커 없이 한 줄만 보이는 밴드는 '화면 중심 기준' 폴백(플립 원인)을 버리고,
        last_known_center prior로 양가설(이 선이 좌? 우?)을 판정한다. 확신이 없으면
        (점수 차 < reacquire_margin) 그 밴드를 무효 처리한다(판정 보류).
    hugging(branch_hint != 0) 경로는 필터/게이트/EMA 대상에서 제외 — 동작 그대로 보존.
    """
    h = mask.shape[0]
    band_h = max(1, h // num_bands)
    bands = []
    running_left = seed_left
    running_right = seed_right
    running_line = hug_line_seed     # hugging 중 추종하는 라인 x (밴드 아래→위 연속)
    gate = half_width * anchor_gate_ratio
    _ = seed_center                  # (mod 4) 화면중심 폴백 삭제로 미사용 — 시그니처만 유지

    for i in range(num_bands):
        y1 = h - i * band_h
        y0 = max(0, y1 - band_h)
        if y1 <= 0:
            break
        band = mask[y0:y1, :]
        y_mid = (y0 + y1) // 2
        raw = extract_clusters(band, split_gap)
        if not raw:
            bands.append({'valid': False, 'y': y_mid})
            continue

        left_x = right_x = None
        weight = 0

        if branch_hint != 0:
            # ── hugging: 원본(미필터) 클러스터로 기존 동작 그대로 보존 ──
            clusters = [c['cx'] for c in raw]
            if running_line is None:
                running_line = clusters[0] if branch_hint > 0 else clusters[-1]
            else:
                nearest = min(clusters, key=lambda c: abs(c - running_line))
                inner_jump = (
                    (branch_hint > 0 and nearest > running_line + hug_track_tol) or
                    (branch_hint < 0 and nearest < running_line - hug_track_tol))
                if not inner_jump:
                    running_line = nearest
            if branch_hint > 0:
                left_x = running_line
                center = running_line + hug_bias
            else:
                right_x = running_line
                center = running_line - hug_bias
        else:
            # ── 평소: 필터 통과 클러스터만, 앵커 게이트로 배정 ──
            clusters = valid_clusters(raw, cluster_max_width_px, cluster_min_pixels)
            if not clusters:
                bands.append({'valid': False, 'y': y_mid})
                continue

            if running_left is not None and running_right is not None:
                lc = min(clusters, key=lambda c: abs(c['cx'] - running_left))
                rc = min(clusters, key=lambda c: abs(c['cx'] - running_right))
                if lc is rc:
                    # 같은 클러스터가 양쪽 최근접 → 더 가까운 앵커로만 배정
                    if abs(lc['cx'] - running_left) <= abs(rc['cx'] - running_right):
                        rc = None
                    else:
                        lc = None
                lx = (lc['cx'] if lc is not None
                      and abs(lc['cx'] - running_left) <= gate else None)
                rx = (rc['cx'] if rc is not None
                      and abs(rc['cx'] - running_right) <= gate else None)
                if lx is not None and rx is not None:
                    if lx < rx:
                        left_x, right_x = lx, rx
                        weight = lc['pixels'] + rc['pixels']
                        center = (lx + rx) / 2.0
                    else:
                        bands.append({'valid': False, 'y': y_mid})   # 역전 → 버림
                        continue
                elif lx is not None:
                    left_x = lx
                    weight = lc['pixels']
                    center = lx + half_width
                elif rx is not None:
                    right_x = rx
                    weight = rc['pixels']
                    center = rx - half_width
                else:
                    bands.append({'valid': False, 'y': y_mid})   # 둘 다 게이트 밖
                    continue
            elif len(clusters) >= 2:
                # 앵커 없음 + 두 줄: 최좌/최우를 좌/우로(폭이 충분히 벌어졌을 때만).
                lc = min(clusters, key=lambda c: c['cx'])
                rc = max(clusters, key=lambda c: c['cx'])
                if rc['cx'] - lc['cx'] >= half_width:
                    left_x, right_x = lc['cx'], rc['cx']
                    weight = lc['pixels'] + rc['pixels']
                    center = (left_x + right_x) / 2.0
                else:
                    bands.append({'valid': False, 'y': y_mid})
                    continue
            else:
                # (mod 4) 앵커 없음 + 한 줄: last_known_center prior로 양가설 판정.
                # 화면 중심 기준 폴백(플립 원인)은 삭제. 확신 없으면 밴드 무효(판정 보류).
                boundary = clusters[0]['cx']
                pick = pick_side_hypothesis(
                    boundary, half_width, width, last_known_center, reacquire_margin)
                if pick == 'left':
                    left_x = boundary
                    weight = clusters[0]['pixels']
                    center = boundary + half_width
                elif pick == 'right':
                    right_x = boundary
                    weight = clusters[0]['pixels']
                    center = boundary - half_width
                else:
                    bands.append({'valid': False, 'y': y_mid})
                    continue

        center = float(np.clip(center, 0.0, width))
        # 본 라인만 앵커 갱신(EMA+이동량 상한), 안 보인 쪽은 직전값 유지(안정적 앵커).
        # hugging 경로는 running_left/right가 이후 밴드에 쓰이지 않으므로 갱신 생략.
        if branch_hint == 0:
            if left_x is not None:
                running_left = ema_step(
                    running_left, left_x, anchor_alpha_band, anchor_max_step_band_px)
            if right_x is not None:
                running_right = ema_step(
                    running_right, right_x, anchor_alpha_band, anchor_max_step_band_px)
        bands.append({
            'valid': True, 'y': y_mid,
            'left': left_x, 'right': right_x, 'center': center, 'weight': weight,
        })

    return bands, running_line


# ------------------------------------------------------------------ #
#  유효 밴드 전체에 가중 1차 피팅 → offset / angle (mod 3)
# ------------------------------------------------------------------ #
def fit_lane_line(ys, xs, weights, width):
    """(y, lane_center)들에 가중 1차 직선을 맞춰 offset/angle을 뽑는다.

    2점(near/far) 차분보다 밴드 전체를 쓰므로 한 밴드 튐에 둔감하다. 부호·스케일은
    기존 규약과 동일하게 맞춘다:
      offset = (최하단 y의 x − width/2) / (width/2)
      angle  = (최상단 x − 최하단 x) / (width/2)   # = 기존 (far.center − near.center)
    가중치 합이 0이면(예: hugging) 균등 가중으로 대체한다.
    """
    ys = np.asarray(ys, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.sum() <= 0:
        w = np.ones_like(ys)
    coeffs = np.polyfit(ys, xs, 1, w=w)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    y_bottom = float(ys.max())
    y_top = float(ys.min())
    x_bottom = slope * y_bottom + intercept
    x_top = slope * y_top + intercept
    half = width / 2.0
    offset = (x_bottom - half) / half
    angle = (x_top - x_bottom) / half
    return offset, angle
