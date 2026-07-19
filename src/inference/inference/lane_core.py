"""lane_node의 순수 차선-분석 로직 (ROS 의존 없음, numpy만).

lane_node.py에서 분리한 이유: rclpy/inference_msgs 없이 pytest로 밴드 분석·체인
로직을 직접 검증하기 위해서다. 여기 함수들은 상태를 self에 두지 않고 전부 인자로만
받는 순수 함수라, 합성 마스크를 넣어 결정적으로 테스트할 수 있다.

── 구조 (branch_hint==0 = 평소 경로) ──
  베이스 탐색(find_bases) → 하단에서 위로 슬라이딩 윈도우 체인(analyze_chains) →
  밴드별 center → fit_lane_line(가중 1차 피팅) → offset/angle. 잃으면 coast_decision.

  * 정체성(좌/우)은 프레임당 베이스 탐색에서 x 순서로 단 한 번 정해지고, 그 뒤엔
    체인 소속으로 고정된다(밴드별 재판정·앵커 경쟁 없음 → 좌/우 플립 구조적 불가).
  * 프레임 간 상태는 lane_node가 소유한다: prev_base_left/right, half_est, coast_count,
    직전 발행 offset/angle(+슬루용 prev_pub_angle). core는 순수 함수.

hugging(branch_hint != 0)은 analyze_bands가 그대로 담당한다(지시 방향 최외곽 라인
하나 추종). 체인 로직 도입 전과 byte-identical.
"""

import numpy as np


# ------------------------------------------------------------------ #
#  앵커/추종선 갱신: EMA + 이동량 상한 (hugging이 사용)
# ------------------------------------------------------------------ #
def ema_step(anchor, meas, alpha, max_step):
    """추종선을 meas 쪽으로 EMA 갱신하되 한 스텝 이동량을 max_step으로 제한.

    커브에서 라인이 빠르게 움직여도 한 번에 튀지 않게 하면서, alpha만큼은 실제 관측을
    따라가게 한다. anchor가 None(첫 관측)이면 meas를 그대로 채택한다.
    """
    if anchor is None:
        return float(meas)
    step = float(np.clip(meas - anchor, -max_step, max_step))
    return float(anchor + alpha * step)


# ------------------------------------------------------------------ #
#  가로 밴드 → 라인 클러스터 추출 (hugging이 사용)
# ------------------------------------------------------------------ #
def extract_clusters(band, split_gap):
    """가로 밴드(2D 마스크 슬라이스)를 split_gap 틈으로 나눠 라인 클러스터 목록 반환.

    각 클러스터 dict:
      'cx'     : 멤버 열의 평균 x
      'x_min'  : 첫(최좌) 열 x  → 좌-hug의 바깥 모서리(분리섬 병합에도 진짜 차선 위)
      'x_max'  : 마지막(최우) 열 x → 우-hug의 바깥 모서리
      'width'  : 열 폭(마지막 열 − 첫 열)  → 십자 가로획(넓음) 판별용
      'pixels' : 실제 흰 픽셀 수(열별 nonzero 합, '열 수'가 아님) → 잡음 조각 판별용
    필터는 여기서 하지 않는다 — 호출 측에서 거른다.
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
            'x_min': float(cols[0]),
            'x_max': float(cols[-1]),
            'width': float(cols[-1] - cols[0]),
            'pixels': int(col_counts[cols].sum()),
        })
    return clusters


def valid_clusters(clusters, max_width_px, min_pixels):
    """배정 전 필터: 너무 넓은(십자 가로획) / 픽셀 적은(잡음) 클러스터 제거."""
    return [c for c in clusters
            if c['width'] <= max_width_px and c['pixels'] >= min_pixels]


# ------------------------------------------------------------------ #
#  베이스 탐색: 하단 1/3 히스토그램으로 좌/우 베이스 x를 잡는다 (프레임당 한 번)
# ------------------------------------------------------------------ #
def histogram_peaks(mask, half_width, width_tol):
    """마스크 하단 1/3 열 히스토그램에서 두 라인 피크(좌, 우)를 추정한다.

    최대 피크 + 그로부터 half_width 이상 떨어진 차순위 피크를 잡고, 두 피크 간격이
    차선폭 가정 2·half_width·(1±width_tol) 안일 때만 (left, right)를 반환한다.
    아니면 None(확정 보류). find_bases의 두-피크 판정이 이 함수를 이용한다.
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


def find_bases(mask, half_est, width_tol, prev_left, prev_right, match_tol):
    """하단 1/3 히스토그램으로 이 프레임 좌/우 베이스 x를 정한다(정체성 판정 지점).

    반환: (base_left, base_right). 각 값은 float 또는 None.
      · 피크 2개(간격 2·half_est·(1±width_tol) 이내) → 좌=작은 x, 우=큰 x. 콜드스타트 가능.
      · 피크 1개뿐 → 직전 베이스(prev_left/prev_right) 중 match_tol 이내로 가까운 쪽에
        매칭해 그 side 베이스로 삼는다. 둘 다 밖이거나 직전 베이스 없으면 → 그 side None.
      · 피크 0개 → (None, None).
    앵커 경쟁·유령·유도 없음. x 순서가 곧 정체성이다.
    """
    h = mask.shape[0]
    band = mask[(2 * h) // 3:, :]
    hist = np.count_nonzero(band, axis=0).astype(np.float64)
    if hist.max() <= 0:
        return None, None
    p1 = int(np.argmax(hist))
    masked = hist.copy()
    lo = max(0, p1 - int(half_est))
    hi = min(len(masked), p1 + int(half_est) + 1)
    masked[lo:hi] = 0.0
    if masked.max() > 0:
        p2 = int(np.argmax(masked))
        left, right = sorted((p1, p2))
        gap = right - left
        lo_gap = 2.0 * half_est * (1.0 - width_tol)
        hi_gap = 2.0 * half_est * (1.0 + width_tol)
        if lo_gap <= gap <= hi_gap:
            return float(left), float(right)
    # 단일(유효) 피크 → 직전 베이스에 매칭. 직전 베이스가 없으면 추측 금지.
    dl = abs(p1 - prev_left) if prev_left is not None else float('inf')
    dr = abs(p1 - prev_right) if prev_right is not None else float('inf')
    if dl <= match_tol and dl <= dr:
        return float(p1), None
    if dr <= match_tol:
        return None, float(p1)
    return None, None


# ------------------------------------------------------------------ #
#  실측 반폭 half_est
# ------------------------------------------------------------------ #
def update_half_est(half_est, left_x, right_x, base_half, alpha):
    """좌/우가 공존하는 윈도우의 실측 반폭 (right−left)/2로 half_est를 EMA 갱신.

    항상 base_half(=lane_half_width_px)의 [0.7, 1.3] 범위로 클램프한다. 실측이 그
    범위를 벗어나면 클램프 경계로 붙는다(엉뚱한 반폭으로 단측 center가 튀는 것 방지).
    """
    meas = (float(right_x) - float(left_x)) / 2.0
    new = (1.0 - alpha) * float(half_est) + alpha * meas
    lo = base_half * 0.7
    hi = base_half * 1.3
    return float(np.clip(new, lo, hi))


# ------------------------------------------------------------------ #
#  슬라이딩 윈도우 체인: 베이스에서 위로 라인을 따라 올라간다
# ------------------------------------------------------------------ #
def _slide_chain(mask, base_x, windows, margin, min_pixels, max_gap):
    """base_x에서 위로 각 윈도우(현재 중심 ± margin)를 훑어 라인 점을 기록한다.

    반환: {윈도우 인덱스 i: {'x': cx, 'pixels': n, 'y': y_mid}} — 점 잡힌 윈도우만.
    창 안 흰 픽셀 ≥ min_pixels면 픽셀 평균 x로 재중심화하고 다음 창으로. 미만이면 빈 창.
    빈 창이 max_gap 연속이면 체인 종료. 체인은 자기 창 안만 보므로 반대 라인·잡음은 무시.
    """
    pts = {}
    cx = float(base_x)
    w = mask.shape[1]
    empty = 0
    for i, (y0, y1, y_mid) in enumerate(windows):
        x_lo = int(max(0, round(cx - margin)))
        x_hi = int(min(w, round(cx + margin) + 1))
        col_counts = (np.count_nonzero(mask[y0:y1, x_lo:x_hi], axis=0)
                      if x_hi > x_lo else np.empty(0, int))
        n = int(col_counts.sum())
        if n >= min_pixels:
            xs = np.nonzero(col_counts)[0]
            cx = x_lo + float((xs * col_counts[xs]).sum() / col_counts[xs].sum())
            pts[i] = {'x': cx, 'pixels': n, 'y': y_mid}
            empty = 0
        else:
            empty += 1
            if empty >= max_gap:
                break
    return pts


def analyze_chains(mask, width, num_bands, base_left, base_right,
                   window_margin, min_pixels, chain_max_gap,
                   half_est, base_half, half_alpha, cross_guard=True):
    """좌/우 베이스에서 슬라이딩 윈도우 체인을 쌓아 밴드별 차선 center를 만든다.

    반환: (bands, centers, new_half_est, chain_base_left, chain_base_right).
      · bands: 디버그용 밴드 목록(아래→위). 유효 밴드 = {'valid':True,'y','left','right',
        'center','weight'}, 무효 = {'valid':False,'y','left':None,'right':None}.
      · centers: [(y, center, weight)] — fit_lane_line 입력.
      · new_half_est: 두 체인 공존 윈도우 실측으로 갱신·클램프한 반폭.
      · chain_base_*: 각 체인의 최하단 재중심 x(다음 프레임 prev_base). 체인이 없으면
        base_* 그대로(또는 None).

    center 규칙(윈도우 인덱스별):
      두 점 다 있으면 (l+r)/2, 한 점만 있으면 그 점 ± new_half_est(side는 체인 소속으로
      고정 — 밴드별 재판정 없음), 둘 다 없으면 그 높이는 center 없음.
    """
    h = mask.shape[0]
    band_h = max(1, h // num_bands)
    windows = []
    for i in range(num_bands):
        y1 = h - i * band_h
        y0 = max(0, y1 - band_h)
        if y1 <= 0:
            break
        windows.append((y0, y1, (y0 + y1) // 2))

    left_pts = (_slide_chain(mask, base_left, windows,
                             window_margin, min_pixels, chain_max_gap)
                if base_left is not None else {})
    right_pts = (_slide_chain(mask, base_right, windows,
                              window_margin, min_pixels, chain_max_gap)
                 if base_right is not None else {})

    # (안전핀) 같은 높이에서 좌 체인 중심 ≥ 우 체인 중심으로 교차하면 그 높이 이후 종료.
    if cross_guard and left_pts and right_pts:
        cut = None
        for i in range(len(windows)):
            lp, rp = left_pts.get(i), right_pts.get(i)
            if lp is not None and rp is not None and lp['x'] >= rp['x']:
                cut = i
                break
        if cut is not None:
            left_pts = {i: p for i, p in left_pts.items() if i < cut}
            right_pts = {i: p for i, p in right_pts.items() if i < cut}

    # half_est: 두 체인 공존 윈도우의 실측 반폭으로 갱신(단측 center에 쓰기 전에 확정).
    new_half = float(half_est)
    for i in range(len(windows)):
        lp, rp = left_pts.get(i), right_pts.get(i)
        if lp is not None and rp is not None:
            new_half = update_half_est(new_half, lp['x'], rp['x'], base_half, half_alpha)

    bands = []
    centers = []
    for i, (y0, y1, y_mid) in enumerate(windows):
        lp, rp = left_pts.get(i), right_pts.get(i)
        lx = lp['x'] if lp is not None else None
        rx = rp['x'] if rp is not None else None
        if lx is not None and rx is not None:
            center = (lx + rx) / 2.0
            weight = lp['pixels'] + rp['pixels']
        elif lx is not None:
            center = lx + new_half
            weight = lp['pixels']
        elif rx is not None:
            center = rx - new_half
            weight = rp['pixels']
        else:
            bands.append({'valid': False, 'y': y_mid, 'left': None, 'right': None})
            continue
        center = float(np.clip(center, 0.0, width))
        bands.append({'valid': True, 'y': y_mid, 'left': lx, 'right': rx,
                      'center': center, 'weight': weight})
        centers.append((y_mid, center, weight))

    chain_base_left = (left_pts[0]['x'] if 0 in left_pts
                       else (float(base_left) if base_left is not None else None))
    chain_base_right = (right_pts[0]['x'] if 0 in right_pts
                        else (float(base_right) if base_right is not None else None))
    return bands, centers, new_half, chain_base_left, chain_base_right


# ------------------------------------------------------------------ #
#  coast: 잃으면 가던 방향 유지, 추측 금지 (무한 coast·급발진 방지)
# ------------------------------------------------------------------ #
def coast_decision(coast_count, coast_max, last_offset, last_angle, last_conf):
    """베이스 없음/유효 center<2인 프레임의 발행값을 정한다(순수 함수).

    coast_count는 이번 프레임 포함 누적(1,2,…). 반환:
      (detected, offset, angle, confidence, expired).
      · coast_count ≤ coast_max 이고 직전 발행값이 있으면: 직전 offset/angle 그대로,
        detected=True, confidence = last_conf × 0.7^coast_count (감쇠), expired=False.
      · coast_max 초과(또는 직전 발행값 없음): detected=False, 0/0/0, expired=True →
        호출측이 coast·베이스·직전값을 리셋하고 정지(decision 계약)로 넘어간다.
    """
    if coast_count <= coast_max and last_offset is not None:
        conf = float(last_conf) * (0.7 ** coast_count)
        return True, float(last_offset), float(last_angle), float(conf), False
    return False, 0.0, 0.0, 0.0, True


# ------------------------------------------------------------------ #
#  hugging 종료 웜스타트: 추종하던 라인으로 다음 프레임 베이스를 시드한다
# ------------------------------------------------------------------ #
def hug_warmstart_bases(branch_hint, hug_line, half_est):
    """hugging 해제 첫 프레임의 좌/우 베이스 시드(prior)를 만든다.

    hugging은 어느 쪽 라인을 따라가고 있었는지 안다 — 이것이 강점이다. 콜드스타트(피크
    2개)를 기다리는 대신 그 라인 위치를 베이스로 시드하면, 갈림길 직후 두 차선이 아직
    선명하지 않아도 1피크 매칭으로 즉시 추적을 잇는다. 실제 관측이 오면 실측으로 대체된다.

    반환: (base_left, base_right).
      · branch_hint > 0 (좌 라인 hugging): 좌=hug_line, 우=hug_line + 2×half_est.
      · branch_hint < 0 (우 라인 hugging): 우=hug_line, 좌=hug_line − 2×half_est.
      · hug_line이 None이면 시드 없음 (None, None) → 콜드스타트로 폴백.
    """
    if hug_line is None:
        return None, None
    if branch_hint > 0:
        return float(hug_line), float(hug_line) + 2.0 * float(half_est)
    return float(hug_line) - 2.0 * float(half_est), float(hug_line)


# ------------------------------------------------------------------ #
#  발행 변화율 제한(angle 슬루)
# ------------------------------------------------------------------ #
def slew_limit(value, prev, max_delta):
    """발행 변화율 제한: |value − prev| > max_delta면 prev ± max_delta로 클램프.

    반환: (limited_value, was_limited). prev가 None(첫 발행/재획득)이면 그대로 통과.
    한 프레임 피팅선이 눕는 angle 폭주를 흡수해 조향 튐을 막는다.
    """
    if prev is None:
        return float(value), False
    lo, hi = prev - max_delta, prev + max_delta
    if value < lo:
        return float(lo), True
    if value > hi:
        return float(hi), True
    return float(value), False


# ------------------------------------------------------------------ #
#  hugging (branch_hint != 0): 지시 방향 최외곽 라인 하나 추종 (byte-identical)
# ------------------------------------------------------------------ #
def analyze_bands(mask, width, num_bands, split_gap, half_width,
                  branch_hint, hug_bias, hug_line_seed, hug_track_tol,
                  cluster_min_pixels):
    """hugging 전용 밴드 분석. branch_hint>0=좌 라인, <0=우 라인을 밴드마다 추종한다.

    (A-2) 대표값을 cx가 아니라 '바깥 모서리'로 잡는다: 좌-hug(>0)는 x_min, 우-hug(<0)는
      x_max. 분리섬 V자 빗변과 병합돼 뚱뚱해진 클러스터라도 바깥 모서리는 진짜 차선 위에
      남으므로 cx처럼 안쪽으로 끌려가지 않는다.
    (A-3c) 픽셀 적은 잡음/반사 조각은 후보에서 제외한다. 폭 필터는 적용 안 함(병합 클러스터를
      지우면 모서리를 못 쓴다).
    (A-3a) |nearest − running_line| > tol이면 그 밴드는 갱신 안 하고 running_line 유지
      (라인이 사라지고 섬 V가 등장 → 마지막 위치로 coast). (A-3b) tol 이내 갱신은 raw 대입
      대신 EMA로 흡수(모서리 지터 완화).

    이 함수는 체인 로직(branch_hint==0)과 무관하게 도입 전과 동일 출력을 낸다.
    """
    h = mask.shape[0]
    band_h = max(1, h // num_bands)
    bands = []
    running_line = hug_line_seed     # hugging 중 추종하는 라인 x (밴드 아래→위 연속)

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

        edge = 'x_min' if branch_hint > 0 else 'x_max'
        clusters = [c[edge] for c in raw if c['pixels'] >= cluster_min_pixels]
        if not clusters:
            bands.append({'valid': False, 'y': y_mid})
            continue
        if running_line is None:
            running_line = clusters[0] if branch_hint > 0 else clusters[-1]
        else:
            nearest = min(clusters, key=lambda c: abs(c - running_line))
            if abs(nearest - running_line) <= hug_track_tol:
                running_line = ema_step(running_line, nearest, 0.5, hug_track_tol)
        if branch_hint > 0:
            left_x = running_line
            center = running_line + hug_bias
        else:
            right_x = running_line
            center = running_line - hug_bias

        center = float(np.clip(center, 0.0, width))
        bands.append({
            'valid': True, 'y': y_mid,
            'left': left_x, 'right': right_x, 'center': center, 'weight': 0,
            'rejected': [],
        })

    return bands, running_line


# ------------------------------------------------------------------ #
#  유효 밴드 전체에 가중 1차 피팅 → offset / angle
# ------------------------------------------------------------------ #
def fit_lane_line(ys, xs, weights, width):
    """(y, lane_center)들에 가중 1차 직선을 맞춰 offset/angle을 뽑는다.

    2점(near/far) 차분보다 밴드 전체를 쓰므로 한 밴드 튐에 둔감하다. 부호·스케일은
    기존 규약과 동일하게 맞춘다:
      offset = (최하단 y의 x − width/2) / (width/2)
      angle  = (최상단 x − 최하단 x) / (width/2)   # = 기존 (far.center − near.center)
    가중치 합이 0이면 균등 가중으로 대체한다.
    """
    ys = np.asarray(ys, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones_like(ys)
    # np.polyfit의 w는 '잔차에 곱해지는' 값이라 목적함수엔 w²로 들어간다. 픽셀수 비례
    # 가중(의도)이 되려면 w=sqrt(픽셀수)를 넘겨야 실효 가중이 픽셀수에 비례한다.
    coeffs = np.polyfit(ys, xs, 1, w=np.sqrt(weights))
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    y_bottom = float(ys.max())
    y_top = float(ys.min())
    x_bottom = slope * y_bottom + intercept
    x_top = slope * y_top + intercept
    half = width / 2.0
    offset = (x_bottom - half) / half
    angle = (x_top - x_bottom) / half
    return offset, angle
