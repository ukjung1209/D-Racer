"""decision_core 순수 함수 pytest (rclpy/inference_msgs/시간 불필요).

상태머신·스로틀 중재·표지판 선택 로직을 결정적으로 검증한다. ROS 없이 돌도록
inference 패키지 경로만 잡고 decision_core를 직접 import 한다(패키지 __init__은 비어 있음).

실행:  cd src/inference && python3 -m pytest test/test_decision_core.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from inference import decision_core   # noqa: E402


# ================================================================== #
#  신호등 상태머신
# ================================================================== #
def test_light_green_reaches_go_exactly_at_n():
    """STOPPED에서 green 연속 N(=3)번째에 정확히 GO, N-1번은 유지."""
    n = 3
    state, vg, vr = 'STOPPED', 0, 0
    # 1, 2번째: 아직 STOPPED (카운터만 증가)
    for i in range(1, n):
        state, vg, vr, transitioned = decision_core.update_light_state(
            state, vg, vr, True, False, n, 2)
        assert state == 'STOPPED', f'{i}번째에 조기 전이'
        assert transitioned is False
        assert vg == i
    # N번째: GO로 전이, 전이 순간 vote_green 리셋
    state, vg, vr, transitioned = decision_core.update_light_state(
        state, vg, vr, True, False, n, 2)
    assert state == 'GO'
    assert transitioned is True
    assert vg == 0


def test_light_green_miss_resets_counter():
    """green 연속 도중 miss가 끼면 카운터가 0으로 리셋돼 다시 처음부터."""
    n = 3
    state, vg, vr = 'STOPPED', 0, 0
    state, vg, vr, _ = decision_core.update_light_state(state, vg, vr, True, False, n, 2)
    state, vg, vr, _ = decision_core.update_light_state(state, vg, vr, True, False, n, 2)
    assert vg == 2
    # miss (green_seen=False) → 리셋
    state, vg, vr, transitioned = decision_core.update_light_state(
        state, vg, vr, False, False, n, 2)
    assert state == 'STOPPED'
    assert vg == 0
    assert transitioned is False


def test_light_go_to_stopped_on_red():
    """GO에서 red 연속 N(=2)번째에 STOPPED, N-1은 유지, miss면 리셋."""
    n = 2
    state, vg, vr = 'GO', 0, 0
    # 1번째: 유지
    state, vg, vr, transitioned = decision_core.update_light_state(
        state, vg, vr, False, True, 3, n)
    assert state == 'GO' and vr == 1 and transitioned is False
    # miss가 끼면 리셋
    state, vg, vr, _ = decision_core.update_light_state(state, vg, vr, False, False, 3, n)
    assert vr == 0
    # 다시 연속 2번 → STOPPED
    state, vg, vr, _ = decision_core.update_light_state(state, vg, vr, False, True, 3, n)
    state, vg, vr, transitioned = decision_core.update_light_state(
        state, vg, vr, False, True, 3, n)
    assert state == 'STOPPED'
    assert transitioned is True
    assert vr == 0


# ================================================================== #
#  동적 장애물 상태머신 (이벤트 + 시간 전이)
# ================================================================== #
def test_obstacle_full_scenario():
    """CRUISE→APPROACH→STOP→CRUISE 전체 시나리오 1회 (BOOST 없는 3상태)."""
    state = 'CRUISE'
    # CRUISE→APPROACH: red_votes가 임계(3) 도달
    state, rv, av, tr = decision_core.update_obstacle_event(state, 3, 0, 3, 1)
    assert state == 'APPROACH' and tr == 'CRUISE_TO_APPROACH' and rv == 0

    # APPROACH→STOP: aruco_votes가 임계(1) 도달
    state, rv, av, tr = decision_core.update_obstacle_event(state, 0, 1, 3, 1)
    assert state == 'STOP' and tr == 'APPROACH_TO_STOP' and av == 0

    # STOP→CRUISE: 마커가 clear_time 이상 사라짐(aruco_gone=True) → 바로 재출발
    state, tr = decision_core.update_obstacle_time(state, red_gone=False, aruco_gone=True)
    assert state == 'CRUISE' and tr == 'STOP_TO_CRUISE'


def test_obstacle_approach_returns_to_cruise_on_red_gone():
    """APPROACH에서 red_gone(빨강 오검출)이면 CRUISE로 복귀."""
    state, tr = decision_core.update_obstacle_time('APPROACH', red_gone=True, aruco_gone=False)
    assert state == 'CRUISE' and tr == 'APPROACH_TO_CRUISE'


def test_obstacle_no_transition_below_threshold():
    """표수가 임계 미만이면 전이 없음."""
    state, rv, av, tr = decision_core.update_obstacle_event('CRUISE', 2, 0, 3, 1)
    assert state == 'CRUISE' and tr is None and rv == 2
    # STOP인데 aruco_gone=False면 전이 없음
    state, tr = decision_core.update_obstacle_time('STOP', red_gone=True, aruco_gone=False)
    assert state == 'STOP' and tr is None


# ================================================================== #
#  스로틀 중재
# ================================================================== #
def test_arbitrate_light_stopped_forces_zero():
    assert decision_core.arbitrate_throttle(
        0.3, light_stopped=True, obstacle_state='CRUISE',
        slow_factor=0.7, fork_stopping=False) == 0.0


def test_arbitrate_obstacle_stop_forces_zero():
    assert decision_core.arbitrate_throttle(
        0.3, light_stopped=False, obstacle_state='STOP',
        slow_factor=0.7, fork_stopping=False) == 0.0


def test_arbitrate_obstacle_approach_scales_by_slow_factor():
    out = decision_core.arbitrate_throttle(
        0.3, light_stopped=False, obstacle_state='APPROACH',
        slow_factor=0.7, fork_stopping=False)
    assert abs(out - 0.3 * 0.7) < 1e-9


def test_arbitrate_fork_stopping_forces_zero():
    assert decision_core.arbitrate_throttle(
        0.3, light_stopped=False, obstacle_state='CRUISE',
        slow_factor=0.7, fork_stopping=True) == 0.0


def test_arbitrate_no_constraint_passes_cruise():
    out = decision_core.arbitrate_throttle(
        0.3, light_stopped=False, obstacle_state='CRUISE',
        slow_factor=0.7, fork_stopping=False)
    assert out == 0.3


# ================================================================== #
#  곡률 가변속도
# ================================================================== #
def test_curve_throttle_straight_returns_base():
    out = decision_core.curve_throttle(
        0.3, angle=0.0, offset=0.0, speed_ka=0.4, speed_ko=0.4, min_throttle=0.1)
    assert abs(out - 0.3) < 1e-9


def test_curve_throttle_large_angle_clamps_to_min():
    # 큰 angle이면 base·(1−큰값)이 음수/저값 → min_throttle로 클램프
    out = decision_core.curve_throttle(
        0.3, angle=10.0, offset=0.0, speed_ka=0.4, speed_ko=0.4, min_throttle=0.1)
    assert out == 0.1


# ================================================================== #
#  슬루 제한
# ================================================================== #
def test_slew_accel_limited_by_max_up():
    # 목표가 prev+max_up보다 크면 prev+max_up까지만
    out = decision_core.slew_throttle(0.5, prev=0.1, max_up=0.05)
    assert abs(out - 0.15) < 1e-9


def test_slew_decel_is_immediate():
    # 감속(target<prev)은 즉시 target
    out = decision_core.slew_throttle(0.05, prev=0.3, max_up=0.05)
    assert out == 0.05


def test_slew_within_max_up_passes_target():
    out = decision_core.slew_throttle(0.12, prev=0.1, max_up=0.05)
    assert out == 0.12


# ================================================================== #
#  표지판 선택
# ================================================================== #
def test_pick_sign_excludes_low_conf_and_area():
    dets = [
        ('left', 0.4, 0.05),    # conf 미달 (min 0.5)
        ('right', 0.9, 0.005),  # area 미달 (min 0.01)
    ]
    assert decision_core.pick_sign(dets, conf_min=0.5, area_min=0.01) is None


def test_pick_sign_selects_largest_area():
    dets = [
        ('left', 0.9, 0.02),
        ('right', 0.9, 0.05),   # 최대 area
        ('left', 0.9, 0.03),
        ('green', 0.99, 0.9),   # 신호등은 제외
    ]
    best = decision_core.pick_sign(dets, conf_min=0.5, area_min=0.01)
    assert best is not None
    assert best[0] == 'right'
    assert abs(best[2] - 0.05) < 1e-9


# ================================================================== #
#  표지판 방향 투표 (C-1): 1프레임 오검출 래치 방지
# ================================================================== #
def test_sign_vote_needs_two_of_recent_to_latch():
    """1프레임 검출로는 확정 안 되고, 같은 클래스 2회째에 방향 확정."""
    hist = []
    hist, d = decision_core.update_sign_vote(hist, 'left')
    assert d == 0, '1프레임 즉시 래치됨'
    hist, d = decision_core.update_sign_vote(hist, 'left')
    assert d == 1, '2회째에 좌 확정 안 됨'


def test_sign_vote_opposite_resets_counter():
    """반대 클래스가 끼면 카운터 리셋 → 다시 2회 모여야 확정."""
    hist = []
    hist, d = decision_core.update_sign_vote(hist, 'left')
    assert d == 0
    # 반대(right)가 들어와 좌 카운터 리셋. right는 이제 1회 → 미확정.
    hist, d = decision_core.update_sign_vote(hist, 'right')
    assert d == 0
    # right 2회째 → 우(-1) 확정.
    hist, d = decision_core.update_sign_vote(hist, 'right')
    assert d == -1


def test_sign_vote_flicker_never_latches():
    """좌/우가 매 프레임 번갈아 흔들리면 어느 쪽도 확정되지 않는다."""
    hist = []
    for name in ('left', 'right', 'left', 'right', 'left'):
        hist, d = decision_core.update_sign_vote(hist, name)
        assert d == 0, f'흔들리는 검출인데 {name}에서 확정됨'
