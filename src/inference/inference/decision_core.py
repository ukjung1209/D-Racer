"""decision_node의 순수 판단 로직 (ROS 의존 없음, 표준 라이브러리만).

decision_node.py에서 분리한 이유: rclpy/inference_msgs/시간(get_clock) 없이 pytest로
상태머신·스로틀 중재·표지판 선택 로직을 결정적으로 검증하기 위해서다. 여기 함수들은
상태를 self에 두지 않고 전부 인자로만 받는 순수 함수라, 값을 직접 넣어 테스트할 수 있다.

decision_node.py는 이 함수들을 얇게 감싼다. ROS 파라미터·타임스탬프 경과계산·로그 출력은
노드에 남기고(core는 transition 이름만 반환), 전이 조건/부등호/기본값은 노드와 100% 동일.

lane_core.py와 같은 스타일(순수 함수, 상태는 인자로만, 한국어 주석)을 따른다.
"""


def _clamp(value, low, high):
    """decision_node.clamp와 동일: max(low, min(high, value))."""
    return max(low, min(high, value))


# ------------------------------------------------------------------ #
#  신호등 상태머신 1스텝 (STOPPED↔GO)
# ------------------------------------------------------------------ #
def update_light_state(light_state, vote_green, vote_red, green_seen, red_seen,
                       green_votes_needed, red_votes_needed):
    """green/red 연속 관측 표수로 STOPPED↔GO를 1스텝 전이한다.

    STOPPED: green이 연속 green_votes_needed번이면 GO(초록 출발). 중간에 green을
             못 보면 vote_green=0으로 리셋. GO 상태의 vote_red는 항상 0.
    GO:      red가 연속 red_votes_needed번이면 STOPPED(빨강 정지). 미스면 리셋.
    transitioned: 이번 스텝에 상태가 바뀌었으면 True(로그는 노드가 새 상태로 판단).
    반환: (new_state, new_vote_green, new_vote_red, transitioned)
    """
    transitioned = False
    if light_state == 'STOPPED':
        vote_green = vote_green + 1 if green_seen else 0
        vote_red = 0
        if vote_green >= green_votes_needed:
            light_state = 'GO'
            vote_green = 0
            transitioned = True
    elif light_state == 'GO':
        vote_red = vote_red + 1 if red_seen else 0
        vote_green = 0
        if vote_red >= red_votes_needed:
            light_state = 'STOPPED'
            vote_red = 0
            transitioned = True
    return light_state, vote_green, vote_red, transitioned


# ------------------------------------------------------------------ #
#  동적 장애물: 이벤트(표수) 기반 전이 CRUISE→APPROACH→STOP
# ------------------------------------------------------------------ #
def update_obstacle_event(obstacle_state, red_votes, aruco_votes,
                          red_votes_needed, aruco_votes_needed):
    """이미 갱신된 연속 표수로 이벤트 전이 1스텝(표수 증가/타임스탬프는 노드).

    CRUISE:   red_votes가 red_votes_needed 이상이면 APPROACH(빨강 구간 감속), red_votes=0.
    APPROACH: aruco_votes가 aruco_votes_needed 이상이면 STOP(아루코 정지), aruco_votes=0.
    반환: (new_state, new_red_votes, new_aruco_votes, transition_name_or_None)
          transition: 'CRUISE_TO_APPROACH' | 'APPROACH_TO_STOP' | None
    """
    transition = None
    if obstacle_state == 'CRUISE':
        if red_votes >= red_votes_needed:
            obstacle_state = 'APPROACH'
            red_votes = 0
            transition = 'CRUISE_TO_APPROACH'
    elif obstacle_state == 'APPROACH':
        if aruco_votes >= aruco_votes_needed:
            obstacle_state = 'STOP'
            aruco_votes = 0
            transition = 'APPROACH_TO_STOP'
    return obstacle_state, red_votes, aruco_votes, transition


# ------------------------------------------------------------------ #
#  동적 장애물: 시간 기반 전이 STOP→CRUISE, APPROACH→CRUISE
# ------------------------------------------------------------------ #
def update_obstacle_time(obstacle_state, red_gone, aruco_gone):
    """경과시간 판정 결과(bool)로 시간 기반 전이 1스텝(경과계산은 노드).

    red_gone:   빨강이 red_clear_time_sec 이상 안 보임(관측 없음 포함 → 노드가 판정).
    aruco_gone: 마커를 봤었고(last_aruco_time 존재) clear_time_sec 이상 안 보임 → 노드가 판정.
    STOP:     aruco_gone이면 바로 CRUISE(마커 통과 → 재출발). BOOST 상태는 제거했다.
    APPROACH: red_gone이면 CRUISE(빨강 오검출 복귀).
    반환: (new_state, transition_name_or_None)
          transition: 'STOP_TO_CRUISE' | 'APPROACH_TO_CRUISE' | None
    """
    transition = None
    if obstacle_state == 'STOP':
        if aruco_gone:
            obstacle_state = 'CRUISE'
            transition = 'STOP_TO_CRUISE'
    elif obstacle_state == 'APPROACH':
        if red_gone:
            obstacle_state = 'CRUISE'
            transition = 'APPROACH_TO_CRUISE'
    return obstacle_state, transition


# ------------------------------------------------------------------ #
#  곡률 가변속도: 직선 base, 코너 감속 (freshness 판정은 노드)
# ------------------------------------------------------------------ #
def curve_throttle(base, angle, offset, speed_ka, speed_ko, min_throttle):
    """target = base·(1 − ka·|angle| − ko·|offset|)를 [min_throttle, base]로 클램프."""
    curve = speed_ka * abs(angle) + speed_ko * abs(offset)
    target = base * (1.0 - curve)
    return _clamp(target, min_throttle, base)


# ------------------------------------------------------------------ #
#  슬루 제한: 가속은 max_up까지, 감속은 즉시 (max_up 계산은 노드)
# ------------------------------------------------------------------ #
def slew_throttle(target, prev, max_up):
    """가속(target>prev)은 한 tick에 max_up까지만 올리고, 감속은 target 그대로."""
    if target > prev + max_up:
        return prev + max_up
    return target


# ------------------------------------------------------------------ #
#  스로틀 중재: cruise를 미션 목표들의 min으로 깎는다
# ------------------------------------------------------------------ #
def arbitrate_throttle(cruise, light_stopped, obstacle_state, slow_factor, fork_stopping):
    """cruise(가변속도)를 각 미션 목표의 min으로 깎아 최종 target을 낸다.

    light_stopped(신호등 STOPPED, enable 반영 후)면 0. obstacle STOP이면 0,
    APPROACH면 cruise·slow_factor. fork_stopping(갈림길 정지)이면 0.
    미션 enable 게이팅은 호출 측이 인자에 이미 반영한다(light_stopped bool,
    obstacle_state는 꺼졌으면 'CRUISE'를 넘겨 제약 없음).
    """
    target = cruise
    if light_stopped:
        target = min(target, 0.0)
    if obstacle_state == 'STOP':
        target = min(target, 0.0)
    elif obstacle_state == 'APPROACH':
        target = min(target, cruise * slow_factor)
    if fork_stopping:
        target = min(target, 0.0)
    return target


# ------------------------------------------------------------------ #
#  갈림길 표지판 선택: conf·크기 조건 통과 중 최대 area(가장 가까운) 표지판
# ------------------------------------------------------------------ #
def pick_sign(detections, conf_min, area_min):
    """detections는 (class_name, confidence, area_ratio) 튜플 리스트.

    left/right 중 confidence≥conf_min이고 area_ratio≥area_min인 것들에서 area 최대를
    고른다. 조건 통과가 없으면 None. 반환값은 뽑힌 (class_name, confidence, area_ratio).
    """
    best = None
    for name, conf, area in detections:
        if name not in ('left', 'right'):
            continue
        if conf < conf_min or area < area_min:
            continue
        if best is None or area > best[2]:
            best = (name, conf, area)
    return best


# ------------------------------------------------------------------ #
#  갈림길 표지판 방향 latch: 연속 투표로 곡선 오인식 래치 오발사 방지 (C-1)
# ------------------------------------------------------------------ #
def update_sign_latch(cur_dir, streak_dir, streak_count, observed, votes_needed):
    """게이트(conf·area) 통과 표지판 관측 1건으로 latch 방향을 1스텝 갱신한다.

    곡선을 화살표로 1~2프레임 오인식하면 즉시 래치돼 엉뚱한 hugging이 발동한다. 그래서
    '같은 방향이 votes_needed회 연속'일 때만 방향을 바꾼다(진입도, 이미 latch된 상태에서
    반대 방향으로의 변경도 동일하게 연속을 요구 → 1프레임 오인식으로 방향이 뒤집히지 않음).

    observed: 'left' | 'right' | None. None(게이트 통과 표지판 없는 콜백)이면 연속이
    끊긴 것이므로 streak을 리셋한다(스쳐가는 오인식은 연속을 못 채운다). cur_dir(현재 latch)는
    유지 — 해제는 시간/red-zone 경로 소관이고 이 함수는 latch 확정에만 관여한다.

    ※ streak 1스텝 = detection 콜백 1회. object_node가 process_every_n으로 10Hz 발행하므로
      votes_needed=3은 약 0.3초다. process_every_n을 바꾸면 이 시간 의미가 변한다.

    반환: (new_cur_dir, new_streak_dir, new_streak_count).
      direction 부호: +1=left, -1=right, 0=미latch.
    """
    if observed is None:
        return cur_dir, 0, 0               # 연속 끊김 → streak 리셋(latch는 유지)
    d = 1 if observed == 'left' else -1
    if streak_dir == d:
        streak_count += 1
    else:
        streak_dir = d                     # 반대/첫 관측 → 그 방향으로 1부터
        streak_count = 1
    if streak_count >= votes_needed and d != cur_dir:
        cur_dir = d
    return cur_dir, streak_dir, streak_count


# ------------------------------------------------------------------ #
#  갈림길 hugging 상태머신: 시작 → 유지(감속) → 종료 → 방향 남으면 재-hug(여러 번)
# ------------------------------------------------------------------ #
def hug_maneuver_step(is_hugging, elapsed, duration_sec,
                      latched_dir, hug_done, sign_gone):
    """continuous hugging + 지속시간 상한 상태머신 1스텝(시간 경과 계산은 노드).

    표지판 방향이 확정되면(latched_dir≠0) hugging을 시작하고, 시작 duration_sec가 지나면
    이번 hugging을 종료한다. '한 갈림길 1회' 제한은 없다 — 방향이 여전히 확정돼 있으면
    (표지판이 계속 보이면) 종료 직후 다음 tick에 곧바로 다시 hugging을 시작해 여러 번
    반복한다. 방향이 풀리면(latched_dir=0; 표지판 상실·빨강구간 해제는 노드 소관) IDLE로
    대기한다. hug_done은 잠금에 더는 쓰이지 않고(항상 False) 인터페이스 호환 위해 남겨둔다.

    반환: (action, hugging_active, new_hug_done).
      action:
        'START' : 이번 tick에 hugging 시작(노드가 시작시각 기록).
        'HOLD'  : hugging 진행 중(노드가 hint=±1 발행 + 감속).
        'END'   : duration 경과 → 이번 hugging 종료(노드가 hint=0). 방향이 남아 있으면 재-hug.
        'IDLE'  : 방향 미확정 → 할 일 없음.
      hugging_active: 이번 tick에 hugging을 유지하는가(hint·감속 게이트).
      sign_gone: 더는 참조하지 않는다(재무장 잠금이 없어짐). 인자만 유지.
    """
    if is_hugging:
        if elapsed >= duration_sec:
            return 'END', False, False    # 종료 즉시 재무장 → 방향 남아 있으면 재-hug
        return 'HOLD', True, hug_done
    if latched_dir != 0:                   # (제거) hug_done 1회 잠금 없이 방향 확정되면 재-hug
        return 'START', True, hug_done
    return 'IDLE', False, hug_done
