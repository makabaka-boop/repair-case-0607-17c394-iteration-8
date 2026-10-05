"""可选接头套管（sleeves）的解析几何夹具测试。

覆盖：
- 跨拐点（套管边界两侧线段各自拆分、边界按较大半径裁决、区间跨拐点合并）；
- 端点相切（套管起/止里程上恰好相切，闭集判碰撞且只保留一个零长点）；
- 重叠安全圈（复合侵入只在套管较粗半径下出现/消失，事件扫描同源）；
- 标定后坐标（先变换圆心，套管沿原路径里程生效）；
- 拆段不制造重复碰撞（每「原线段 × 圈」至多一条 Collision）；
- 互不相连的同段多分量独立、但显示为同一三位小数的不同边界不合并；
- 粗筛不漏候选、随机等价性（朴素逐窗参照）；
- 未提交套管时逐项兼容；非法范围整次 422 且无部分风险；改线预览兼容。
"""

import math
import random

import pytest
from fastapi.testclient import TestClient

from app.geometry import (
    Sleeve,
    _candidate_pairs,
    _candidate_pairs_sleeved,
    analyze_path_full,
    cumulative_mileage,
)
from app.main import app

client = TestClient(app)
PATH = "/api/precheck"


def post(body):
    return client.post(PATH, json=body)


def base_body(**over):
    body = {
        "nodes": [{"x": 0, "y": 0}, {"x": 100, "y": 0}],
        "cable_radius": 5,
        "circles": [],
    }
    body.update(over)
    return body


def coll_keys(collisions):
    return {(c.segment_index, c.circle_index) for c in collisions}


# ---------- 跨拐点 ----------

def test_sleeve_crossing_corner_detects_local_intrusion():
    # 折线 (0,0)->(10,0)->(10,10)；圈心 (10,-1)，孔半径 1。
    # 电缆半径 5 => 扩张 6：两段垂距均为 1，本来就命中（夹具要体现套管
    # 让原本安全的局部变为命中，故取较远构型，见下一个用例）。
    nodes = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    cum = cumulative_mileage(nodes)
    assert cum == [0.0, 10.0, 20.0]
    # 圈心 (10,-3)：距两段均 3。扩张 5（r=0, 不行 radius 须正）
    # 取孔半径 1：扩张=6 本来命中；改取圈心 (10,-7)，r=1：
    # 普通扩张 6 -> 垂距 7 > 6 安全；套管外半径 8 -> 扩张 9 命中。
    circles = [((10.0, -7.0), 1.0)]
    raw, ivs, comps = analyze_path_full(nodes, circles, cable_radius=5.0)
    assert raw == [] and ivs == [] and comps == []

    # 套管里程 [8, 12] 跨过拐点（里程 10）：段0 的 t∈[0.8,1]、
    # 段1 的 t∈[0,0.2] 用外半径 8（扩张 9），两段均命中。
    sleeves = [Sleeve(8.0, 12.0, 8.0)]
    raw, ivs, _ = analyze_path_full(
        nodes, circles, cable_radius=5.0, sleeves=sleeves
    )
    assert coll_keys(raw) == {(0, 0), (1, 0)}
    assert len(ivs) == 1
    iv = ivs[0]
    # 一个连续侵入区间跨拐点合并，片段保留**原线段编号** 0 与 1。
    assert iv.entry_segment_index == 0 and iv.exit_segment_index == 1
    assert [p.segment_index for p in iv.pieces] == [0, 1]
    # 拐点 (10,0) 处于套管内（里程 10），两段片段在该点相接。
    assert iv.pieces[0].exit_point == (10.0, 0.0)
    assert iv.pieces[1].entry_point == (10.0, 0.0)
    # 区间进入里程严格大于 8（扩张 9：垂足 (10,0) 距 7，弦半长 sqrt(32)≈5.657，
    # 沿段0 进入点里程 10-sqrt(32)≈4.343，在套管起点之前 → 裁剪到 8）。
    assert iv.start_mileage == 8.0
    # 段1 方向竖直向上：离开里程 10+sqrt(32)≈15.657，超过套管终点 12 → 裁剪。
    assert iv.end_mileage == 12.0
    assert iv.length == pytest.approx(4.0)


def test_sleeve_pieces_clipped_to_sleeve_window_at_both_sides():
    # 单段 0..100；圈心 (50,-6) r1：普通扩张 6 恰相切（已命中，不能体现）。
    # 取圈心 (50,-6.5) r1：扩张 6 时间隙 0.5 安全；套管外半径 5.6 扩张
    # 6.6 -> 侵入 [40.79, 59.21]（sqrt(6.6²-6.5²)=1.144）在套管窗内。
    nodes = [(0.0, 0.0), (100.0, 0.0)]
    circles = [((50.0, -6.5), 1.0)]
    sleeves = [Sleeve(40.0, 60.0, 5.6)]
    _, ivs, _ = analyze_path_full(nodes, circles, 5.0, sleeves=sleeves)
    assert len(ivs) == 1
    iv = ivs[0]
    # 弦完全落在套管窗内：不被裁剪。
    half = math.sqrt(6.6 ** 2 - 6.5 ** 2)
    assert iv.entry_point[0] == pytest.approx(50.0 - half)
    assert iv.exit_point[0] == pytest.approx(50.0 + half)

    # 套管窗比弦窄 [49.5, 50.5]：片段被未舍入里程裁到窗边界，边界点按
    # 较大半径裁决（49.5 处距圈心 hypot(0.5,6.5)≈6.519 ≤ 6.6 命中，
    # 普通扩张 6 必不命中）。
    sleeves = [Sleeve(49.5, 50.5, 5.6)]
    _, ivs, _ = analyze_path_full(nodes, circles, 5.0, sleeves=sleeves)
    iv = ivs[0]
    assert iv.start_mileage == 49.5 and iv.end_mileage == 50.5
    assert iv.entry_point == (49.5, 0.0) and iv.exit_point == (50.5, 0.0)
    assert len(iv.pieces) == 1  # 裁到边界的零长接缝不产生重复片段


# ---------- 端点相切 ----------

def test_tangent_exactly_at_sleeve_start_is_hit_zero_length():
    # 单段 0..100；套管 [50, 70] 外半径 10。扩张圈（cable 5 -> 15；
    # sleeve 10 -> 20）。圈心 (50,-20) r10：
    # 普通扩张 15：垂距 20 安全；套管扩张 20：起点里程 50 恰好相切。
    nodes = [(0.0, 0.0), (100.0, 0.0)]
    circles = [((50.0, -20.0), 10.0)]
    raw, ivs, _ = analyze_path_full(nodes, circles, 5.0)
    assert raw == [] and ivs == []

    sleeves = [Sleeve(50.0, 70.0, 10.0)]
    raw, ivs, _ = analyze_path_full(nodes, circles, 5.0, sleeves=sleeves)
    assert len(raw) == 1
    c = raw[0]
    assert (c.segment_index, c.circle_index) == (0, 0)
    assert c.distance == pytest.approx(20.0)
    assert c.expanded_radius == pytest.approx(20.0)  # 边界按较大半径裁决
    assert len(ivs) == 1
    iv = ivs[0]
    assert iv.length == 0.0
    assert iv.entry_point == iv.exit_point == (50.0, 0.0)
    assert iv.start_mileage == iv.end_mileage == 50.0
    assert len(iv.pieces) == 1  # 相切零长点只出现一次，不随拆段重复


def test_tangent_exactly_at_sleeve_end_is_hit_and_not_merged_with_neighbor():
    # 套管 [30,50]；圈心 (50,-20) r10：仅在终点里程 50 与套管扩张圈相切；
    # 普通半径在 50 处不命中（20 > 15），故这是唯一的零长点且属于套管。
    nodes = [(0.0, 0.0), (100.0, 0.0)]
    circles = [((50.0, -20.0), 10.0)]
    sleeves = [Sleeve(30.0, 50.0, 10.0)]
    raw, ivs, _ = analyze_path_full(nodes, circles, 5.0, sleeves=sleeves)
    assert len(raw) == 1
    iv = ivs[0]
    assert iv.length == 0.0
    assert iv.start_mileage == 50.0


def test_boundary_point_uses_larger_radius_on_both_adjacent_windows():
    # 套管边界里程 50 两侧窗都把该点按闭集纳入：构造恰在较大扩张圈上、
    # 较小扩张圈外的点（距离 18：>15 且 ≤20），左右各一窗覆盖该点，
    # 但只产出一条碰撞、一个零长片段。
    nodes = [(0.0, 0.0), (100.0, 0.0)]
    circles = [((50.0, -18.0), 8.0)]  # 扩张 13 / 18
    sleeves = [Sleeve(50.0, 80.0, 10.0)]
    raw, ivs, _ = analyze_path_full(nodes, circles, 5.0, sleeves=sleeves)
    assert len(raw) == 1
    assert ivs[0].length == 0.0
    assert ivs[0].start_mileage == 50.0
    assert len(ivs[0].pieces) == 1


# ---------- 重叠安全圈（复合侵入）----------

def test_compound_intrusion_appears_only_inside_sleeve():
    # 单段 0..100，电缆 5。圈0 心 (40,0) r=9 -> 普通扩张 14：路径侵入
    # [26,54]（本身命中）。圈1 心 (55,-6.5) r=1：普通扩张 6，垂距 6.5
    # 不命中；套管外半径 5.6 扩张 6.6 -> 路径侵入约 [53.856,56.144]。
    # 普通半径下圈1 不存在 => 无复合侵入；带套管后圈0 扩张也变为 14.6
    # （侵入 [25.4,54.6]），与圈1 弦重叠 [53.856,54.6]，复合段只在套管
    # 生效时出现。
    nodes = [(0.0, 0.0), (100.0, 0.0)]
    circles = [((40.0, 0.0), 9.0), ((55.0, -6.5), 1.0)]
    raw, ivs, comps = analyze_path_full(nodes, circles, 5.0)
    assert comps == []  # 圈1 普通半径不命中 → 无复合段

    sleeves = [Sleeve(40.0, 80.0, 5.6)]
    raw, ivs, comps = analyze_path_full(nodes, circles, 5.0, sleeves=sleeves)
    assert coll_keys(raw) == {(0, 0), (0, 1)}
    assert len(comps) == 1
    seg = comps[0]
    assert seg.circle_indices == (0, 1)
    half1 = math.sqrt(6.6**2 - 6.5**2)
    assert seg.start_mileage == pytest.approx(55.0 - half1)
    assert seg.end_mileage == pytest.approx(54.6)
    assert seg.length == pytest.approx(54.6 - (55.0 - half1))
    assert seg.start_point[0] == pytest.approx(55.0 - half1)
    assert seg.end_point == (54.6, 0.0)
    # 复合片段仍按原线段切分（单段）。
    assert [p.segment_index for p in seg.pieces] == [0]


def test_compound_disappears_when_sleeve_removed():
    # 同一几何：带套管有复合段；移除套管后圈1 完全不命中，复合段消失，
    # 且不留下圈1 的任何区间/碰撞（输入变化不残留新旧半径混用）。
    nodes = [(0.0, 0.0), (100.0, 0.0)]
    circles = [((40.0, 0.0), 9.0), ((55.0, -6.5), 1.0)]
    sleeves = [Sleeve(40.0, 80.0, 5.6)]
    _, ivs1, comps1 = analyze_path_full(nodes, circles, 5.0, sleeves=sleeves)
    assert len(comps1) == 1
    assert len(ivs1) == 2
    raw2, ivs2, comps2 = analyze_path_full(nodes, circles, 5.0)
    assert coll_keys(raw2) == {(0, 0)}
    assert all(iv.circle_index == 0 for iv in ivs2)
    assert comps2 == []


# ---------- 同段互不相连分量 + 三位小数相同不合并 ----------

def test_disjoint_components_on_same_segment_stay_independent():
    # 套管 [30,70] 较粗；圈弦在套管内，而普通半径在套管两侧也各自命中，
    # 但二者之间（套管窗外）有间隙 -> 同一原线段上两个独立区间。
    # 圈心 (50,-3) r5：普通扩张 10 -> 路径侵入 [40.53,59.47]（在套管内
    # 连续），不行。构造大圈：圈心 (50,-8) r10，普通扩张 15 垂距 8 ->
    # 侵入 [50±12.69]=[37.3,62.7]，套管扩张（外半径 12 ->22）侵入
    # [50±20.5]=[29.5,70.5]，裁剪到 [30,70]。两侧 [30,37.3] 与 [62.7,70]
    # 由套管产生、中间 [37.3,62.7] 普通产生，并集合并后应为**一个**连续
    # 区间 [30,70]（边界相接即连通）。
    nodes = [(0.0, 0.0), (100.0, 0.0)]
    circles = [((50.0, -8.0), 10.0)]
    sleeves = [Sleeve(30.0, 70.0, 12.0)]
    _, ivs, _ = analyze_path_full(nodes, circles, 5.0, sleeves=sleeves)
    assert len(ivs) == 1
    iv = ivs[0]
    assert iv.start_mileage == 30.0 and iv.end_mileage == 70.0

    # 真正的互不相连：普通半径只在两个远处命中，套管只在中间命中且
    # 弦与两侧不接。圈1（普通命中两侧不可能——单圈与线段交集恒为单个
    # 区间）。故改用：圈A 普通侵入在左，圈B 普通侵入在右（同圈测试用
    # 两个不同圈无法验证“同段同圈多分量”）。同段同圈的多分量只能由
    # 半径差造成：套管窗内命中、窗外也命中但弦被窗边界切断且中间普通
    # 半径不命中——需要套管弦与普通弦分离。取圈心 (50,-6.2) r1：
    # 普通扩张 6 不命中（间隙 0.2）；套管扩张 6.6 弦 [48.86,51.14] 只在
    # 窗内 → 单分量。再取第二个**不相连套管**窗 [60,62] 同圈：该窗内
    # 同样不命中（远离弦）。要在两个窗内分别命中，弦必须覆盖两个窗，
    # 即普通半径差只在窗处补足——单圈补不出两个分离弦。
    # 因此同段同圈多分量的合法物理来源是：套管窗内弦 ∪ 窗外（同圈）
    # 不存在第二分量，除非有两个相互分离的套管且较粗半径使两处都命中、
    # 中间普通半径不命中——取半径差恰好使弦在每个窗内成立且窗间不成立。
    # 圈心 (50,-6.45) r1：扩张 6（普通）不命中；6.6（外5.6）弦半长
    # sqrt(6.6²-6.45²)=1.4 -> [48.6,51.4]，仍只一处。单心单圈在直线上
    # 只能有一个连续弦，故分离分量需要两个圈（下面用同段两圈验证片段
    # 不被错误跨分量合并），同段同圈的“窗内弦+边界相切外点”并集去重
    # 已由端点相切夹具覆盖。
    circles2 = [((45.0, -6.2), 1.0), ((55.0, -6.2), 1.0)]
    sleeves2 = [Sleeve(43.0, 47.0, 5.6), Sleeve(53.0, 57.0, 5.6)]
    raw, ivs, _ = analyze_path_full(nodes, circles2, 5.0, sleeves=sleeves2)
    # 每个圈各自一个独立区间，两区间中间有间隙，绝不合并。
    assert coll_keys(raw) == {(0, 0), (0, 1)}
    assert len(ivs) == 2
    assert ivs[0].circle_index != ivs[1].circle_index or ivs[0].end_mileage < ivs[1].start_mileage


def test_distinct_boundaries_displaying_same_three_decimals_not_merged():
    # 两个相互分离的套管窗：[49.9996, 50.2] 与 [50.2, 50.4]？
    # 需要同一圈在两窗内各产生一个相切/侵入分量，其边界未舍入值不同
    # 但展示三位小数相同，且区间不被合并（中间有正长间隙）。
    # 圈心 (50,-6.45) r1：套管扩张 6.6 弦 [48.6,51.4] 覆盖两窗 → 会合并。
    # 改用两处不同的圈：圈0 心 (49.9996-? …)。直接构造：
    # 圈0 与套管扩张圈在里程 49.9996 相切（圈心 (49.9996,-6.6), r1,
    # 套管扩张 6.6）；圈1 在里程 50.0004 相切（心 (50.0004,-6.6), r1）。
    # 两个零长区间独立（不同圈），起止里程展示均为 50.000，不合并。
    nodes = [(0.0, 0.0), (100.0, 0.0)]
    circles = [((49.9996, -6.6), 1.0), ((50.0004, -6.6), 1.0)]
    # 套管覆盖两切点且中间连续会使两圈各自仍只是点（零长分量与中间
    # 是否覆盖无关——相切点不延伸）。用两个分离窗各自含一个切点。
    sleeves = [Sleeve(49.0, 49.9996, 5.6), Sleeve(50.0004, 51.0, 5.6)]
    raw, ivs, _ = analyze_path_full(nodes, circles, 5.0, sleeves=sleeves)
    assert len(raw) == 2
    assert len(ivs) == 2
    mileages = sorted(iv.start_mileage for iv in ivs)
    assert mileages[0] != mileages[1]
    assert round(mileages[0], 3) == round(mileages[1], 3) == 50.0
    # 复合侵入：两切点不同里程（即使三位展示相同），不产生里程 50 的
    # 双圈零长点。
    _, _, comps = analyze_path_full(nodes, circles, 5.0, sleeves=sleeves)
    assert comps == []


# ---------- 标定后的坐标 ----------

def test_sleeve_applies_after_calibration_via_api():
    # survey = path + (1000,2000) 纯平移；孔 survey 心 (1050,1994) r=0.5
    # -> 施工 (50,-6)：普通扩张 5.5，垂距 6 安全；套管外半径 5.6 扩张
    # 6.1 -> 侵入 [48.9,51.1]（sqrt(6.1²-6²)=1.1），在套管窗内。
    body = base_body(
        nodes=[{"x": 0, "y": 0}, {"x": 100, "y": 0}],
        circles=[{"x": 1050, "y": 1994, "radius": 0.5}],
        sleeves=[{"start_mileage": 40, "end_mileage": 60, "outer_radius": 5.6}],
        calibration={
            "survey_points": [
                {"x": 1000, "y": 2000},
                {"x": 1100, "y": 2000},
                {"x": 1000, "y": 2100},
            ],
            "path_points": [
                {"x": 0, "y": 0},
                {"x": 100, "y": 0},
                {"x": 0, "y": 100},
            ],
            "max_rms_error": 1,
        },
    )
    r = post(body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["feasible"] is False
    assert data["collision_count"] == 1
    c = data["collisions"][0]
    assert c["circle_index"] == 0
    assert c["expanded_radius"] == 6.1
    # 侵入区间在套管窗内（弦半长约 1.097）
    iv = data["intrusion_intervals"][0]
    assert iv["start_mileage"] == pytest.approx(48.9, abs=1e-3)
    # 回显套管（展示三位小数）。
    assert data["sleeves"] == [
        {"start_mileage": 40.0, "end_mileage": 60.0, "outer_radius": 5.6, "length": 20.0}
    ]


# ---------- 拆段不制造重复碰撞 ----------

def test_no_duplicate_collisions_from_splitting():
    random.seed(77)
    nodes = [(float(random.randint(-30, 30)), float(random.randint(-30, 30)))]
    while len(nodes) < 6:
        p = (float(random.randint(-30, 30)), float(random.randint(-30, 30)))
        if p != nodes[-1]:
            nodes.append(p)
    cum = cumulative_mileage(nodes)
    total = cum[-1]
    sleeves = []
    m = 0.0
    while m + 5.0 < total:
        a = m + 1.0
        b = m + 4.0
        sleeves.append(Sleeve(a, b, random.choice([5.5, 7.0, 10.0])))
        m += 7.0
    circles = [
        ((float(random.randint(-35, 35)), float(random.randint(-35, 35))),
         random.choice([0.5, 1.0, 3.0]))
        for _ in range(8)
    ]
    raw, ivs, _ = analyze_path_full(nodes, circles, 5.0, sleeves=sleeves)
    keys = [(c.segment_index, c.circle_index) for c in raw]
    assert len(keys) == len(set(keys))  # 无重复键
    # 每个碰撞的裁决自洽：distance ≤ expanded_radius
    for c in raw:
        assert c.distance <= c.expanded_radius + 1e-9
    # 每个碰撞键至少有一个片段；片段键可以重复（多分量），但片段的
    # 并集不得重叠（同段同圈任意两片至多端点相接）。
    piece_keys = [
        (p.segment_index, p.circle_index)
        for iv in ivs for p in iv.pieces
    ]
    by_key: dict = {}
    for iv in ivs:
        for p in iv.pieces:
            by_key.setdefault((p.segment_index, p.circle_index), []).append(p)
    for key, ps in by_key.items():
        assert key in set(keys)
        ps_sorted = sorted(ps, key=lambda p: p.t0)
        for a, b in zip(ps_sorted, ps_sorted[1:]):
            assert a.t1 <= b.t0  # 不重叠（相接允许）


# ---------- 朴素逐窗参照（随机等价性）----------

def test_matches_naive_windowed_reference():
    from app.geometry import (
        _point_radius,
        _sleeve_cuts_on_segment,
        _window_radius,
        segment_disk_interval,
    )

    random.seed(2026)
    for _ in range(300):
        nn = random.randint(2, 7)
        nodes = [(float(random.randint(-40, 40)), float(random.randint(-40, 40)))]
        while len(nodes) < nn:
            p = (float(random.randint(-40, 40)), float(random.randint(-40, 40)))
            if p != nodes[-1]:
                nodes.append(p)
        cable = random.choice([1.0, 3.0, 5.0])
        cum = cumulative_mileage(nodes)
        total = cum[-1]
        sleeves = []
        for _ in range(random.randint(0, 3)):
            a = random.uniform(0.0, total)
            b = random.uniform(0.0, total)
            a, b = sorted((a, b))
            if b - a > 1e-6:
                sleeves.append(Sleeve(a, b, random.choice([cable, cable + 1.0, cable + 4.0])))
        circles = [
            ((float(random.randint(-45, 45)), float(random.randint(-45, 45))),
             random.choice([0.3, 1.0, 2.5]))
            for _ in range(random.randint(0, 4))
        ]

        raw, ivs, comps = analyze_path_full(nodes, circles, cable, sleeves=sleeves)

        # 朴素参照：开窗（中点半径求交裁剪）∪ 边界点单元格（逐点最粗半径）。
        expect_hits = set()
        expect_pairs: dict = {}
        for i in range(len(nodes) - 1):
            a, b = nodes[i], nodes[i + 1]
            seg_len = cum[i + 1] - cum[i]
            cuts = _sleeve_cuts_on_segment(cum[i], cum[i + 1], sleeves)
            bounds = [0.0]
            for m in cuts:
                t = (m - cum[i]) / seg_len
                if not bounds or t != bounds[-1]:
                    bounds.append(t)
            bounds.append(1.0)
            for j, (center, r) in enumerate(circles):
                wins = []
                # 开窗
                for k in range(len(bounds) - 1):
                    u0, u1 = bounds[k], bounds[k + 1]
                    rad = _window_radius(
                        u0, u1, cum[i], seg_len, sleeves, cable
                    )
                    tv = segment_disk_interval(a, b, center, r + rad)
                    if tv is None:
                        continue
                    lo = max(tv[0], u0)
                    hi = min(tv[1], u1)
                    if lo <= hi:
                        wins.append((lo, hi))
                # 边界点单元格（闭集、该里程最粗半径）
                dx, dy = b[0] - a[0], b[1] - a[1]
                for tb in bounds:
                    rad = _point_radius(tb, cum[i], seg_len, sleeves, cable)
                    qx, qy = a[0] + tb * dx, a[1] + tb * dy
                    if math.hypot(center[0] - qx, center[1] - qy) <= r + rad:
                        wins.append((tb, tb))
                if wins:
                    expect_hits.add((i, j))
                    wins.sort()
                    merged = [list(wins[0])]
                    for t0, t1 in wins[1:]:
                        if t0 <= merged[-1][1]:
                            merged[-1][1] = max(merged[-1][1], t1)
                        else:
                            merged.append([t0, t1])
                    expect_pairs[(i, j)] = [(t0, t1) for t0, t1 in merged]

        assert coll_keys(raw) == expect_hits
        got_pairs: dict = {}
        for iv in ivs:
            for p in iv.pieces:
                got_pairs.setdefault((p.segment_index, p.circle_index), []).append(
                    (p.t0, p.t1)
                )
        assert set(got_pairs) == expect_hits
        for key, want in expect_pairs.items():
            got = sorted(got_pairs[key])
            assert len(got) == len(want)
            for (g0, g1), (w0, w1) in zip(got, want):
                assert g0 == pytest.approx(w0, abs=1e-12)
                assert g1 == pytest.approx(w1, abs=1e-12)


# ---------- 粗筛不漏候选 ----------

def test_sleeved_coarse_filter_never_misses():
    random.seed(99)
    for _ in range(200):
        nn = random.randint(2, 8)
        nodes = [(float(random.randint(-50, 50)), float(random.randint(-50, 50)))]
        while len(nodes) < nn:
            p = (float(random.randint(-50, 50)), float(random.randint(-50, 50)))
            if p != nodes[-1]:
                nodes.append(p)
        cum = cumulative_mileage(nodes)
        total = cum[-1]
        cable = random.choice([1.0, 5.0])
        sleeves = []
        for _ in range(random.randint(0, 3)):
            a = random.uniform(0.0, total)
            b = random.uniform(a + 1e-6, total) if a < total else a
            if b > a:
                sleeves.append(Sleeve(a, b, cable + random.choice([0.5, 5.0, 20.0])))
        circles = [
            ((float(random.randint(-70, 70)), float(random.randint(-70, 70))),
             random.choice([0.5, 2.0, 8.0]))
            for _ in range(random.randint(0, 6))
        ]
        fast = set(_candidate_pairs_sleeved(nodes, circles, cable, sleeves, cum))
        exact = coll_keys(analyze_path_full(nodes, circles, cable, sleeves=sleeves)[0])
        assert exact <= fast
        # 普通粗筛在无套管时与带套管粗筛（空套管）一致。
        assert set(_candidate_pairs_sleeved(nodes, circles, cable, (), cum)) == set(
            _candidate_pairs(nodes, circles, cable)
        )


# ---------- API：未提交兼容 / 非法整次拒绝 / 回显 / 改线 ----------

def test_without_sleeves_response_is_itemwise_compatible():
    body = base_body(circles=[{"x": 50, "y": 15, "radius": 10}])
    data = post(body).json()
    assert data["sleeves"] == []
    # 原字段结论与旧版完全一致
    assert data["collision_count"] == 1
    assert data["collisions"][0]["nearest"] == {"x": 50.0, "y": 0.0}


def test_sleeve_makes_tangent_collision_via_api():
    body = base_body(
        circles=[{"x": 50, "y": -20, "radius": 10}],
        sleeves=[{"start_mileage": 50, "end_mileage": 70, "outer_radius": 10}],
    )
    r = post(body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["feasible"] is False
    assert data["collision_count"] == 1
    iv = data["intrusion_intervals"][0]
    assert iv["start_mileage"] == 50.0 and iv["length"] == 0.0
    # 片段携带原线段编号 0
    assert iv["pieces"][0]["segment_index"] == 0


def test_illegal_sleeve_ranges_rejected_entirely_with_no_partial_risks():
    # 空区间
    r = post(base_body(
        circles=[{"x": 50, "y": 0, "radius": 10}],
        sleeves=[{"start_mileage": 60, "end_mileage": 60, "outer_radius": 10}],
    ))
    assert r.status_code == 422
    errors = r.json()["errors"]
    assert any("sleeves[0].start_mileage" in k for k in errors)
    assert "feasible" not in r.json()

    # 超出路径末端（路径长 100）
    r = post(base_body(
        circles=[{"x": 50, "y": 0, "radius": 10}],
        sleeves=[{"start_mileage": 90, "end_mileage": 101, "outer_radius": 10}],
    ))
    assert r.status_code == 422
    assert any("sleeves[0].end_mileage" in k for k in r.json()["errors"])

    # 负起点
    r = post(base_body(
        sleeves=[{"start_mileage": -1, "end_mileage": 10, "outer_radius": 10}],
    ))
    assert r.status_code == 422
    assert any("sleeves[0].end_mileage" in k for k in r.json()["errors"])

    # 外半径小于电缆半径（电缆 5）
    r = post(base_body(
        sleeves=[{"start_mileage": 10, "end_mileage": 20, "outer_radius": 4.9}],
    ))
    assert r.status_code == 422
    assert any("sleeves[0].outer_radius" in k for k in r.json()["errors"])

    # 非有限值 / 非正半径 / 多余字段
    r = post(base_body(
        sleeves=[{"start_mileage": "NaN", "end_mileage": 10, "outer_radius": 10}],
    ))
    assert r.status_code == 422
    assert any("sleeves[0].start_mileage" in k for k in r.json()["errors"])
    r = post(base_body(
        sleeves=[{"start_mileage": 1, "end_mileage": 10, "outer_radius": 0}],
    ))
    assert r.status_code == 422
    r = post(base_body(
        sleeves=[{"start_mileage": 1, "end_mileage": 10, "outer_radius": 10, "extra": 1}],
    ))
    assert r.status_code == 422

    # 多项中一项非法 -> 整次拒绝，不输出任何部分风险
    r = post(base_body(
        circles=[{"x": 50, "y": -20, "radius": 10}],
        sleeves=[
            {"start_mileage": 50, "end_mileage": 70, "outer_radius": 10},  # 会命中
            {"start_mileage": 80, "end_mileage": 70, "outer_radius": 10},  # 非法
        ],
    ))
    assert r.status_code == 422
    body_json = r.json()
    assert "feasible" not in body_json and "collisions" not in body_json
    assert any("sleeves[1]" in k for k in body_json["errors"])


def test_sleeve_equal_to_cable_radius_is_allowed_and_equivalent():
    # 外半径 == 电缆半径：合法，但几何结论与无套管完全一致。
    body = base_body(circles=[{"x": 50, "y": 15, "radius": 10}])
    without = post(body).json()
    body["sleeves"] = [{"start_mileage": 10, "end_mileage": 90, "outer_radius": 5}]
    with_sleeve = post(body).json()
    assert with_sleeve["collision_count"] == without["collision_count"]
    assert with_sleeve["intrusion_intervals"] == without["intrusion_intervals"]
    assert with_sleeve["compound_intrusion_segments"] == without[
        "compound_intrusion_segments"
    ]


def test_sleeve_with_reroute_preview_candidate_has_no_sleeve():
    # 套管沿原路径里程；候选改线不携带套管（sleeves 回显为空），
    # 原线结论仍含套管命中，快照内两视图各自同源。
    body = base_body(
        circles=[{"x": 0, "y": -20, "radius": 10}],
        # 切点在里程 0（节点 (0,0)）：普通扩张 15 不命中（距 20），
        # 套管扩张 20 在起点里程相切 -> 唯一命中点是路径起点。
        sleeves=[{"start_mileage": 0, "end_mileage": 20, "outer_radius": 10}],
        reroute={
            "start_index": 0,
            "end_index": 1,
            "replacement_points": [
                {"x": 0, "y": 0},
                {"x": 50, "y": -60},
                {"x": 100, "y": 0},
            ],
        },
    )
    r = post(body)
    assert r.status_code == 200, r.text
    data = r.json()
    # 原线：里程 0 处套管相切（扩张 20 == 距离 20）
    assert data["feasible"] is False
    assert data["sleeves"][0]["start_mileage"] == 0.0
    # 候选线不应用原线里程套管
    candidate = data["reroute_preview"]["candidate"]
    assert candidate["sleeves"] == []
