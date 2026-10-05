"""可选加粗接头套管（sleeve）几何与接口测试。

用解析几何夹具覆盖：

- 套管范围**跨拐点**：未舍入里程拆段，两个原线段编号都命中并在拐点合并；
- **端点相切**：套管区间端点恰好是扩张圈与路径的切点——边界点按较大
  半径裁决，不能漏报，也不能因拆段产生重复碰撞；
- **重叠安全圈**：套管加粗后新进入扩张圈，与既有命中圈形成复合侵入；
- **标定后坐标**：survey→path 刚体变换后，按原路径里程的套管仍命中；
- 粗半径只在套管内生效（范围外漏检补齐、范围内不误报细半径场景）；
- 不拆段等价性：无套管 / 外半径等于电缆半径时结果与旧链路逐项一致；
- 一个（原线段, 圈）至多一处碰撞（拆段去重），片段映回原线段编号与
  未舍入累计里程；
- 仅三位小数显示相等的不同边界不合并；
- 非法范围（空区间、越界、外半径小于电缆半径、NaN）整次 422 拒绝，
  不输出部分风险；未提交 sleeve 时响应逐项兼容。
"""

import math

import pytest

from app.geometry import (
    SleeveSpec,
    analyze_path_full,
    cumulative_mileage,
    detect_collisions,
)


# ---------- 基本：套管范围内按较粗半径补漏 ----------


def test_sleeve_detects_intrusion_missed_by_uniform_radius():
    # 直线 0..100；孔 (50, 14) r10。细电缆半径 5：扩张 15 > 14 会碰撞，
    # 取间隙场景：cable=4 → 扩张 14，距离 14 恰好相切（细半径也命中）。
    # 用 cable=4、孔 y=15：细扩张 14 < 15 不命中；粗套管 r=5 → 15 相切。
    nodes = [(0.0, 0.0), (100.0, 0.0)]
    # 无套管：距离 15 > 14，完全安全
    coll0, ivs0, _ = analyze_path_full(nodes, [((50.0, 15.0), 10.0)], 4.0)
    assert coll0 == [] and ivs0 == []

    # 套管覆盖里程 [40,60]（含切点里程 50），外半径 5：相切命中
    sleeve = SleeveSpec(40.0, 60.0, 5.0)
    coll, ivs, comp = analyze_path_full(
        nodes, [((50.0, 15.0), 10.0)], 4.0, sleeve
    )
    assert len(coll) == 1
    c = coll[0]
    assert (c.segment_index, c.circle_index) == (0, 0)
    assert c.nearest == (50.0, 0.0)
    assert c.distance == 15.0
    assert c.expanded_radius == 15.0
    assert c.cable_radius == 5.0  # 判定点位于套管内
    # 零长相切区间，映回原线段 0 与里程 50
    assert len(ivs) == 1
    iv = ivs[0]
    assert iv.entry_segment_index == iv.exit_segment_index == 0
    assert iv.entry_point == iv.exit_point == (50.0, 0.0)
    assert iv.start_mileage == iv.end_mileage == 50.0
    assert comp == []


def test_sleeve_outside_range_keeps_base_radius():
    # 同样的孔在里程 50 相切于粗半径；套管只覆盖 [0,40]，切点不在其中。
    nodes = [(0.0, 0.0), (100.0, 0.0)]
    sleeve = SleeveSpec(0.0, 40.0, 5.0)
    coll, ivs, _ = analyze_path_full(
        nodes, [((50.0, 15.0), 10.0)], 4.0, sleeve
    )
    # 边界 40 距切点里程 50 有 10mm 间隙，细半径扩张 14 < 15：不命中。
    assert coll == [] and ivs == []


def test_sleeve_outer_radius_equal_to_base_is_bitwise_compatible():
    nodes = [(0.0, 0.0), (120.0, 30.0)]
    circles = [((60.0, 40.0), 8.0), ((200.0, 200.0), 5.0)]
    sleeve = SleeveSpec(30.0, 90.0, 5.0)
    a = analyze_path_full(nodes, circles, 5.0)
    b = analyze_path_full(nodes, circles, 5.0, sleeve)
    # 外半径 == 电缆半径时拆段不改变任何结论
    assert [(c.segment_index, c.circle_index) for c in a[0]] == [
        (c.segment_index, c.circle_index) for c in b[0]
    ]
    assert len(a[1]) == len(b[1])
    assert len(a[2]) == len(b[2])
    for iva, ivb in zip(a[1], b[1]):
        assert iva.start_mileage == ivb.start_mileage
        assert iva.end_mileage == ivb.end_mileage
        assert len(iva.pieces) == len(ivb.pieces)


# ---------- 跨拐点：未舍入里程拆段 ----------


def test_sleeve_across_corner_hits_both_original_segments():
    # 折线路径 (0,0)->(10,0)->(10,10)；圆心 (11,-1) 在拐点 (10,0) 的外角，
    # 到两段的最近点都是拐点本身，距离 sqrt2。细扩张（r+0.5≈0.914）
    # 不命中；套管外半径 1 → 扩张 sqrt2，恰在拐点双相切（端点零长点）。
    r = math.sqrt(2.0) - 1.0
    nodes = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    sleeve = SleeveSpec(5.0, 15.0, 1.0)  # 跨拐点里程 10
    coll, ivs, _ = analyze_path_full(nodes, [((11.0, -1.0), r)], 0.5, sleeve)
    # 两个原线段编号（0、1）各一处碰撞，并跨拐点合并为一个零长区间
    assert sorted((c.segment_index, c.circle_index) for c in coll) == [(0, 0), (1, 0)]
    assert len(ivs) == 1
    iv = ivs[0]
    assert iv.entry_segment_index == 0 and iv.exit_segment_index == 1
    assert iv.entry_point == iv.exit_point == (10.0, 0.0)
    assert iv.length == 0.0
    assert [(p.segment_index, p.t0, p.t1) for p in iv.pieces] == [
        (0, 1.0, 1.0),
        (1, 0.0, 0.0),
    ]


def test_sleeve_split_at_noninteger_mileage_maps_to_original_segment():
    # 单段 0..100；粗套管 [33.3, 66.7]。孔 (50, 12) r10：细 cable2 扩张
    # 12 恰好相切（细半径也命中），改成细扩张 12、距离 13 的孔 y=13：
    # 粗 cable3 → 扩张 13 在里程 50 相切；细扩张 12 不命中。
    nodes = [(0.0, 0.0), (100.0, 0.0)]
    sleeve = SleeveSpec(100.0 / 3.0, 200.0 / 3.0, 3.0)
    coll, ivs, _ = analyze_path_full(nodes, [((50.0, 13.0), 10.0)], 2.0, sleeve)
    assert len(coll) == 1
    assert coll[0].segment_index == 0
    # 区间是切点 (50,0) 的零长点，里程严格映回原路径
    assert ivs[0].start_mileage == 50.0
    # 拆分边界（33.333…/66.666…）不产生额外片段
    assert len(ivs[0].pieces) == 1


# ---------- 端点相切：边界点按较大半径裁决 ----------


def test_tangent_exactly_at_sleeve_start_boundary_is_hit_once():
    # 粗扩张恰好与路径相切于里程 s（=套管起点）；细扩张不命中。
    # 边界点必须按粗半径裁决（命中），且全线段只有一处碰撞。
    nodes = [(0.0, 0.0), (100.0, 0.0)]
    s = 40.0
    sleeve = SleeveSpec(s, 70.0, 5.0)
    coll, ivs, _ = analyze_path_full(
        nodes, [((s, 15.0), 10.0)], 4.0, sleeve
    )
    keys = [(c.segment_index, c.circle_index) for c in coll]
    assert keys == [(0, 0)]  # 拆两段也不重复
    # 相切零长点恰在边界里程
    assert ivs[0].start_mileage == ivs[0].end_mileage == s
    assert ivs[0].entry_point == (s, 0.0)


def test_tangent_exactly_at_sleeve_end_boundary_is_hit_once():
    nodes = [(0.0, 0.0), (100.0, 0.0)]
    e = 70.0
    sleeve = SleeveSpec(40.0, e, 5.0)
    coll, ivs, _ = analyze_path_full(
        nodes, [((e, 15.0), 10.0)], 4.0, sleeve
    )
    assert [(c.segment_index, c.circle_index) for c in coll] == [(0, 0)]
    assert ivs[0].start_mileage == ivs[0].end_mileage == e


def test_sleeve_boundary_at_path_endpoint():
    # 套管终点 == 路径终点里程；端点 (100,0) 在粗扩张圈上。
    nodes = [(0.0, 0.0), (100.0, 0.0)]
    sleeve = SleeveSpec(80.0, 100.0, 5.0)
    # 孔 (115,0) r10：细扩张 14，端点距 15 → 不命中；粗扩张 15 → 相切。
    coll, ivs, _ = analyze_path_full(
        nodes, [((115.0, 0.0), 10.0)], 4.0, sleeve
    )
    assert len(coll) == 1
    assert coll[0].nearest == (100.0, 0.0)
    assert coll[0].cable_radius == 5.0
    p = ivs[0].pieces[0]
    assert (p.t0, p.t1) == (1.0, 1.0)
    assert ivs[0].end_mileage == 100.0


def test_sleeve_ending_exactly_at_node_adjacent_segment_uses_thick_radius():
    # 套管 [0,10] 的终点恰在拐点（里程 10）；拐点后的段 1 整条在套管外，
    # 但其起点（=边界点）必须按较大半径裁决：圆在拐点外角双相切。
    r = math.sqrt(2.0) - 1.0
    nodes = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    center = (11.0, -1.0)
    sleeve = SleeveSpec(0.0, 10.0, 1.0)
    coll, ivs, _ = analyze_path_full(nodes, [(center, r)], 0.5, sleeve)
    keys = sorted((c.segment_index, c.circle_index) for c in coll)
    # 段 0（套管内）与段 1（套管外，但起点=套管终点）都命中且不重复
    assert keys == [(0, 0), (1, 0)]
    assert {c.cable_radius for c in coll} == {1.0}
    # 两个同坐标零长片段在拐点合并为一个零长区间
    assert len(ivs) == 1
    assert ivs[0].entry_point == ivs[0].exit_point == (10.0, 0.0)
    assert [(p.segment_index, p.t0, p.t1) for p in ivs[0].pieces] == [
        (0, 1.0, 1.0),
        (1, 0.0, 0.0),
    ]


def test_sleeve_starting_exactly_at_node_previous_segment_uses_thick_radius():
    # 套管 [10,20] 的起点恰在拐点（里程 10）；拐点前的段 0 整条在套管外，
    # 其终点（=边界点）同样必须按较大半径裁决。
    r = math.sqrt(2.0) - 1.0
    nodes = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    center = (11.0, -1.0)
    sleeve = SleeveSpec(10.0, 20.0, 1.0)
    coll, ivs, _ = analyze_path_full(nodes, [(center, r)], 0.5, sleeve)
    keys = sorted((c.segment_index, c.circle_index) for c in coll)
    assert keys == [(0, 0), (1, 0)]
    assert len(ivs) == 1
    assert [(p.segment_index, p.t0, p.t1) for p in ivs[0].pieces] == [
        (0, 1.0, 1.0),
        (1, 0.0, 0.0),
    ]


def test_sleeve_node_boundary_thin_radius_still_misses():
    # 同样的拐点构型，但外半径只等于电缆半径：边界探针也不得抬出碰撞。
    r = math.sqrt(2.0) - 1.0
    nodes = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    sleeve = SleeveSpec(10.0, 20.0, 0.5)
    coll, ivs, _ = analyze_path_full(
        nodes, [((11.0, -1.0), r)], 0.5, sleeve
    )
    assert coll == [] and ivs == []


def test_no_duplicate_collision_when_sleeve_covers_whole_segment():
    # 套管把整条线段包住：拆段逻辑仍只给一个（线段,圈）碰撞。
    nodes = [(0.0, 0.0), (10.0, 10.0)]
    sleeve = SleeveSpec(0.0, math.hypot(10.0, 10.0), 8.0)
    # 远心大圈：端点都在粗扩张盘内 -> 整段 [0,1] 命中且单片段
    coll, ivs, _ = analyze_path_full(
        nodes, [((-50.0, -50.0), 100.0)], 5.0, sleeve
    )
    assert len(coll) == 1
    p = ivs[0].pieces[0]
    assert (p.t0, p.t1) == (0.0, 1.0)
    assert coll[0].cable_radius == 8.0


# ---------- 粗半径产生正长侵入区间，片段映回原线段 ----------


def test_splitting_never_duplicates_hits_and_preserves_mileage_order():
    """随机夹具不变量（200 组）：

    - 每个（原线段, 禁入圈）至多一处碰撞——在未舍入里程处拆子段绝不
      制造重复碰撞；
    - 区间片段映回原线段后沿线段连续，里程单调，区间端点等于首尾片段。
    """
    import random

    random.seed(77)
    for _ in range(200):
        nodes = [(random.uniform(-5, 5), random.uniform(-5, 5))]
        for _ in range(random.randint(1, 4)):
            nodes.append(
                (
                    nodes[-1][0] + random.uniform(5, 40),
                    nodes[-1][1] + random.uniform(-20, 20),
                )
            )
        total = cumulative_mileage(nodes)[-1]
        cable = random.uniform(0.5, 4.0)
        outer = cable + random.uniform(0.5, 6.0)
        sleeve = SleeveSpec(
            random.uniform(0, total * 0.4),
            random.uniform(total * 0.6, total),
            outer,
        )
        circles = [
            (
                (random.uniform(-30, total + 30), random.uniform(-40, 40)),
                random.uniform(0.5, 8.0),
            )
            for _ in range(random.randint(1, 4))
        ]
        coll, ivs, _ = analyze_path_full(nodes, circles, cable, sleeve)
        hit_keys = [(c.segment_index, c.circle_index) for c in coll]
        assert len(hit_keys) == len(set(hit_keys))
        for iv in ivs:
            segs = [p.segment_index for p in iv.pieces]
            assert segs == list(range(segs[0], segs[-1] + 1))
            for p in iv.pieces:
                assert p.start_mileage <= p.end_mileage
            assert iv.start_mileage == iv.pieces[0].start_mileage
            assert iv.end_mileage == iv.pieces[-1].end_mileage


def test_positive_length_intrusion_within_sleeve_maps_back():
    # 粗扩张 18（cable8）与路径相交于里程 50±sqrt(18²-13²)。
    nodes = [(0.0, 0.0), (100.0, 0.0)]
    sleeve = SleeveSpec(20.0, 80.0, 8.0)
    coll, ivs, _ = analyze_path_full(nodes, [((50.0, 13.0), 10.0)], 5.0, sleeve)
    # 细扩张 15 > 13 其实也命中；改距离使细不命中、粗正长相交：
    # 距离 16：细扩张 15 < 16 不命中；粗扩张 18 > 16 正长。
    coll, ivs, _ = analyze_path_full(nodes, [((50.0, 16.0), 10.0)], 5.0, sleeve)
    assert len(coll) == 1
    assert coll[0].cable_radius == 8.0
    iv = ivs[0]
    half = math.sqrt(18.0 * 18.0 - 16.0 * 16.0)
    assert iv.start_mileage == pytest.approx(50.0 - half)
    assert iv.end_mileage == pytest.approx(50.0 + half)
    assert iv.length == pytest.approx(2 * half)
    # 整段都在套管 [20,80] 内（half≈8.25 → [41.75,58.25]）
    assert iv.start_mileage > 20.0 and iv.end_mileage < 80.0
    assert iv.pieces[0].segment_index == 0


# ---------- 重叠安全圈：复合侵入 ----------


def test_sleeve_creates_compound_overlap_with_existing_circle():
    # 圈0 细半径已在里程 [5,35] 命中（且位于套管之外，区间不被加粗）；
    # 圈1 仅在粗半径下命中，套管从里程 35 开始：圈1 粗区间 [40-sqrt68,
    # 40+sqrt68] 的起点≈31.75 < 35，故其在套管左边界 35 处被裁决命中，
    # 与圈0 的右端 35 闭集相接，形成起点 35 的双圈复合段。
    nodes = [(0.0, 0.0), (100.0, 0.0)]
    circles = [
        ((20.0, 0.0), 10.0),   # 扩张（细5）15 → [5,35]，在套管外
        ((40.0, 16.0), 10.0),  # 细扩张 15 < 16 不命中；粗（8）18 命中
    ]
    sleeve = SleeveSpec(35.0, 100.0, 8.0)
    coll, ivs, comp = analyze_path_full(nodes, circles, 5.0, sleeve)
    # 圈0 一个、圈1 一个碰撞
    assert sorted(c.circle_index for c in coll) == [0, 1]
    assert len(comp) == 1
    seg = comp[0]
    assert seg.circle_indices == (0, 1)
    hi1 = 40.0 + math.sqrt(68.0)
    # 套管内两圈都按粗半径：圈0 区间自 35 延伸到 38（粗扩张 18），圈1
    # 自边界 35 起活动；重叠复合段 [35,38]（圈1 完全由套管加粗新增）。
    assert seg.start_mileage == pytest.approx(35.0)
    assert seg.end_mileage == pytest.approx(38.0)
    assert seg.length == pytest.approx(3.0)
    assert seg.pieces[0].segment_index == 0
    assert hi1 > 38.0


# ---------- 不合并仅三位小数相等的边界 ----------


def test_sleeve_does_not_merge_distinct_three_decimal_boundaries():
    # 折返路径上同坐标异里程两处；套管覆盖两段，各自的圈区间边界在三位
    # 小数下显示相同也必须保持独立。
    nodes = [(0.0, 0.0), (1000.0, 0.0), (0.0, 0.0)]
    # 圆心略偏，使两段的进入/离开点在双精度上相差一次反向求值舍入
    # （显示同为 485.000），物理上是两个独立区间。
    sleeve = SleeveSpec(0.0, 2000.0, 6.0)
    _, ivs, _ = analyze_path_full(
        nodes, [((500.0, 0.0005), 10.0)], 5.0, sleeve
    )
    assert len(ivs) == 2
    assert [iv.entry_segment_index for iv in ivs] == [0, 1]


# ---------- 粗筛不漏套管候选 ----------


def test_coarse_filter_with_sleeve_does_not_miss_candidate():
    import random

    from app.geometry import _candidate_pairs, _subsegments

    random.seed(2026)
    nodes = [(0.0, 0.0)]
    x = 0.0
    for _ in range(30):
        x += random.uniform(20.0, 60.0)
        nodes.append((x, random.uniform(-20.0, 20.0)))
    total = cumulative_mileage(nodes)[-1]
    sleeve = SleeveSpec(total * 0.3, total * 0.7, 9.0)
    cable = 2.0
    circles = [
        ((random.uniform(0, x), random.uniform(-40, 40)), random.uniform(1, 6))
        for _ in range(40)
    ]
    total_eps = 1e-10 * max(1.0, total)
    n = len(nodes) - 1
    cum = cumulative_mileage(nodes)
    seg_radii = []
    for i in range(n):
        covered = sleeve.start_mileage < cum[i + 1] and sleeve.end_mileage > cum[i]
        seg_radii.append(sleeve.outer_radius if covered else cable)
    pairs = set(_candidate_pairs(nodes, circles, cable, seg_radii))
    # 朴素参照：任一子段的精确扩张半径与子段包围盒相交即应是候选
    for i in range(n):
        subs = _subsegments(i, nodes, cum, cable, sleeve, total_eps)
        for j, (center, r) in enumerate(circles):
            expect = False
            for a, b, r_eff, *_ in subs:
                R = r + r_eff
                x0, x1 = sorted((a[0], b[0]))
                y0, y1 = sorted((a[1], b[1]))
                if x0 <= center[0] + R and x1 >= center[0] - R:
                    if y0 <= center[1] + R and y1 >= center[1] - R:
                        expect = True
                        break
            if expect:
                assert (i, j) in pairs


# ---------- 与 detect_collisions 门面兼容 ----------


def test_detect_collisions_without_sleeve_unchanged():
    nodes = [(0.0, 0.0), (10.0, 10.0)]
    circles = [((5.0, 5.0), 1.0)]
    only = detect_collisions(nodes, circles, 2.0)
    full, _, _ = analyze_path_full(nodes, circles, 2.0)
    assert [(c.segment_index, c.circle_index) for c in only] == [
        (c.segment_index, c.circle_index) for c in full
    ]
    assert only[0].cable_radius == 2.0


# ---------- HTTP 接口 ----------


from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)
PATH = "/api/precheck"


def _body(**over):
    body = {
        "nodes": [{"x": 0, "y": 0}, {"x": 100, "y": 0}],
        "cable_radius": 4,
        "circles": [{"x": 50, "y": 15, "radius": 10}],
    }
    body.update(over)
    return body


def test_api_sleeve_tangent_hit_and_echo():
    body = _body(sleeve={"start_mileage": 40, "end_mileage": 60, "outer_radius": 5})
    r = client.post(PATH, json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["feasible"] is False
    assert data["collision_count"] == 1
    c = data["collisions"][0]
    assert c["nearest"] == {"x": 50.0, "y": 0.0}
    assert c["cable_radius"] == 5.0
    assert c["expanded_radius"] == 15.0
    # 回显供区间详情与 SVG 同源
    assert data["sleeve"] == {
        "start_mileage": 40.0,
        "end_mileage": 60.0,
        "outer_radius": 5.0,
    }
    assert data["intrusion_intervals"][0]["start_mileage"] == 50.0


def test_api_without_sleeve_is_compatible():
    r = client.post(PATH, json=_body())
    data = r.json()
    assert r.status_code == 200
    assert data["feasible"] is True  # 细半径扩张 14 < 15
    assert data["sleeve"] is None
    assert data["collisions"] == []


def test_api_sleeve_missing_hit_without_sleeve_shows_feasible():
    # 套管未提交：漏检场景保持「可敷设」
    r = client.post(PATH, json=_body())
    assert r.json()["feasible"] is True


@pytest.mark.parametrize(
    "sleeve,field",
    [
        ({"start_mileage": 60, "end_mileage": 60, "outer_radius": 5}, "sleeve.start_mileage"),
        ({"start_mileage": 80, "end_mileage": 120, "outer_radius": 5}, "sleeve.start_mileage"),
        ({"start_mileage": -1, "end_mileage": 50, "outer_radius": 5}, "sleeve.start_mileage"),
        ({"start_mileage": 40, "end_mileage": 30, "outer_radius": 5}, "sleeve.start_mileage"),
        ({"start_mileage": 40, "end_mileage": 60, "outer_radius": 3}, "sleeve.outer_radius"),
        ({"start_mileage": 40, "end_mileage": 60, "outer_radius": 0}, "sleeve.outer_radius"),
        ({"start_mileage": "NaN", "end_mileage": 60, "outer_radius": 5}, "sleeve.start_mileage"),
    ],
)
def test_api_invalid_sleeve_rejected_entirely(sleeve, field):
    r = client.post(PATH, json=_body(sleeve=sleeve))
    assert r.status_code == 422
    errors = r.json()["errors"]
    assert any(field in k for k in errors), errors
    # 整次拒绝：不输出任何部分风险（422 体只有错误结构）
    assert set(r.json().keys()) == {"ok", "errors"}


def test_api_sleeve_extra_field_rejected():
    body = _body(
        sleeve={
            "start_mileage": 40,
            "end_mileage": 60,
            "outer_radius": 5,
            "unexpected": 1,
        }
    )
    r = client.post(PATH, json=body)
    assert r.status_code == 422


def test_api_sleeve_with_calibration_hits_in_path_coordinates():
    # survey = path + (1000,2000)：孔心全站仪 (1050,2015) 变换到 (50,15)，
    # 细扩张 14 < 15 不命中；套管外半径 5 覆盖里程 40..60 -> 相切命中。
    body = {
        "nodes": [{"x": 0, "y": 0}, {"x": 100, "y": 0}],
        "cable_radius": 4,
        "circles": [{"x": 1050, "y": 2015, "radius": 10}],
        "calibration": {
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
        "sleeve": {"start_mileage": 40, "end_mileage": 60, "outer_radius": 5},
    }
    r = client.post(PATH, json=body)
    assert r.status_code == 200, r.json()
    data = r.json()
    assert data["feasible"] is False
    assert data["collisions"][0]["nearest"] == {"x": 50.0, "y": 0.0}
    assert data["sleeve"]["outer_radius"] == 5.0


def test_api_sleeve_with_reroute_candidate_uses_uniform_radius():
    # 套管只作用于原线；候选改线统一半径，候选视图 sleeve 为 null，
    # 其碰撞的 cable_radius 一律为普通半径，不混用粗细半径。
    body = _body(
        sleeve={"start_mileage": 40, "end_mileage": 60, "outer_radius": 5},
        reroute={
            "start_index": 0,
            "end_index": 1,
            "replacement_points": [
                {"x": 0, "y": 0},
                {"x": 50, "y": -30},
                {"x": 100, "y": 0},
            ],
        },
    )
    r = client.post(PATH, json=body)
    assert r.status_code == 200, r.json()
    data = r.json()
    # 原线：套管导致相切碰撞
    assert data["feasible"] is False
    assert data["collisions"][0]["cable_radius"] == 5.0
    candidate = data["reroute_preview"]["candidate"]
    assert candidate["sleeve"] is None
    for c in candidate["collisions"]:
        assert c["cable_radius"] == 4.0
