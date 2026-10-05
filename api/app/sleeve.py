"""可选加粗接头套管的跨字段校验（未舍入双精度）。

请求可携带一段按**原路径累计里程**给出的闭区间套管：

- 区间必须非空（``start_mileage < end_mileage``）；
- 必须位于路径内（``0 ≤ start < end ≤ 路径总长``，边界点允许落在
  路径起终点）；
- ``outer_radius`` 为正数且不得小于普通电缆半径。

任何一条不满足都返回带字段定位的错误（由 main 转成 422），整次
预检拒绝、不输出任何部分风险。字段级类型/正数限制由 Pydantic 模型
负责（NaN/Infinity、字符串、布尔等在此之前已被拒绝）。
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .geometry import SleeveSpec, cumulative_mileage

Point = Tuple[float, float]


def validate_sleeve(
    nodes: Sequence[Point],
    cable_radius: float,
    start_mileage: float,
    end_mileage: float,
    outer_radius: float,
) -> List[Tuple[Tuple[str, ...], str]]:
    """校验套管范围与外半径；返回 ``(字段定位, 消息)`` 错误列表（空为通过）。

    字段定位形如 ``('sleeve', 'start_mileage')``，由 main 的异常处理器
    转成与现有约定一致的键 ``sleeve.start_mileage``。
    """
    errors: List[Tuple[Tuple[str, ...], str]] = []
    total = cumulative_mileage(nodes)[-1]

    def bad(field: str, msg: str) -> None:
        errors.append((("sleeve", field), msg))

    if not (0.0 <= start_mileage <= total and 0.0 <= end_mileage <= total):
        bad(
            "start_mileage",
            f"套管里程必须位于路径内 [0, {total:g}]（含两端）",
        )
    if start_mileage >= end_mileage:
        bad(
            "start_mileage",
            "套管区间必须非空：start_mileage 必须严格小于 end_mileage",
        )
    if outer_radius < cable_radius:
        bad(
            "outer_radius",
            f"套管外半径不得小于原电缆半径（{outer_radius:g} < {cable_radius:g}）",
        )
    return errors


def build_sleeve_spec(
    start_mileage: float,
    end_mileage: float,
    outer_radius: float,
) -> SleeveSpec:
    """构造几何层使用的不可变套管描述（调用前应已通过校验）。"""
    return SleeveSpec(
        start_mileage=start_mileage,
        end_mileage=end_mileage,
        outer_radius=outer_radius,
    )
