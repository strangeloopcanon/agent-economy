from __future__ import annotations

import pytest

from agent_economy.planner import (
    DecompositionPlan,
    PlannedTask,
    toposort_plan,
    validate_plan_dag,
)


def test_validate_plan_dag_accepts_simple_chain() -> None:
    plan = DecompositionPlan(
        tasks=[
            PlannedTask(id="T1", title="one", deps=[]),
            PlannedTask(id="T2", title="two", deps=["T1"]),
        ]
    )
    validate_plan_dag(plan=plan)


def test_validate_plan_dag_accepts_out_of_order_chain() -> None:
    plan = DecompositionPlan(
        tasks=[
            PlannedTask(id="T2", title="two", deps=["T1"]),
            PlannedTask(id="T1", title="one", deps=[]),
        ]
    )
    validate_plan_dag(plan=plan)


def test_toposort_plan_orders_tasks() -> None:
    plan = DecompositionPlan(
        tasks=[
            PlannedTask(id="T2", title="two", deps=["T1"]),
            PlannedTask(id="T1", title="one", deps=[]),
        ]
    )
    ordered = toposort_plan(plan=plan)
    assert [t.id for t in ordered.tasks] == ["T1", "T2"]


def test_validate_plan_dag_rejects_duplicate_ids() -> None:
    plan = DecompositionPlan(
        tasks=[
            PlannedTask(id="T1", title="one", deps=[]),
            PlannedTask(id="T1", title="dup", deps=[]),
        ]
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_plan_dag(plan=plan)


def test_validate_plan_dag_rejects_unknown_deps() -> None:
    plan = DecompositionPlan(
        tasks=[
            PlannedTask(id="T1", title="one", deps=[]),
            PlannedTask(id="T2", title="two", deps=["T999"]),
        ]
    )
    with pytest.raises(ValueError, match="unknown dep"):
        validate_plan_dag(plan=plan)


def test_validate_plan_dag_rejects_cycles() -> None:
    plan = DecompositionPlan(
        tasks=[
            PlannedTask(id="T1", title="one", deps=["T2"]),
            PlannedTask(id="T2", title="two", deps=["T1"]),
        ]
    )
    with pytest.raises(ValueError, match="cycle"):
        validate_plan_dag(plan=plan)
