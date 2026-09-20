from core.contracts import Plan, PlanStatus, PlanStep, PlanStepStatus


def test_plan_step_is_json_friendly():
    step = PlanStep(
        id="inspect-repo",
        description="Inspect the repository structure",
        required_capabilities=["filesystem"],
        completion_conditions=["repository structure collected"],
    )

    payload = step.to_dict()

    assert payload["id"] == "inspect-repo"
    assert payload["status"] == PlanStepStatus.PENDING.value
    assert payload["required_capabilities"] == ["filesystem"]


def test_plan_validates_dependencies_and_exposes_ready_steps():
    plan = Plan(
        goal="Diagnose the project issue",
        steps=[
            PlanStep(
                id="inspect",
                description="Inspect the repository",
            ),
            PlanStep(
                id="test",
                description="Run the test suite",
                depends_on=["inspect"],
            ),
            PlanStep(
                id="report",
                description="Report the findings",
                depends_on=["test"],
            ),
        ],
    )

    ready = plan.ready_steps()

    assert [step.id for step in ready] == ["inspect"]
    assert plan.status is PlanStatus.DRAFT


def test_plan_rejects_unknown_dependencies():
    try:
        Plan(
            goal="Invalid plan",
            steps=[
                PlanStep(
                    id="test",
                    description="Run tests",
                    depends_on=["missing"],
                )
            ],
        )
    except ValueError as exc:
        assert "unknown step" in str(exc)
    else:
        raise AssertionError("Expected unknown dependency validation error")


def test_plan_rejects_dependency_cycles():
    try:
        Plan(
            goal="Cyclic plan",
            steps=[
                PlanStep(
                    id="a",
                    description="Step A",
                    depends_on=["b"],
                ),
                PlanStep(
                    id="b",
                    description="Step B",
                    depends_on=["a"],
                ),
            ],
        )
    except ValueError as exc:
        assert "dependency cycle" in str(exc)
    else:
        raise AssertionError("Expected dependency cycle validation error")


def test_completed_dependency_unlocks_next_step():
    plan = Plan(
        goal="Complete dependent work",
        steps=[
            PlanStep(id="first", description="First step"),
            PlanStep(
                id="second",
                description="Second step",
                depends_on=["first"],
            ),
        ],
    )

    plan.steps[0].status = PlanStepStatus.COMPLETED

    assert [step.id for step in plan.ready_steps()] == ["second"]
