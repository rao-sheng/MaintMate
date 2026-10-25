
from app.agent.troubleshooting import all_steps_failed

steps = ["step_1", "step_2", "step_3"]

# 所有步骤都已尝试且未解决
history1 = {
    "step_1": "not_resolved",
    "step_2": "not_resolved",
    "step_3": "not_resolved"
}

# 还有一步没有尝试
history2 = {
    "step_1": "not_resolved",
    "step_2": "not_resolved"
}

assert all_steps_failed(steps, history1) is True
assert all_steps_failed(steps, history2) is False
assert all_steps_failed([], history1) is False

print("排障判断测试通过！")
