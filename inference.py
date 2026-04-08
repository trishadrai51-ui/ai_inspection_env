import os
from my_env import SystemInspectionEnv, Action

# No API needed (rule-based baseline)

env = SystemInspectionEnv()
obs = env.reset()

task_name = "system_inspection"
benchmark = "ai_inspection_env"
model_name = "rule_based"

print(f"[START] task={task_name} env={benchmark} model={model_name}")

rewards = []
done = False
step = 0

while not done and step < 5:
    step += 1

    # rule-based decision logic
    if obs.error_rate > 0.7 or obs.cpu_usage > 0.8:
        decision = "escalate"
    elif obs.error_rate > 0.4:
        decision = "alert"
    else:
        decision = "ignore"

    action = Action(decision=decision)

    obs, reward, done, _ = env.step(action)

    rewards.append(reward)

    print(f"[STEP] step={step} action={decision} reward={reward:.2f} done={str(done).lower()} error=null")

score = max(0, sum(rewards) / len(rewards)) if rewards else 0

print(f"[END] success={str(score>0.5).lower()} steps={step} score={score:.2f} rewards={','.join(f'{r:.2f}' for r in rewards)}")