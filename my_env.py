from pydantic import BaseModel
import random

class Observation(BaseModel):
    cpu_usage: float
    memory_usage: float
    error_rate: float
    traffic: float

class Action(BaseModel):
    decision: str  # ignore / alert / escalate

class Reward(BaseModel):
    value: float

class SystemInspectionEnv:
    def __init__(self):
        self.step_count = 0
        self.max_steps = 5
        self.current_state = None

    def reset(self):
        self.step_count = 0
        self.current_state = self._generate_state()
        return self.current_state

    def state(self):
        return self.current_state

    def step(self, action: Action):
        self.step_count += 1

        # decision logic
        if self.current_state.error_rate > 0.7 or self.current_state.cpu_usage > 0.8:
            correct = "escalate"
        elif self.current_state.error_rate > 0.4:
            correct = "alert"
        else:
            correct = "ignore"

        # reward
        if action.decision == correct:
            reward = 1.0
        elif action.decision == "alert":
            reward = 0.5
        else:
            reward = -1.0

        done = self.step_count >= self.max_steps

        self.current_state = self._generate_state()

        return self.current_state, reward, done, {}

    def _generate_state(self):
        return Observation(
            cpu_usage=round(random.uniform(0, 1), 2),
            memory_usage=round(random.uniform(0, 1), 2),
            error_rate=round(random.uniform(0, 1), 2),
            traffic=round(random.uniform(0, 1), 2),
        )

# -------- TASK GRADERS --------

def grader_easy(action, obs):
    if obs.error_rate > 0.8:
        return 1.0 if action == "escalate" else 0.0
    return 1.0 if action == "ignore" else 0.0


def grader_medium(action, obs):
    if obs.error_rate > 0.7 or obs.cpu_usage > 0.8:
        return 1.0 if action == "escalate" else 0.0
    elif obs.error_rate > 0.4:
        return 1.0 if action == "alert" else 0.0
    return 1.0 if action == "ignore" else 0.0


def grader_hard(action, obs):
    if obs.error_rate > 0.6 and obs.cpu_usage > 0.7:
        return 1.0 if action == "escalate" else 0.0
    elif obs.error_rate > 0.3:
        return 0.7 if action == "alert" else 0.0
    return 1.0 if action == "ignore" else 0.0