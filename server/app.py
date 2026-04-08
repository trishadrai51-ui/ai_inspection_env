from fastapi import FastAPI
from my_env import SystemInspectionEnv, Action

app = FastAPI()
env = SystemInspectionEnv()

@app.post("/reset")
def reset():
    obs = env.reset()
    return {
        "cpu_usage": obs.cpu_usage,
        "memory_usage": obs.memory_usage,
        "error_rate": obs.error_rate,
        "traffic": obs.traffic
    }

@app.post("/step")
def step(action: dict):
    action_obj = Action(**action)
    obs, reward, done, _ = env.step(action_obj)

    return {
        "observation": {
            "cpu_usage": obs.cpu_usage,
            "memory_usage": obs.memory_usage,
            "error_rate": obs.error_rate,
            "traffic": obs.traffic
        },
        "reward": reward,
        "done": done,
        "info": {}
    }
