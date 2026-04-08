# AI System Inspection Environment

## Description
This environment simulates real-world system monitoring tasks performed by SRE/DevOps teams.  
The agent analyzes system metrics and decides whether to ignore, alert, or escalate issues.

## Motivation
Companies like Google manage large-scale systems and must continuously monitor CPU usage, memory, and error rates to prevent failures.

## Observation Space
- cpu_usage (0–1)
- memory_usage (0–1)
- error_rate (0–1)
- traffic (0–1)

## Action Space
- ignore → no issue
- alert → moderate issue
- escalate → critical issue

## Tasks
- Easy → based on high error rate
- Medium → error + CPU
- Hard → combined system metrics

## Reward
- Correct decision: +1
- Partial (alert): +0.5
- Wrong: -1

## Setup
```bash
pip install -r requirements.txt
python inference.py