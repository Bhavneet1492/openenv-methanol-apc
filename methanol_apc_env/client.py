"""Methanol APC Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import MethanolAPCAction, MethanolAPCObservation


class MethanolAPCEnv(EnvClient[MethanolAPCAction, MethanolAPCObservation, State]):
    """Async WebSocket client for the Methanol APC Environment.

    Example::

        async with MethanolAPCEnv(base_url="http://localhost:8000") as env:
            result = await env.reset(task_name="startup")
            action = MethanolAPCAction(
                feed_rate_h2=3.0, feed_rate_co=1.5,
                cooling_water_flow=50.0, compressor_power=40.0,
            )
            result = await env.step(action)
            print(result.observation.temperature)
    """

    def _step_payload(self, action: MethanolAPCAction) -> Dict:
        return {
            "feed_rate_h2": action.feed_rate_h2,
            "feed_rate_co": action.feed_rate_co,
            "cooling_water_flow": action.cooling_water_flow,
            "compressor_power": action.compressor_power,
        }

    def _parse_result(self, payload: Dict) -> StepResult[MethanolAPCObservation]:
        obs_data = payload.get("observation", {})
        observation = MethanolAPCObservation(
            temperature=obs_data.get("temperature", 0.0),
            pressure=obs_data.get("pressure", 0.0),
            feed_rate_h2=obs_data.get("feed_rate_h2", 0.0),
            feed_rate_co=obs_data.get("feed_rate_co", 0.0),
            h2_co_ratio=obs_data.get("h2_co_ratio", 2.0),
            cooling_water_flow=obs_data.get("cooling_water_flow", 0.0),
            cooling_water_temp=obs_data.get("cooling_water_temp", 25.0),
            catalyst_health=obs_data.get("catalyst_health", 1.0),
            methanol_produced=obs_data.get("methanol_produced", 0.0),
            reaction_rate=obs_data.get("reaction_rate", 0.0),
            profit_this_step=obs_data.get("profit_this_step", 0.0),
            cumulative_profit=obs_data.get("cumulative_profit", 0.0),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 100),
            task_name=obs_data.get("task_name", "startup"),
            safety_warning=obs_data.get("safety_warning"),
            temperature_trend=obs_data.get("temperature_trend", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
