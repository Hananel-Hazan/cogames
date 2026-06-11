# Chunk 1 Recon Report

**Mission:** arena (`cogsguard.missions.arena.make_basic_mission` with `.with_cogs(4)`)
**Status:** complete — no blockers triggered.
**Consolidated:** 2026-05-24, supersedes the chat summary; [chunk_1_followup.md](chunk_1_followup.md) retained as historical record.

---

## 6.1 Install Status

Captured by inspecting the installed environment and the editable `cogames` checkout. The Chunk 1 work order asked for these values but did not write them to a dedicated `install_status.txt`; the table below was reconstructed at consolidation time from `pip show`, `git rev-parse`, and the project plan.

| Field | Value | Source |
|---|---|---|
| `cogames` version | `0.0.0` (editable install) | `pip show cogames` |
| `cogames` repo path | `/mnt/exDisk0/git_repo/cogames` | [PROJECT_PLAN.md §4](../../research/reference/ENVIRONMENT.md) |
| `cogsguard` version | `0.0.0.post1.dev12` | `pip show cogsguard` |
| `cogsguard` repo HEAD | `bc8ac3d` (in `/mnt/exDisk0/git_repo/cogames`) | `git rev-parse --short HEAD` |
| `mettagrid` version | `0.26.21` | `pip show mettagrid` |
| `torch` version | `2.9.0+cu128` (CUDA 12.8) | `python -c "import torch; print(torch.__version__)"` |
| Python | 3.12 | conda env path `…/envs/cogames/lib/python3.12/…` |
| Conda env name | `cogames` | [PROJECT_PLAN.md §4](../../research/reference/ENVIRONMENT.md) |
| Hardware | RTX 2070 8 GB VRAM, 32 cores, 120 GB RAM | [PROJECT_PLAN.md §4](../../research/reference/ENVIRONMENT.md) |

## 6.2 Mission Inventory

Mission factory files in
`/home/hananel/miniconda3/envs/cogames/lib/python3.12/site-packages/cogsguard/missions/`.

| File | Factory function | Default `num_cogs` | Default `max_steps` | One-line summary |
|---|---|---|---|---|
| `arena.py` | `make_basic_mission(num_cogs=8, max_steps=1000)` | 8 | 1000 | Arena map + full `machina_1` reward stack. **Used for Chunk 1 recon.** |
| `empty.py` | `make_empty_mission(num_cogs=8, max_steps=1000)` | 8 | 1000 | Empty map, full reward stack — sanity / debug baseline. |
| `four_score.py` | `Mission` class with `num_cogs: int = 32`, `max_steps: int = 10000` | 32 | 10000 | Four-score variant (large team, long horizon). |
| `machina_1.py` | `make_machina1_mission(num_agents=8, max_steps=10000)` (class default `num_cogs=8`, `max_steps=10000`); also `make_game(num_cogs=2)` | 8 | 10000 | Canonical machina_1 mission; per-cog reward writes live here ([§6.3](#63-env-construction) cites lines 152–164). |
| `terrain.py` | *no `def make_*` matched by grep* | UNKNOWN — reason: factory not in the standard grep pattern; out-of-scope for consolidation | UNKNOWN | Present in directory; structure not inspected during Chunk 1. |
| `tutorial.py` | `make_tutorial_mission()`; class default `num_cogs=4`, `max_steps=1000` | 4 | 1000 | Tutorial mission; uses `TutorialMission` subclass. |

**Shaped variants:** **NO.** `grep -rln "shaped" /…/cogsguard/missions/` returned **zero matches** at consolidation time. The cogsguard mission stack does not currently expose shaped-reward variants under that name.

## 6.3 Env Construction

Verbatim from [chunk_1_followup.md Q1](chunk_1_followup.md) — the snippet below was executed end-to-end against the installed env and produced the printed output shown.

### Canonical preamble

```python
import numpy as np
import cogsguard.missions.machina_1           # registers machina_1 variants
from cogsguard.missions.arena import make_basic_mission
from mettagrid import PufferMettaGridEnv
from mettagrid.simulator import Simulator

mission = make_basic_mission().with_cogs(4)   # arena map, full machina_1 stack
cfg     = mission.make_env()                  # -> MettaGridConfig (pure config, not runnable)
sim     = Simulator()                         # event-handler hub; required by PufferMettaGridEnv
env     = PufferMettaGridEnv(sim, cfg, seed=42)

obs, info = env.reset(seed=42)                # obs: (num_agents, 500, 3) uint8
sim_obj   = env.current_simulation
action_ids = dict(sim_obj.action_ids)         # {'noop':0, 'move_north':1, ...}

# Per-cog observations are AgentObservation objects (token streams), not the raw uint8 buffer.
per_agent_obs = sim_obj.observations()        # list[AgentObservation], length == num_agents

actions = np.zeros(env.num_agents, dtype=np.int32)  # combined-index encoding, range [0,40)
obs, rewards, terminals, truncations, info = env.step(actions)
```

Observed output:

```
num_agents: 4   obs: (4, 500, 3) uint8
step tuple len: 5
obs.shape: (4, 500, 3)   rewards.shape: (4,) float32   rewards: [0. 0. 0. 0.]
terminals: [F F F F]     truncations: [F F F F]        info: {}
```

> **Correction to initial assumption.** `env.step()` returns a **5-tuple** `(obs, rewards, terminals, truncations, info)` (Gymnasium style), not the 4-tuple `(obs, reward, done, info)` originally framed. `rewards`, `terminals`, `truncations` are per-cog arrays of length `num_agents` (`float32`, `bool`, `bool`). Source: `mettagrid/envs/mettagrid_puffer_env.py:296` (signature), lines 285–293 and 396+ (behavior).

### Bridge imports

```python
# Mission factory + arena map
from cogsguard.missions.arena import make_basic_mission
# (Mission class is cogsguard.missions.mission.CvCMission, exposed via make_basic_mission)

# MettaGridConfig (pydantic config object returned by mission.make_env())
from mettagrid import MettaGridConfig
# canonically:  from mettagrid.config.mettagrid_config import MettaGridConfig

# Runtime env wrapper (PufferEnv subclass)
from mettagrid import PufferMettaGridEnv
# canonically:  from mettagrid.envs.mettagrid_puffer_env import MettaGridPufferEnv as PufferMettaGridEnv

# Simulator + event-handler hub
from mettagrid.simulator import Simulator
```

References:
- `mettagrid/__init__.py:49,54,68` (lazy re-exports of `Simulator` and the `MettaGridPufferEnv → PufferMettaGridEnv` alias).
- `cogsguard/missions/arena.py:53` (`make_basic_mission`).
- `cogsguard/missions/mission.py:13` (`class CvCMission`).
- `cogames/core.py:163` (`CoGameMission.make_env() -> MettaGridConfig`).

There is **no `from cogsguard import make_env`** top-level bridge — go through the mission factory.

### `PufferMettaGridEnv(...)` constructor

```python
# mettagrid/envs/mettagrid_puffer_env.py:66 / 76-84
class MettaGridPufferEnv(PufferEnv):
    def __init__(
        self,
        simulator: Simulator,
        cfg: MettaGridConfig,
        supervisor_policy_spec: Optional[PolicySpec] = None,
        step_info_keys: Optional[Sequence[str]] = None,
        buf: Any = None,
        seed: int = 0,
    ):
```

| Param | Type | Meaning |
|---|---|---|
| `simulator` | `Simulator` | Event-handler hub; can attach `StatsTracker`, `EarlyResetHandler`, etc. (see `cogames/train.py:481-483` for a typical attachment pattern). |
| `cfg` | `MettaGridConfig` | Pydantic config returned by `mission.make_env()`. |
| `supervisor_policy_spec` | `PolicySpec \| None` | Optional supervisor that authors vibe actions on the cog's behalf — leave `None` for our purposes. |
| `step_info_keys` | `Sequence[str] \| None` | Which extra stats to surface in the `info` dict each step (`"game/...", "team/...", "attributes/...", "agent/..."`). |
| `buf` | `Any` | Optional pre-allocated PufferLib buffers (for vectorized rollouts). `None` is fine for single-env work. |
| `seed` | `int` | Episode seed (default `0`). Passed straight to `Simulator.new_simulation(...)` at line 205. |

**RNG knobs.**
- Constructor takes only `seed=`. **No `map_seed=` on `PufferMettaGridEnv` itself.**
- `env.reset(seed=...)` (line 285) overrides `self._current_seed` for the next sim.
- `map_seed` exists one layer up in the training wrapper `_EnvBuilder` (`cogames/train.py:451, 478-480`), where it's combined with the per-env seed offset to derive `map_builder.seed`. For deterministic maps in the DT pipeline, mutate `cfg.game.map_builder.seed` (a `MapGen.Config.seed: int | None`) before constructing the env — that's the same hook the trainer uses.

### Per-cog reward source

```python
# cogsguard/missions/machina_1.py:152-164
for agent in env.game.agents:
    team_name = team_v.team_name(agent.team_id)
    if team_name is None:
        continue
    held_junction_values = _held_junction_values(team_name=team_name, clips_ship_count=clips_ship_count)
    # net:* includes the team's root node, so subtract it and reward only held junctions.
    agent.rewards["aligned_junction_held"] = reward(
        held_junction_values,
        weight=1.0 / mission.max_steps,
        per_tick=True,
    )
```

The write is at `machina_1.py:160` (inside `Machina1Variant.modify_env`). Each agent gets its own `agent.rewards["aligned_junction_held"]` entry — the loop iterates per agent and per team, so the stream is genuinely per-cog, not team-broadcast.

Verified at runtime: `env.step(...)` returned `rewards.shape == (num_agents,)`, dtype `float32`.

**Caveat.** Only one `agent.rewards[...]` write exists in cogsguard (`grep -rn 'agent\.rewards\b' cogsguard/`). Additional per-cog reward terms (heart pickups, gear, damage, etc.) are wired through events / global handlers / mutations and aggregated by the simulator into the per-cog `rewards` buffer. The buffer is **shape `(num_agents,)`** but is the *sum* of all configured per-cog reward terms for that step, not a decomposed multi-term tensor.

## 6.4 Policy Interface

Verbatim from [chunk_1_followup.md Q3](chunk_1_followup.md).

### Base classes

From `mettagrid/policy/policy.py`:

```python
# policy.py:28-86  (per-agent base)
class AgentPolicy:
    def __init__(self, policy_env_info: PolicyEnvInterface):
        self._policy_env_info = policy_env_info
        self._infos: dict[str, Any] = {}

    def step(self, obs: AgentObservation) -> Action:
        raise NotImplementedError(...)

    def reset(self, simulation: Optional[Simulation] = None) -> None:
        pass

    # Optional batched fast-path:
    def can_step_group(self, policies: Sequence["AgentPolicy"]) -> bool: ...
    def step_group(self, observations: list[tuple[int, AgentObservation]]) -> list[Action]: ...
```

```python
# policy.py:88-161  (multi-agent registry)
class MultiAgentPolicy(metaclass=PolicyRegistryMeta):
    short_names: list[str] | None = None
    minimum_action_timeout_ms: ClassVar[int] = 0

    def __init__(self, policy_env_info: PolicyEnvInterface, device: str = "cpu", **kwargs: Any):
        ...

    @abstractmethod
    def agent_policy(self, agent_id: int) -> AgentPolicy: ...

    def step_batch(self, raw_observations: np.ndarray, raw_actions: np.ndarray) -> None:
        raise NotImplementedError(...)

    # plus: load_policy_data / save_policy_data / network / reset / configure_action_timeout_ms
```

CLI shorthand `random` resolves to `RandomMultiAgentPolicy` via `short_names = ["random"]` (`mettagrid/policy/random_agent.py:46`). `starter` resolves to `StarterPolicy` via `short_names = ["starter"]` (`cogames/policy/starter_agent.py:469`).

### `RandomMultiAgentPolicy.__init__`

From `mettagrid/policy/random_agent.py:43-60`:

```python
class RandomMultiAgentPolicy(MultiAgentPolicy):
    short_names = ["random"]

    def __init__(self, policy_env_info: PolicyEnvInterface, device: str = "cpu", **kwargs):
        super().__init__(policy_env_info, device=device)
        vibe_action_p = float(kwargs.get("vibe_action_p", 0.5))
        if not 0.0 <= vibe_action_p <= 1.0:
            raise ValueError(f"vibe_action_p must be in [0.0, 1.0], got {vibe_action_p}")
        self._vibe_action_p = vibe_action_p

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return RandomAgentPolicy(self._policy_env_info, self._vibe_action_p)
```

And the per-agent class (`random_agent.py:10-40`):

```python
class RandomAgentPolicy(AgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface, vibe_action_p: float = 0.5):
        super().__init__(policy_env_info)
        ...
    def step(self, obs: AgentObservation) -> Action:
        ...
        return Action(name=random.choice(chosen_category))
```

Samples a category (`primary` vs `vibe`) weighted by `vibe_action_p`, then a uniform name from that category — so it returns vibe-only actions 50% of the time by default (`vibe_action_p=0.5`).

### `StarterPolicy.__init__`

From `cogames/policy/starter_agent.py:450-470`:

```python
class BaseStarterPolicy(MultiAgentPolicy):
    short_names: list[str]
    _role: str | None = None

    def __init__(self, policy_env_info: PolicyEnvInterface, device: str = "cpu"):
        super().__init__(policy_env_info, device=device)
        self._agent_policies: dict[int, StatefulAgentPolicy[StarterCogState]] = {}

    def agent_policy(self, agent_id: int) -> StatefulAgentPolicy[StarterCogState]:
        if agent_id not in self._agent_policies:
            self._agent_policies[agent_id] = StatefulAgentPolicy(
                StarterCogPolicyImpl(self._policy_env_info, agent_id, role=self._role),
                self._policy_env_info,
                agent_id=agent_id,
            )
        return self._agent_policies[agent_id]


class StarterPolicy(BaseStarterPolicy):
    short_names = ["starter"]
    pass
```

Implementation class (`starter_agent.py:56-67`):

```python
class StarterCogPolicyImpl(StatefulPolicyImpl[StarterCogState]):
    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int, role: str | None = None):
        self._policy_env_info = policy_env_info
        self._role = role or STARTER_ROLE_CYCLE[agent_id % len(STARTER_ROLE_CYCLE)]
        ...
```

The starter cycles per-agent roles `("miner", "aligner")` by `agent_id % 2`. Role-locked variants (`MinerRolePolicy`, `AlignerRolePolicy`, …) live in `cogames/policy/role_policies.py` and just set `_role` on the same base.

### Action method signature

From `cogames/policy/starter_agent.py:186` (`step_with_state`) and `mettagrid/policy/policy.py:53,316` (`step`):

```python
# AgentPolicy.step is the universal call used by rollout code:
def step(self, obs: AgentObservation) -> Action: ...
```

- `obs`: `mettagrid.simulator.interface.AgentObservation` — a token-stream object (`obs.tokens` is iterable of `ObservationToken` with `.feature`, `.value`, `.location`). **It is not the raw uint8 buffer**; it's the per-agent decoded observation. Obtain it via `env.current_simulation.observations() -> list[AgentObservation]` (one per agent, in agent_id order).
- Return: `Action(name=str)`. Convert to the env's int encoding via `env.current_simulation.action_ids[action.name]` (yields an int in `[0, 12)` covering the 5 primaries + 7 vibe-change names; `PufferMettaGridEnv.step` accepts this directly in its 1-D combined-int input).

For trainable / batched policies, `MultiAgentPolicy.step_batch(raw_observations, raw_actions) -> None` writes actions in place into the provided `raw_actions` buffer (shape `(num_agents,)`, dtype `int32`) given raw obs of shape `(num_agents, num_tokens, token_dim)`.

### Verbatim contents of `recon/policies.txt`

```
bitworld_random_action      mettagrid.policy.bitworld.BitWorldRandomActionPolicy
bitworld_random_dpad          mettagrid.policy.bitworld.BitWorldRandomDpadPolicy
noop                                            mettagrid.policy.noop.NoopPolicy
random                      mettagrid.policy.random_agent.RandomMultiAgentPolicy
chaos-monkey                       cogames.policy.chaos_monkey.ChaosMonkeyPolicy
substrate_memory        cogames.policy.cognitive_substrate_diagnostics.Substrat…
substrate_exploration   cogames.policy.cognitive_substrate_diagnostics.Substrat…
substrate_planning      cogames.policy.cognitive_substrate_diagnostics.Substrat…
starter                               cogames.policy.starter_agent.StarterPolicy
miner                               cogames.policy.role_policies.MinerRolePolicy
scout                               cogames.policy.role_policies.ScoutRolePolicy
aligner                           cogames.policy.role_policies.AlignerRolePolicy
scrambler                       cogames.policy.role_policies.ScramblerRolePolicy
tutorial_noop           cogames.policy.tutorial_overlay_policy.TutorialOverlayP…
custom                                                  path.to.your.PolicyClass
```

### Instantiation example

```python
from mettagrid.policy.random_agent import RandomMultiAgentPolicy
from cogames.policy.starter_agent import StarterPolicy

pei = env._policy_env_info            # PolicyEnvInterface
rand    = RandomMultiAgentPolicy(pei)
starter = StarterPolicy(pei)

ap_rand = [rand.agent_policy(i)    for i in range(env.num_agents)]
ap_star = [starter.agent_policy(i) for i in range(env.num_agents)]
for ap in ap_star:
    ap.reset(simulation=env.current_simulation)   # starter needs the sim for state init
```

The starter's per-agent policies are `StatefulAgentPolicy[StarterCogState]` and **require `reset(simulation=...)` before the first `step()`** — without it, `_initialize_state` raises because `_simulation` is `None`. Random has no such requirement (its `reset()` is a no-op).

## 6.5 Smoke-Test Scores

| Mission | Cogs | Steps | Episodes | Seed | Policy | Team return | Per-cog return |
|---|---|---|---|---|---|---|---|
| arena | 4 | 100 | 20 | 42 | random  | 0.000 (all 20 episodes) | 0.000 |
| arena | 4 | 100 | 20 | 42 | starter | 7.234 ± 1.156 | 1.808 ± 0.289 |

**Caveat.** Episode length was **100 steps**, not the locked-for-data-collection 1500 (per [PROJECT_PLAN.md §6 Decision B](../../research/reference/ENVIRONMENT.md)). At 100 steps, random has insufficient horizon to reach a junction, so the 0.000 score reflects horizon-limited exploration, not a broken policy. Starter scores will scale (non-linearly) with the 15× longer 1500-step rollouts used in Chunk 2.

## 6.6 Introspection Script Output

`recon_introspect.py` was **not produced during Chunk 1** as a standalone script. The action-space enumeration below was generated programmatically during the recon run from `env._policy_env_info.action_names` (length 5) and `env._policy_env_info.vibe_action_names` (length 7) and captured in [chunk_1_followup.md Q2](chunk_1_followup.md).

### Action-space enumeration (40 actions)

```
 0: noop
 1: move_north
 2: move_south
 3: move_west
 4: move_east
 5: noop + change_vibe_default
 6: noop + change_vibe_heart
 7: noop + change_vibe_gear
 8: noop + change_vibe_scrambler
 9: noop + change_vibe_aligner
10: noop + change_vibe_miner
11: noop + change_vibe_scout
12: move_north + change_vibe_default
13: move_north + change_vibe_heart
14: move_north + change_vibe_gear
15: move_north + change_vibe_scrambler
16: move_north + change_vibe_aligner
17: move_north + change_vibe_miner
18: move_north + change_vibe_scout
19: move_south + change_vibe_default
20: move_south + change_vibe_heart
21: move_south + change_vibe_gear
22: move_south + change_vibe_scrambler
23: move_south + change_vibe_aligner
24: move_south + change_vibe_miner
25: move_south + change_vibe_scout
26: move_west + change_vibe_default
27: move_west + change_vibe_heart
28: move_west + change_vibe_gear
29: move_west + change_vibe_scrambler
30: move_west + change_vibe_aligner
31: move_west + change_vibe_miner
32: move_west + change_vibe_scout
33: move_east + change_vibe_default
34: move_east + change_vibe_heart
35: move_east + change_vibe_gear
36: move_east + change_vibe_scrambler
37: move_east + change_vibe_aligner
38: move_east + change_vibe_miner
39: move_east + change_vibe_scout
```

`env.single_action_space = Discrete(5)`, `env.single_vibe_action_space = Discrete(7)`, `env.single_transport_action_space = Discrete(40)`.

### Decoding rule (combined → primary, vibe)

From `mettagrid/envs/mettagrid_puffer_env.py:336-369`:

```
[0, N_primary)                                → primary action only, no vibe
[N_primary, N_primary + N_primary * N_vibe)   → primary + vibe
  where offset = val - N_primary, primary = offset // N_vibe, vibe = offset % N_vibe
```

With `N_primary = 5`, `N_vibe = 7`: `i ∈ [0, 5)` → primary `i`, vibe unchanged; `i ∈ [5, 40)` → primary `(i-5) // 7`, vibe `(i-5) % 7`. Total = 5 + 5×7 = **40** ✓. Clean product, primary as major axis.

### Feature / tag counts

**UNKNOWN — reason:** the work order references "(74 features, 21 tags)" but no such counts appear in the on-disk Chunk 1 artifacts (`chunk_1_followup.md`, `cli_help.txt`, `cli_run_help.txt`, `inventory_missions.txt`, `policies.txt`). Quantities may have been mentioned in the original chat summary that was not persisted. **Deferred to Chunk 2** — verify by reading `env._policy_env_info.feature_names` and `env._policy_env_info.tag_names` (or equivalent) during the data-collection pipeline build-out.

## 6.7 Files Produced

In `recon/` after this consolidation task:

- [chunk_1_recon_report.md](chunk_1_recon_report.md) — this file (consolidated authoritative report).
- [chunk_1_followup.md](chunk_1_followup.md) — historical record of how the two corrections (5-tuple step return, action-space breakdown) were re-derived from the installed packages.
- [cli_help.txt](cli_help.txt) — `cogames --help` output (top-level CLI surface).
- [cli_run_help.txt](cli_run_help.txt) — `cogames run --help` output (the canonical evaluation command).
- [inventory_missions.txt](inventory_missions.txt) — captured the failure of `cogames missions` (no such command — confirms README is stale; see PROJECT_PLAN §5.7).
- [policies.txt](policies.txt) — `cogames policies` output (15 entries, registered policy shorthand names).

**Not produced during Chunk 1** (deferred or not required):
- `recon_introspect.py` — not written; action enumeration was inline (see §6.6 above).
- `recon_harness.py` — not written; recon snippets were executed inline as described in [chunk_1_followup.md](chunk_1_followup.md).
- `template_trainable.py`, `template_scripted.py` — not produced; templates will be drafted as part of Chunk 4 (DT scaffold) and Chunk 2 (data collection), respectively.
- `install_status.txt` — not produced as a standalone artifact; reconstructed into [§6.1](#61-install-status) at consolidation time.

---

## Corrections to Initial Assumptions

Two corrections to the Chunk 1 v5 work order's framing were discovered during execution and verified end-to-end. Both are already mirrored in [PROJECT_PLAN.md §5](../../research/reference/ENVIRONMENT.md); reproduced here for the standalone record.

1. **5-tuple step return, not 4-tuple.** `env.step(actions)` returns `(obs, rewards, terminals, truncations, info)` (Gymnasium style). The work order quoted a 4-tuple `(obs, reward, done, info)`. Per-cog arrays for `rewards` (float32), `terminals` (bool), `truncations` (bool), each of shape `(num_agents,)`. Source: `mettagrid/envs/mettagrid_puffer_env.py:296`. Implication for Chunk 2: data collector must persist `terminals` and `truncations` separately for RTG handling under truncation.

2. **Action-space breakdown is inverted from the initial framing.** The initial framing was "5 moves × 7 vibes = 35 base + 5 'special' extras." The actual encoding is the other way around: **5 primary-only indices `[0, 5)`** (`noop, move_N, move_S, move_W, move_E`) with vibe **unchanged**, and **35 paired (primary, vibe) indices `[5, 40)`** = product of 5 × 7. There are no exotic actions hiding at the head — the "extras" are simply the primaries with no vibe change. Source: `mettagrid/envs/mettagrid_puffer_env.py:336-369`. Implication for the DT design: factored heads `Discrete(5) × Discrete(8-with-sentinel)` reproduce the manifold exactly; a flat `Discrete(40)` softmax also works and matches the env's native input.

---

*End of Chunk 1 Recon Report. Chunk 2 (Offline Data Collection Pipeline) is being drafted separately.*
