# Chunk 1 Follow-up

**Note on provenance.** The work order references a prior `recon/chunk_1_recon_report.md`
with sections §6.3, §6.4, §6.6 and a `recon_harness.py`. Neither file exists in this
repo (`recon/` only contains `cli_help.txt`, `cli_run_help.txt`, `inventory_missions.txt`,
`policies.txt`). Everything below was re-derived directly from the installed packages at
`/home/hananel/miniconda3/envs/cogames/lib/python3.12/site-packages/{mettagrid,cogsguard}`
and the in-tree `cogames` source at `src/cogames/`. Cited line numbers are from those
files; I have run every snippet end-to-end before pasting.

> All "verbatim" quotes below are verbatim from those source files. The "compilable
> snippet" was executed against the installed env and produced the printed output shown.

---

## Q1 — Env construction

### 1. Compilable snippet

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

**Correction to the work order's stated contract.** `env.step()` returns a **5-tuple**
`(obs, rewards, terminals, truncations, info)` (Gymnasium style), not the 4-tuple
`(obs, reward, done, info)` the work order quotes. `rewards`, `terminals`, `truncations`
are all per-cog arrays of length `num_agents` (dtypes `float32`, `bool`, `bool`).
Source: `mettagrid/envs/mettagrid_puffer_env.py:296` (signature) and lines 285–293 / 396–\*
(behavior).

### 2. Bridge imports

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
- `mettagrid/__init__.py:49,54,68` (lazy re-exports of `Simulator` and the
  `MettaGridPufferEnv → PufferMettaGridEnv` alias).
- `cogsguard/missions/arena.py:53` (`make_basic_mission`).
- `cogsguard/missions/mission.py:13` (`class CvCMission`).
- `cogames/core.py:163` (`CoGameMission.make_env() -> MettaGridConfig`).

There is **no `from cogsguard import make_env`** style top-level bridge — you go through
the mission factory.

### 3. `PufferMettaGridEnv(...)` constructor

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

| param | type | meaning |
|---|---|---|
| `simulator` | `Simulator` | event-handler hub; can attach `StatsTracker`, `EarlyResetHandler`, etc. (see `cogames/train.py:481-483` for a typical attachment pattern). |
| `cfg` | `MettaGridConfig` | the pydantic config returned by `mission.make_env()`. |
| `supervisor_policy_spec` | `PolicySpec \| None` | optional supervisor that authors vibe actions on the cog's behalf — leave `None` for our purposes. |
| `step_info_keys` | `Sequence[str] \| None` | which extra stats to surface in the `info` dict each step (`"game/...", "team/...", "attributes/...", "agent/..."`). |
| `buf` | `Any` | optional pre-allocated PufferLib buffers (for vectorized rollouts). `None` is fine for single-env work. |
| `seed` | `int` | episode seed (default `0`). Passed straight to `Simulator.new_simulation(...)` at line 205. |

**RNG knobs.**
- Constructor takes only `seed=`. **No `map_seed=` on `PufferMettaGridEnv` itself.**
- `env.reset(seed=...)` (line 285) overrides `self._current_seed` for the next sim.
- `map_seed` exists one layer up in the *training* wrapper `_EnvBuilder` (`cogames/train.py:451, 478-480`), where it's combined with the per-env seed offset to derive `map_builder.seed`. If you want deterministic maps in our DT pipeline, mutate
  `cfg.game.map_builder.seed` (a `MapGen.Config.seed: int | None`) before constructing
  the env — that's the same hook the trainer uses.

### 4. Per-cog reward source line

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

The write is at [machina_1.py:160](../../../../home/hananel/miniconda3/envs/cogames/lib/python3.12/site-packages/cogsguard/missions/machina_1.py#L160) (in `Machina1Variant.modify_env`, the
`for agent in env.game.agents` loop, lines 152–164). Each agent gets its **own**
`agent.rewards["aligned_junction_held"]` entry — the loop iterates per agent and per
team, so the stream is genuinely per-cog, not team-broadcast.

Verified at runtime: `env.step(...)` returned `rewards.shape == (num_agents,)` with
dtype `float32` (per-cog vector).

Caveat — there is exactly one `agent.rewards[...]` write in cogsguard
(`grep -rn 'agent\.rewards\b' cogsguard/`). Additional per-cog reward terms (heart
pickups, gear, damage, etc.) are wired through other mechanisms (events / global
handlers / mutations) and aggregated by the simulator into the per-cog
`rewards` buffer. The buffer is **shape (num_agents,)** but it is the *sum* of all
configured per-cog reward terms for that step, not a decomposed multi-term tensor.

---

## Q2 — Action space (`Discrete(40)`)

### 1. Ordered list of all 40 action names

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

Generated programmatically from `env._policy_env_info.action_names`
(`['noop','move_north','move_south','move_west','move_east']`, length 5) and
`env._policy_env_info.vibe_action_names`
(`['change_vibe_default','change_vibe_heart','change_vibe_gear','change_vibe_scrambler','change_vibe_aligner','change_vibe_miner','change_vibe_scout']`, length 7).

`env.single_action_space = Discrete(5)`, `env.single_vibe_action_space = Discrete(7)`,
`env.single_transport_action_space = Discrete(40)`.

### 2. Step call signature

`env.step(actions)` accepts **a single 1-D `np.ndarray` of shape `(num_agents,)`,
dtype `int32`** (combined-index encoding into the `Discrete(40)` space). It also accepts
a 2-D form `(num_agents, 2)` with explicit `[primary_idx, vibe_idx]` columns, but the
1-D combined form is the canonical one used in our recon run.

```python
# Verbatim from the recon run:
actions = np.zeros(env.num_agents, dtype=np.int32)
for i, ap in enumerate(ap_rand):
    a = ap.step(per_agent_obs[i])           # AgentPolicy.step -> Action(name=str)
    actions[i] = action_ids[a.name]         # action_ids = dict(sim.action_ids) ∈ [0,12)
obs, rewards, terminals, truncations, info = env.step(actions)
```

Per-cog policies emit `Action(name=str)` objects, which you convert to ints via
`dict(sim.action_ids)`. Important: that dict maps the **flat** name space
`[*action_names, *vibe_action_names]` (12 entries: `noop=0 … change_vibe_scout=11`),
**not** the combined-40 space. The 12-entry flat encoding is what the env's 1-D step
accepts in the `[0, 12)` shorthand (see decoding rule below).

### 3. Decoding rule (combined → primary, vibe)

From `mettagrid/envs/mettagrid_puffer_env.py:336-369` (verbatim comment block):

```
[0, N_primary)                                → primary action only, no vibe
[N_primary, N_primary + N_primary * N_vibe)   → primary + vibe
  where offset = val - N_primary, primary = offset // N_vibe, vibe = offset % N_vibe

This means scripted agents can use indices from the combined action list
[*action_names, *vibe_action_names] directly — combined index i maps to:
  i < N_primary → primary action i, no vibe change
  i >= N_primary → noop primary + vibe action (i - N_primary)
```

With `N_primary = 5`, `N_vibe = 7`:
- `i ∈ [0, 5)`  → primary `i`, vibe unchanged
- `i ∈ [5, 40)` → primary `(i-5) // 7`, vibe `(i-5) % 7`
- Total: 5 + 5×7 = **40** ✓ (matches `single_transport_action_space = Discrete(40)`).

**It is a clean product** — the 35 paired (primary, vibe) combinations fill the upper
range exactly, with the major axis = primary (rows of 7) and the minor axis = vibe.

### 4. The "extras" — *inverted* from the work order's framing

The work order assumed 5 moves × 7 vibes = 35 base + 5 extras. The actual encoding is
the other way around:

- **5 primary-only indices (0–4)** = `{noop, move_N, move_S, move_W, move_E}` with the
  agent's current vibe **unchanged**. These are *not* "talk/communication/REST"
  specials — they are the same 5 primary actions with no vibe change.
- **35 paired (primary, vibe) indices (5–39)** = product of 5 primaries × 7 vibes.

So there are no exotic actions hiding in the head. There is also a separate
12-entry "flat shorthand" used by scripted/Nim policies — same 5 primaries plus the 7
vibes used as *vibe-only* actions (encoded as `noop primary + vibe` per the comment
above). Both encodings land on the same underlying C++ action space.

For the DT design decision: factoring into a primary-head (Discrete 5) × vibe-head
(Discrete 7 + a "no-change" sentinel) reproduces the action manifold exactly. A single
softmax over 40 also works and matches what `PufferMettaGridEnv` natively consumes.

---

## Q3 — Policy constructor patterns

### 1. Base classes (verbatim signatures)

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

CLI shorthand `random` resolves to `RandomMultiAgentPolicy` via `short_names = ["random"]`
(`mettagrid/policy/random_agent.py:46`). `starter` resolves to `StarterPolicy` via
`short_names = ["starter"]` (`cogames/policy/starter_agent.py:469`).

### 2. `RandomMultiAgentPolicy.__init__`

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

It samples a category (`primary` vs `vibe`) weighted by `vibe_action_p`, then a uniform
name from that category — so it returns vibe-only actions 50 % of the time by default
(`vibe_action_p=0.5`).

### 3. `StarterPolicy.__init__`

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

The implementation class (`starter_agent.py:56-67`):

```python
class StarterCogPolicyImpl(StatefulPolicyImpl[StarterCogState]):
    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int, role: str | None = None):
        self._policy_env_info = policy_env_info
        self._role = role or STARTER_ROLE_CYCLE[agent_id % len(STARTER_ROLE_CYCLE)]
        ...
```

The starter cycles per-agent roles `("miner", "aligner")` by `agent_id % 2`. Role-locked
variants (`MinerRolePolicy`, `AlignerRolePolicy`, etc.) live in
`cogames/policy/role_policies.py` and just set `_role` on the same base.

### 4. Action method signature

From `cogames/policy/starter_agent.py:186` (`step_with_state`) and
`mettagrid/policy/policy.py:53,316` (`step`):

```python
# AgentPolicy.step is the universal call used by rollout code:
def step(self, obs: AgentObservation) -> Action: ...
```

- `obs`: `mettagrid.simulator.interface.AgentObservation` — a token-stream object
  (`obs.tokens` is iterable of `ObservationToken` with `.feature`, `.value`, `.location`).
  **It is not the raw uint8 buffer**; it's the per-agent decoded observation.
  Obtain it via `env.current_simulation.observations() -> list[AgentObservation]`
  (one per agent, in agent_id order).
- Return: `Action(name=str)`. Convert to the env's int encoding via
  `env.current_simulation.action_ids[action.name]` (yields an int in `[0, 12)` covering
  the 5 primaries + 7 vibe-change names; `PufferMettaGridEnv.step` accepts this directly
  in its 1-D combined-int input).

For trainable / batched policies, `MultiAgentPolicy.step_batch(raw_observations,
raw_actions) -> None` writes actions in place into the provided `raw_actions` buffer
(shape `(num_agents,)`, dtype `int32`) given raw obs of shape
`(num_agents, num_tokens, token_dim)`.

### 5. Verbatim contents of `recon/cli_policies.txt`

That file does not exist. The directory only has `cli_help.txt`, `cli_run_help.txt`,
`inventory_missions.txt`, `policies.txt`. The closest match — `policies.txt` — was already
produced (15 lines) and reads:

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

### 6. Instantiation lines (no `recon_harness.py` exists — these are from the recon run executed for this report)

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

The starter's per-agent policies are `StatefulAgentPolicy[StarterCogState]` and
**require `reset(simulation=...)` before the first `step()`** — without it,
`_initialize_state` raises because `_simulation` is `None`. Random has no such
requirement (its `reset()` is a no-op).

---

## Cross-cutting notes for Chunk 2

1. **5-tuple step return**, not 4-tuple — DT data collector must store `(obs, action,
   reward, terminal, truncation)` per-cog per-step, not `(obs, action, reward, done)`.
2. **Per-cog reward is a single scalar** per step (the sum of all configured reward
   terms). To decompose into `aligned_junction_held` etc., add `step_info_keys` like
   `("agent/reward_step", "agent/reward_episode")` and any `c_sim.get_agent_stat(...)`
   keys you want — the env will surface them in `info["_per_agent_infos"][agent_idx]`.
3. **`num_cogs`** — `make_basic_mission(num_cogs=4)` does **not** propagate to the env
   (CvCMission's defaults override). Use `make_basic_mission().with_cogs(4)`; this also
   updates the arena's `spawn_count` so the map will fit. Verified: that path gave
   `env.num_agents == 4`.
4. **Map seed control** — set `cfg.game.map_builder.seed` before constructing
   `PufferMettaGridEnv` (the constructor has no `map_seed=` knob; `seed=` only controls
   simulation RNG).
5. **Combined-40 encoding is what the env wants** — collecting actions in the
   12-entry flat shorthand works for scripted-policy rollouts (their
   `Action(name).action_ids[...]` lookup lands in `[0, 12)`), but a learned DT predicting
   over `Discrete(40)` should emit/decode in the combined encoding directly.
