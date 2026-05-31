# Coding Agent Quickstart

This repository contains the game engine, agents, and experiment data for
"Bayesian Social Deduction with Graph-Informed Language Models." The main
research agent is GRAIL: an Avalon agent that combines a probabilistic factor
graph with language-model reasoning.

## Repository Map

- `README.md`: research/project overview, paper links, dataset notes.
- `code/`: runnable game engine and agent code.
- `code/docker-compose.yml`: local multi-service setup for the browser client,
  Colyseus game server, agent manager, and per-seat agent services.
- `code/phaser/client/`: Phaser/browser frontend served on port `1234`.
- `code/phaser/server/`: Node/Colyseus game server served on port `2567`.
- `code/agent/`: Python FastAPI agent manager, agent implementations, TypeChat,
  GRAIL models, ReCon baseline, and DeepSeek/OpenAI reasoning agents.
- `code/agent/our/`: GRAIL factor-graph implementation and pretrained model
  artifacts.
- `code/phaser/server/logs/`: game history JSON logs.
- `code/agent/logs/`: GRAIL agent dumps/logs when GRAIL is used.
- `dataset/`: formatted human-experiment data.

## Local Run Shape

Run Docker Compose from `code/`, or pass the compose file explicitly from the
repo root:

```bash
docker compose -f code/docker-compose.yml build
docker compose -f code/docker-compose.yml up
```

The client is at `http://localhost:1234/`. The admin/spectator UI is at
`http://localhost:1234/admin`. The server listens on `2567`, the agent manager
on `23003`, and individual agent containers on `23005` through `23010`.

If a fresh room does not appear after a restart, recreate the role containers
after `agentmanager` is up:

```bash
docker compose -f code/docker-compose.yml up -d --force-recreate minion-1 minion-2 servant-1 servant-2 servant-3 servant-4
```

Then inspect:

```bash
docker compose -f code/docker-compose.yml logs --tail 300 agentmanager server servant-2 servant-3 servant-4 minion-1 minion-2
```

Look for `Game with ID XXXX created with 6 agents`.

## Configuration

The root `.env` is the active environment file; `code/.env` is a symlink to it.
It is intentionally ignored by git because it contains API keys.

Important `.env` keys:

- `OPENAI_API_KEY`
- `DEEPSEEK_API_KEY`
- `UI_DRIVEN`
- `SERVANT1` through `SERVANT4`
- `MINION1` and `MINION2`

Seat values are agent type identifiers consumed by `code/agent/agent.py`.
Common values:

- `human`: human-controlled player.
- `ours`: full GRAIL agent.
- `ours_graph_only`: GRAIL graph-only ablation.
- `ours_llm_only`: GRAIL LLM-only ablation.
- `reason_openai`: OpenAI reasoning-style baseline.
- `reason_bl`: DeepSeek reasoning baseline.
- `random`: random test agent that can progress a game but says placeholder
  text such as `Idk rn tbh`.
- `recon` / `reconmod`: ReCon baseline variants.
- `hf`: Hugging Face agent path.

Model knobs live in all three config files, because different services mount
different files:

- `code/config.json`
- `code/config_servant.json`
- `code/config_minion.json`

Within the `agent` object:

- `model`: underlying model used by GRAIL.
- `openai_model`: model used by `reason_openai`.
- `deepseek_model`, `deepseek_base_url`, `deepseek_use_ollama`: DeepSeek path.
- `use_json_mode`: enables JSON mode for the OpenAI TypeChat path.

## Current Working Configuration Notes

This workspace has recently been tested with one human servant, GRAIL servants,
and DeepSeek minions:

```bash
SERVANT1=human
SERVANT2=ours
SERVANT3=ours
SERVANT4=ours
MINION1=reason_bl
MINION2=reason_bl
```

The OpenAI reasoning-agent path was also debugged recently. If agents respond
with `I'm thinking about the current situation.`, that is a fallback from
`DeepSeekAgent` after TypeChat fails, not a high-quality model response. One
known cause was a missing TypeChat output directory. The required directory is:

```text
code/agent/TypeChat/typechat/schemas/out/
```

There is also a noisy startup warning:

```text
ModuleNotFoundError: No module named '_distutils_hack'
Remainder of file ignored
```

This warning has not, by itself, prevented services from starting.

## Agent Flow

`agentmanager` reads the configured roles, waits for each non-human role
container to register, and then asks the Colyseus server to create an Avalon
room. The game server then calls each agent service for state updates, private
role data, messages, typing state, and turn actions.

Important files:

- `code/agent/agent_manager.py`: room creation and registration orchestration.
- `code/agent/agent.py`: per-agent FastAPI service and agent factory.
- `code/agent/agent_acl.py`: full GRAIL implementation.
- `code/agent/agent_deepseek.py`: DeepSeek reasoning baseline.
- `code/agent/agent_o1.py`: OpenAI reasoning baseline wrapper.
- `code/agent/agent_test.py`: random placeholder baseline.
- `code/phaser/server/src/rooms/AvalonGame.ts`: main game room logic.

## Testing Checklist

Useful manual coverage:

- Build and boot Compose.
- Confirm a room id is created in server/agentmanager logs.
- Register/login through `http://localhost:1234/`.
- Join the current room id as the configured human seat.
- Verify lobby, private role reveal, discussion, party proposal, party vote,
  quest vote, quest result, failed-party-vote counter, and game end.
- Test admin view at `http://localhost:1234/admin`.
- Verify generated logs in `code/phaser/server/logs/`.
- For GRAIL games, check `code/agent/logs/`.
- Replay existing logs by copying a JSON log into `code/phaser/server/logs/`
  and opening the admin UI with the matching four-letter room id.

For automated or service-level debugging, start with Docker logs. Avoid printing
`.env` wholesale because it contains API keys.

