# R1: OpenAI Responses API dialect (design note)

Source: my knowledge of the Responses API as of mid-2026. Items marked (?) are unverified; check them against
https://platform.openai.com/docs/api-reference/responses and a captured Codex CLI trace before implementing.

## Wire shapes

### Request `POST /v1/responses`
```jsonc
{
  "model": "gpt-5-codex",
  "instructions": "system prompt text",            // = NormRequest.system (plus leading system/developer input messages)
  "input": "text" | [ITEM, ...],
  "tools": [{"type": "function", "name": "shell", "description": "...", "parameters": {...}, "strict": false},
            {"type": "custom", "name": "apply_patch", "format": {...}} /* (?) Codex freeform tool */],
  "tool_choice": "auto" | "required" | {"type": "function", "name": "..."},
  "parallel_tool_calls": true,
  "stream": true,
  "store": false,                                   // Codex: store=false + full input each turn (?)
  "include": ["reasoning.encrypted_content"],
  "reasoning": {"effort": "medium", "summary": "auto"},
  "previous_response_id": "resp_...",               // stateful mode: input then holds only NEW items
  "prompt_cache_key": "...", "max_output_tokens": 1024, "text": {...}
}
```
Input ITEMs:
- message: `{"type":"message","role":"user"|"assistant"|"system"|"developer","content":"..." | [{"type":"input_text","text":..}|{"type":"input_image",..}|{"type":"output_text","text":..}]}` (the `type` may be omitted: "easy input message")
- `{"type":"function_call","id":"fc_...","call_id":"call_...","name":"shell","arguments":"{json string}","status":"completed"}`
- `{"type":"function_call_output","call_id":"call_...","output":"text"}` (pairs on **call_id**, not id)
- `{"type":"reasoning","id":"rs_...","summary":[...],"encrypted_content":"..."}`
- (?) `{"type":"custom_tool_call","call_id":..,"name":"apply_patch","input":"*** Begin Patch..."}` / `custom_tool_call_output`
- (?) older Codex: `local_shell_call {call_id, action:{type:"exec", command:[argv], workdir, timeout_ms}}` / `local_shell_call_output`

### Response (non-stream)
```jsonc
{"id":"resp_...","object":"response","created_at":1759000000,"status":"completed"|"incomplete","model":"...",
 "output":[{"type":"reasoning",...},
           {"type":"function_call","id":"fc_...","call_id":"call_...","name":"shell","arguments":"{...}","status":"completed"},
           {"type":"message","id":"msg_...","role":"assistant","status":"completed","content":[{"type":"output_text","text":"...","annotations":[]}]}],
 "incomplete_details": null | {"reason":"max_output_tokens"|"content_filter"},
 "usage":{"input_tokens":N,"input_tokens_details":{"cached_tokens":C},"output_tokens":M,"output_tokens_details":{"reasoning_tokens":R},"total_tokens":T}}
```
`input_tokens` includes cached tokens (same convention as Chat Completions `prompt_tokens`).

### Streaming (SSE; `event:` line = `data.type`; every data has `sequence_number`; no `[DONE]` sentinel (?))
```
response.created                 {response:{id,status:"in_progress",output:[]...}}
response.in_progress             {response}
response.output_item.added       {output_index, item:{type:"function_call", id, call_id, name, arguments:"", status:"in_progress"}}
response.function_call_arguments.delta {item_id, output_index, delta}
response.function_call_arguments.done  {item_id, output_index, arguments}
response.output_item.done        {output_index, item:{...complete...}}
response.output_item.added       {output_index, item:{type:"message", ...content:[]}}
response.content_part.added      {item_id, output_index, content_index, part:{type:"output_text",text:""}}
response.output_text.delta       {item_id, output_index, content_index, delta}
response.output_text.done / response.content_part.done / response.output_item.done
response.reasoning_summary_part.added / response.reasoning_summary_text.delta / ...   (reasoning models)
response.completed | response.incomplete | response.failed   {response:{... full output + usage}}
error                            {code, message}
```

## `dialects.Responses(Dialect)` (mirrors Anthropic/OpenAI)
- `name = "responses"`, `call_prefix = "call"`; replayed call ids `call_tj_<node>_<conf><rand>[_via]` go in **call_id**;
  the item `id` is `fc_tj<rand>` (never used for recognition).
- `parse_request`: system = `instructions` + leading `system`/`developer` messages; tools = function tools
  `{name, schema: parameters, description}` (flat, not under `function`), custom tools with `schema: {"format": ...}`;
  task = last `user` message (strip `<environment_context>`/`<user_instructions>` wrappers like `strip_reminders`);
  steps: `function_call`/`custom_tool_call`/`local_shell_call` → ToolCall(call_id, name, json.loads(arguments) | {"input": s} | {"command": argv}),
  matched by call_id to `*_output`. `ready` = last item is an output item or a user message and every call has output.
  **`previous_response_id` set → the episode is incomplete**: treejit must reconstruct it (below) or pass through untouched (tier `pass`, note `stateful`).
- `build_response(model, calls)`: `{"id":"resp_tj..","object":"response","status":"completed","output":[function_call items],"usage":{zeros}}`.
- `build_sse`: `response.created` → per call `output_item.added` → `function_call_arguments.delta` (whole JSON) → `.done` → `output_item.done` → `response.completed` (with the full response), each with `sequence_number`.
- `parse_response` / `_ResponsesStream`: calls from `output[]` (or from `response.output_item.done` items / `response.completed.response.output`);
  `stop_reason` = `"tool_calls"` if any call, `"incomplete:"+reason` for `status=="incomplete"` (add to `engine.TRUNCATED`), else `"stop"`;
  usage = `Usage(input - cached, output, cached, 0)`.
- `inject_hint`: append `{"type":"message","role":"user","content":[{"type":"input_text","text":hint}]}` at the end of `input`
  (never edit `instructions`: that would break the prompt cache).
- `prepare_forward`: drop the `id` field of replayed `function_call` items (`fc_tj...`), which the server has never seen (?) — with
  `store:true` the server may try to resolve item ids, and reasoning models complain about a `function_call` whose paired
  reasoning item is missing (seen as "Item 'fc_..' of type 'function_call' was provided without its required 'reasoning' item" (?)).
  Replayed turns carry no reasoning items, analogous to the Anthropic `thinking` rule.

## Statefulness (the hard part)
- `store:false` + full `input` each turn (Codex CLI, I believe): works like Chat Completions; nothing extra.
- `previous_response_id`: the client sends only new items. A replayed response has id `resp_tj...` unknown upstream, so the next
  request would 400. Needed: a `responses` table `(id, parent_id, upstream_id, items_json, ts)`; on each request expand the chain
  (`parse_request` gets the reconstructed input), and on forward rewrite `previous_response_id` to the nearest *upstream* ancestor and
  prepend the synthetic items (replayed function_call + outputs) that came after it. Upstream responses also have to be recorded (their
  `output` items) because treejit cannot fetch them without the client's key (it could: `GET /v1/responses/{id}` with the forwarded auth (?)).
  Recommend phase 1 = stateless only; stateful requests go `pass` with a note.
- `conversation` objects (Conversations API): same, pass-through in phase 1.

## Codex specifics (?)
- `shell` tool args `{"command": ["bash","-lc","git status"], "workdir": "...", "timeout_ms": N}`: argv list. `policy`/`templates`/`shellwords`
  assume a string: normalize `["bash","-lc",S]`/`["sh","-c",S]` → S for shape/readonly checks and re-wrap when rendering; any other argv →
  `shlex.join`. Without this, every Codex shell call is "not read-only" and nothing replays unattended.
- `apply_patch` freeform tool: args are one string → one template slot; T3 can fill it, T0 rarely.

## Files touched
`dialects.py` (class + registry), `proxy.py` (`/v1/responses`, `/responses` routes → `openai_upstream`), `inline.py` (`client.responses.create`),
`compaction.py` (`_rewrite` branch: `function_call_output` by `call_id`), `subcalls.py` (build: `tool_choice {"type":"function","name":"treejit_fill"}`,
flat tool def; `tool_input`: parse `output[].arguments`), `engine.py` (`TRUNCATED`), `policy.py`/`templates.py` (argv shell), `store.py`
(only for stateful phase 2), tests: `tests/test_responses.py` (parse, replay JSON+SSE round-trip through `_ResponsesStream`, forward/complete usage,
compaction rewrite, subcall, proxy route, stateful request passes through untouched).
