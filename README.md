# AILicious

Personality-driven, multi-agent AI system with cloud-based inference and memory, accessible from Android via a web client. All compute runs server-side — no local hardware or model downloads required.

## Android / web client

A single-page PWA is served directly by the backend at `/` (`backend/static/index.html` — no build step, no separate host). Open the Render URL on an Android phone in Chrome, enter the API key in Settings (gear icon), pick an agent from the dropdown, and chat. "Add to Home Screen" installs it as a standalone app icon via the manifest + service worker in `backend/static/`.

The service worker (`sw.js`) fetches network-first, caching only as an offline fallback — every successful load re-fetches from the live server, so a new deploy reaches an already-installed client on its very next load. (An earlier cache-first version pinned whatever `index.html` a client first cached indefinitely, since `CACHE_NAME` never changed between deploys — a real fix could ship and a phone would keep silently serving the old app shell forever. If a very old client still looks stale after this fix ships, one manual site-data clear resets it; every load after that self-heals.)

Client-side notes:
- The API key is stored in the browser's `localStorage`, sent only as an `X-API-Key` header to this same-origin backend — never committed, never sent anywhere else.
- Each agent gets its own `session_id` and message history in `localStorage`, matching the backend's per-agent memory scoping — switching agents shows that agent's own conversation, not a mixed one.
- Verified end to end with a real browser (Playwright): entering a key, picking an agent, sending a message, and getting back a real Mistral-backed reply, plus confirming history correctly persists per agent when switching back and forth.
- A **SAGE panel** (Settings → "Persona (SAGE)") lets you trigger feedback analysis, review/accept/reject proposals, and reset an agent's persona — all from the phone, no API calls by hand. See SAGE below.
- **Ground-truth badges** under a NEXUS reply show what the backend actually did that turn — a "→ FORGE" chip when it really delegated, a search-icon chip with the query when it really searched the web — read straight from `delegated_to`/`searched_web` in the API response, not from whether NEXUS's own prose happened to mention it. Added after live-testing surfaced a real gap: NEXUS having a working `search_web` tool and simply not using it was invisible in the UI until it got fixed at the prompt level (see "Real web search" below) — with these badges, that kind of gap is visible immediately instead of depending on the model's self-report. Persisted in each agent's `localStorage` history alongside the message, so they survive a reload. Verified end to end with Playwright: badges render for a live response carrying both fields and still render correctly after a page reload (proving the `localStorage` round-trip, not just the initial render).

## Backend (Phase 1: minimal walking skeleton)

A FastAPI service that proxies chat messages to Mistral, with SQLite-compatible conversation memory (local file, or Turso for durability — see below) and six personas (NEXUS, FORGE, ORACLE, SENTINEL, CODEX, AVERY) loaded from YAML at startup.

- `POST /chat` — always talks to NEXUS, the default entry point.
- `POST /agents/{name}` — talk to a specific agent directly (`forge`, `oracle`, `sentinel`, `codex`, or `avery`); 404 if the name isn't a loaded persona.
- `GET /agents` — lists the loaded persona names.

Both chat endpoints accept an optional `session_id`; when included, the last 20 messages for that `(session_id, agent)` pair are sent back to Mistral as context. **Memory is scoped per agent, not just per session** — reusing the same `session_id` across different agents does not leak one agent's conversation into another's; each keeps its own thread of history even under a shared session. Persona text lives in `backend/config/personas/*.yaml` — edit those files to change how an agent talks, no code changes needed.

NEXUS's prompt is explicit that none of the other five agents have real-time data, live web access, or knowledge past their training cutoff, and that NEXUS's own web access exists only when a `search_web` tool is actually offered to it that turn (see below) — never assumed. It's equally explicit about what NEXUS *can* actually do: real delegation to the other five agents (see below), not just naming one for you to switch to. An earlier version of the prompt predated the other five agents and described them as not-yet-online, which led NEXUS to invent plausible-sounding fictional capabilities ("Research Agent is fetching...") instead of a real agent or a real capability gap — the same failure mode a prompt that unconditionally claimed web search would risk reintroducing if `TAVILY_API_KEY` were ever unset.

### Real agent delegation

NEXUS carries a `delegate_to_agent` tool (Mistral tool-calling, `tools`/`tool_choice: "auto"` on its chat completion request only — no other agent gets this tool, which structurally rules out delegation loops). When a request clearly fits a specialist better than general conversation, NEXUS can call this tool with `{agent, task}`; the backend runs that task through the exact same `run_agent_chat` path a direct user message would take — so the delegate agent's own memory and KAIROS digest see it exactly as if the user had asked it themselves — and NEXUS gets the real response back to synthesize into its final answer. `ChatResponse` gains a `delegated_to: list[str]` field listing which agents were actually consulted during a turn (empty when NEXUS answers directly).

A misbehaving or malformed tool call never breaks the turn: an unparseable arguments string, a hallucinated agent name outside the real five, or an error from the delegated call itself all come back as a graceful tool result NEXUS can react to, not a crash. A hard cap (`MAX_TOOL_ROUNDS_PER_TURN = 2`, shared with web search below — both are rounds of the same tool-calling loop) forces a plain final answer if the model won't stop calling tools, so a single chat turn can never spiral into unbounded tool-calling rounds.

Cost note: a turn where NEXUS decides to delegate costs an extra Mistral call — decide-to-delegate, the delegate's own turn, then NEXUS synthesizing — versus one call for a direct answer.

### Real web search

NEXUS also carries a `search_web` tool, added to its request only when `TAVILY_API_KEY` is set on the server — omitted entirely otherwise, so an unconfigured deployment never offers a tool it can't back with a real search (and the persona prompt above is written to match: it describes the capability as conditional on the tool actually being offered, not as a standing fact about NEXUS). When offered and called, `{query}` goes to the Tavily search API; the top few results (title, snippet, URL) come back as the tool result for NEXUS to ground its answer in and cite, the same way a delegate's response is fed back. `ChatResponse` gains a `searched_web: list[str]` field listing the actual queries run during a turn.

Same dispatch loop as delegation (same `MAX_TOOL_ROUNDS_PER_TURN` cap, same graceful-failure philosophy — malformed arguments or a Tavily error come back as a tool result NEXUS can react to, never a crash), with one deliberately explicit piece: the tool-name dispatch checks for `delegate_to_agent` and `search_web` by name and has a distinct fallback for anything else, rather than treating an unrecognized name as a (malformed) delegation call by default — confirmed with a deliberate-break test that a name matching neither tool produces its own "unknown tool" result rather than silently being swallowed by the delegation path's own error handling.

Live-tested bug found right after this first shipped: `tool_choice: "auto"` leaves it entirely up to Mistral whether to actually call `search_web`, and NEXUS initially kept falling back to its old pre-search-tool reflex ("I have no live web access") on a clearly time-sensitive question, even with the tool present and working (confirmed separately by calling the Tavily API directly with the same key — real results, 200 OK). The wiring wasn't the problem; the persona prompt wasn't directive enough to overcome the model's trained habit. First fix: made the prompt imperative rather than merely permissive.

That reduced but didn't eliminate it — live-tested again after the prompt fix shipped, and NEXUS produced a confident, fully-formatted "today's news" digest with citation-style links, still without calling `search_web` (confirmed via the ground-truth badges below: none rendered). This is a worse failure than the honest decline it replaced — fabricated headlines with fake-looking sources are more convincing and more harmful than "I can't do that." `tool_choice: "auto"` simply doesn't give a hard guarantee, prompt wording or not.

The real fix is structural, not another prompt tweak: `_looks_time_sensitive()` (`main.py`) is a small keyword heuristic ("today," "news," "weather," "who won," etc.) checked against the current message. When it matches and `TAVILY_API_KEY` is configured, the *first* round of NEXUS's tool loop sets `tool_choice` to force the specific `search_web` function — not "auto," not "any tool," that exact one — so the model has no path to a direct answer without actually searching first. Every later round in the same turn reverts to `"auto"` (confirmed with a deliberate-break test — forcing every round, not just the first, is a distinct bug the test catches on its own), so NEXUS can still synthesize or decline further tool use normally once the forced search has actually happened. A message that doesn't match the heuristic, or a deployment without `TAVILY_API_KEY`, behaves exactly as before — this only tightens the specific case that was already proven to fail.

Third live-test bug, once the search was actually being forced: the forced search happened, but NEXUS cited a real headline to the wrong source — attributing a fact to a different result's URL than the one it actually appeared under. Caught by running the identical Tavily query independently and diffing the real results against what NEXUS said. Two things turned out to be true at once here: the underlying content wasn't fabricated (a real Iran/Bahrain military story was genuinely in the search results — surprising relative to this project's own training cutoff, but real), and the citation was still wrong (attributed to Military.com when the actual source was Task & Purpose). Root cause: `_execute_web_search`'s result formatting put every result's title/content/url on one bulleted line, and for a broad "today's X news" query, Tavily's top-scoring pages are often outlet homepages whose scraped content covers several distinct stories at once — easy to blur together which fact came from which URL when they're formatted as flat, undifferentiated lines. (Tried `search_depth: "advanced"` first, expecting deeper per-article links to fix this outright — it returns materially the same homepage-level URLs for this query shape, since a front page genuinely is the best match for "today's news"; not a bug in Tavily, just not the fix this needed.)

Fixed by reformatting search results as clearly delimited `[Source N] <url>` blocks (each with its own title and content, separated by `---`) instead of one flat bulleted line per result, and updating both the `search_web` tool description and NEXUS's system prompt to explicitly instruct citing the exact URL a fact's block actually appeared under — never mixing content from one source with a different source's URL, even when several sources cover similar topics.

Both chat endpoints are also rate-limited (default 20 requests/minute, per API key — or per client IP locally if `APP_API_KEY` is unset) and cap message length (default 4000 characters), to keep a leaked key or a client bug from running up unbounded Mistral usage. Tune via `RATE_LIMIT_PER_MINUTE` and `MAX_MESSAGE_LENGTH`; exceeding either returns a normal error the PWA already displays inline (429 with a `Retry-After` header, or 400).

Every chat response includes a `message_id`; `POST /feedback` (`{message_id, rating}`, rating `1` or `-1`) records a thumbs up/down on that specific reply. The PWA shows this as small thumb icons under each assistant message. This is entirely optional now — see KAIROS below — but a flagged message still gets folded into the next digest cycle as extra-weighted signal if you use it.

### KAIROS (automatic conversation digestion)

KAIROS originally scored individual messages by a length heuristic plus feedback and ranked which ones to keep in context. It's since been redesigned around a simpler idea: **understand the conversation as a whole, automatically, with no voting required**, rather than ranking messages one at a time.

Instead of a per-message score, each agent keeps one rolling summary — a `conversation_digests` table (`core/digest.py`), one row per agent (not per session: the agent's understanding accumulates across every conversation you have with it). Periodically, new raw turns get folded into that summary by an incremental Mistral call: *"here's what you currently know, here are the new exchanges since then — merge them: keep what's still true, revise what got corrected, add what's new, drop what's stale."* Because only the new slice is ever sent (never the whole transcript), the cost of each digest cycle stays flat no matter how long the relationship with an agent gets.

**Triggers** (checked after every message, no extra API calls to decide):
- **Count** — once 10 new turns have accumulated since the last digest.
- **Pause** — a request whose gap since the previous message exceeds 20 minutes (with at least 3 turns pending) triggers a digest of everything before it, treating the gap itself as the natural conversation boundary. No poller needed — this piggybacks on the very next message.
- **Adaptive** — the digest call reports back whether the new turns represented a topic shift; if so, the next threshold is halved, so a real change in subject gets picked up faster without a dedicated (and costly) drift detector.

**Coalescing** — an in-memory per-agent lock (`core/digest.py`) means at most one digest cycle runs per agent at a time; a trigger that fires while one is already running is a no-op, since the in-flight run reads current state when it actually executes and absorbs whatever accumulated.

**Backoff** — a failed digest call (Mistral down or slow) is never surfaced to the user; it's recorded and doubles the effective trigger threshold (capped at 4×), so a struggling API gets digested against less often rather than retried aggressively. A success resets it.

**Fire-and-forget** — digestion is scheduled via `asyncio.create_task` *after* the chat response is already built, so it adds zero latency to the reply the user sees.

At chat time, `run_agent_chat` injects the agent's current digest (if any) as extra context ahead of a short raw tail (the last 6 messages — the digest now carries the rest), instead of KAIROS's old ranked-message window. With no digest yet (a brand-new agent), this collapses to exactly the original plain-recency behavior — nothing changes until the agent has actually accumulated something to say.

**Implicit signal quality** — the same digest call also infers, from the user's own behavior rather than any explicit rating, what's working and what isn't: rephrasing or re-asking a question signals the prior answer missed the mark, an explicit correction ("no, I meant...") is a strong negative signal, building directly on an answer or affirming it ("exactly", "that works") is a strong positive signal, and silently changing topics is weak/neutral. This is stored as `signal_quality` on `conversation_digests` — a short, actionable note (e.g. "prefers short, direct answers over long ones"), merged forward on every cycle the same way the digest itself is, and left untouched (not reset to empty) on a cycle where nothing in that batch was informative either way. An explicit thumbs up/down (`POST /feedback`) is still folded in as a strong, direct signal when given, but it's no longer required — this is the actual replacement for manual voting as SAGE's input, not just a description of KAIROS.

This isn't a scheduled background worker — Render's free tier has no free cron, and the service sleeps when idle anyway — so every trigger, coalesce, and backoff decision above is computed inline, synchronously, from local state, with the one actual Mistral call always fired off in the background.

**Semantic recall** — the digest is a lossy compressed summary and the recency tail only covers the last 6 messages, so a specific detail from well outside both (something said 40 messages ago, in a different session entirely) can still be exactly what a new question needs but nowhere in the context the model sees. Every stored message now gets embedded in the background (`core/embeddings.py` + `main._schedule_embedding`, via Mistral's `mistral-embed` model — no new provider, reuses the already-configured `MISTRAL_API_KEY`) and, at chat time, the incoming message is embedded too and compared by cosine similarity against every previously-embedded message for that agent (capped at the most recent 500, cross-session like KAIROS's own digestion). The top 4 matches — excluding anything already in the flat recency tail, so nothing shows up twice under two different framings — get injected as one more system message: *"Potentially relevant exchanges from earlier (not already shown above), most relevant first."*

Similarity is plain Python (`sum`/`sqrt`, no library) rather than a real vector-DB ANN query — deliberately, at this app's actual scale (one user, message counts in the hundreds at most) a linear scan over 500 embeddings is effectively instant, and it sidesteps depending on how mature libSQL/Turso's native vector support actually is before that's been verified. Revisit if message volume ever makes the linear scan the bottleneck.

Honest tradeoffs, stated plainly rather than glossed over:
- **This is the one place semantic recall costs real latency.** Storage embedding is fire-and-forget background work, same as digestion — zero cost to the response the user sees. But *retrieving* for the current turn has to happen before that turn's own Mistral call, so it's a real extra network round-trip on every single chat message, not a background nicety. Best-effort throughout (`main._embed_text` returns `None` on any failure — no key, network error, malformed response — never raises), so a slow or down embeddings endpoint costs a bit of latency or silently skips recall; it never breaks the chat.
- **Not independently verified against the real Mistral embeddings API this session** — every other Mistral-backed feature in this project was checked against live responses before being called done; this one is built to Mistral's documented, OpenAI-compatible embeddings shape (`POST /v1/embeddings`, `{"model": "mistral-embed", "input": [...]}` → `{"data": [{"embedding": [...]}]}`) and covered by tests with a mocked response matching that shape, but I don't hold a Mistral key myself to place one real call and confirm the live response matches. Worth a live check after deploy the same way every other feature here got one.

### SAGE (persona evolution)

An agent's `system_prompt` can be overridden at runtime without touching its YAML file or redeploying. Overrides live in a `persona_overrides` table (`agent` primary key, one row per agent) and take precedence over the YAML default whenever present; `run_agent_chat` checks the override table before falling back to `get_system_prompt`.

- `POST /sage/override/{agent}` (`{system_prompt}`) — sets or replaces an agent's override directly. A manual/testing escape hatch — the proposal flow below calls this same storage after a human accepts a proposal.
- `POST /sage/reset/{agent}` — clears the override, reverting the agent to its YAML-defined persona.
- `GET /sage/current/{agent}` — returns `{agent, system_prompt, is_override}`, the exact prompt actually in effect right now (override if one exists, otherwise the YAML default). Exists specifically so the PWA can diff a proposal against reality rather than the proposal text alone.

An override never touches the YAML files themselves — which wouldn't survive Render's ephemeral filesystem anyway — so resetting is always instant and always available.

SAGE proposes revisions itself, on demand, from the agent's KAIROS digest — driven primarily by inferred signal quality, not by the digest's topic content:

- `POST /sage/analyze/{agent}` — sends the agent's current system prompt, its conversation digest, and its inferred `signal_quality` read to Mistral, instructed to treat the signal-quality read as the primary evidence for what to change (the digest is context, not a mandate to cover every topic ever discussed) and to propose only a minimal, low-risk change if that read is uninformative rather than inventing a rewrite from topic content alone. 400 if the agent has no digest yet, and a separate 400 if it has a digest but no signal-quality read yet — chat with it a bit first, naturally; no votes needed. The LLM's response is parsed strictly as `{"rationale": "...", "proposed_system_prompt": "..."}`; anything that doesn't parse as that exact shape is a 502, never silently stored as a proposal. A successful analysis is stored as a `pending` row in `sage_proposals` and returned.
- `GET /sage/proposals` (optional `?agent=&status=` filters) — lists proposals, most recent first.
- `POST /sage/proposals/{id}/accept` — writes the proposed prompt into `persona_overrides` (so it takes effect immediately) and marks the proposal `accepted`.
- `POST /sage/proposals/{id}/reject` — marks it `rejected`; the override table is untouched.
- Both transitions only apply to a `pending` proposal — acting on one that's already been accepted or rejected returns 409, so a proposal can't be double-applied or re-decided.

Since signal quality is inferred automatically from ordinary conversational behavior (see KAIROS above), SAGE has real, preference-grounded material to analyze without you ever touching a thumbs-up button — the gate that used to be "does a digest exist" is now "has anything informative about what works actually been inferred yet," which is the difference between SAGE reacting to what you *talked about* versus what you *responded well or poorly to*.

Nothing here is autonomous: analysis only runs when `/sage/analyze/{agent}` is called, and a proposal only changes live behavior once a human explicitly accepts it. The PWA panel (Settings → "Persona (SAGE)") drives all of this: an "Analyze feedback" button, a list of pending proposals with Accept/Reject, and "Reset persona to default" — the currently selected agent throughout, matching the chat view. Verified end to end with a real browser (Playwright): a seeded pending proposal renders correctly and shows its rationale/proposed prompt, accepting one actually writes the override (confirmed against the database, not just the UI), and rejecting one leaves the override untouched; the no-digest-yet error path from `/sage/analyze` also surfaces correctly in the panel instead of failing silently.

**Proposal diff view** — each pending proposal renders as a word-level diff against the agent's actual current prompt (fetched via `/sage/current/{agent}`), not the flat wall-of-text preview it used to be: removed words struck through, added words highlighted, unchanged text plain — so accepting is "here's exactly what changes" rather than re-reading two full prompts side by side to spot the difference yourself. Diff computed client-side (`index.html`) with a small word-tokenized LCS — no library, no network cost beyond the one `/sage/current` fetch already needed. If that fetch fails for any reason, the card falls back to the old full-text preview rather than failing to render the proposal at all; Accept/Reject stay fully functional either way, since neither depends on the diff succeeding. Verified with Playwright: correct words land in `diff-added` vs. `diff-removed` for a real added/removed/unchanged mix, and the fallback path renders correctly (with a 500 from `/sage/current`) with Accept present and clickable. One thing that Playwright pass surfaced and is worth knowing for future UI testing on this app specifically: `page.route()` mocks don't reach fetches the service worker intercepts (everything GET, per `sw.js`'s network-first handler) unless the browser context is created with `service_workers="block"` — a real gotcha the first attempt at this test hit blind (mocked routes silently never fired, real 404s came back from the static file server instead).

**Live bug: a deploy landed but a real reload kept running the old JS.** After the diff view above shipped, triggering a SAGE analysis and reloading the app twice, live, never produced a single `GET /sage/current/{agent}` in Render's logs — not even a failed one. `curl` against the live server confirmed the correct new code was actually being served, so this wasn't a bad deploy. `curl -sI` against the live server also showed why: it sends `last-modified`/`etag` but no `Cache-Control` header at all, which leaves a browser free to apply RFC 7234 heuristic freshness caching — satisfying a plain `fetch()` entirely from its own local HTTP cache, with zero request ever reaching the network, even on an ordinary reload. `sw.js`'s fetch handler was already labeled "network-first," but a bare `fetch(event.request)` only asks the network *first in priority*, not "for real, unconditionally" — it still honors the browser's own HTTP cache. Fixed by adding `{ cache: "no-store" }` to that `fetch()` call, forcing every request through to the network; the handler's own Cache Storage `.put()` immediately after already provides the offline-fallback caching this app actually wants, so the browser's separate HTTP-cache layer was never buying anything here except this exact staleness trap.

Tried to reproduce this properly before shipping it (per this project's usual break-it-then-fix-it rigor): a Playwright test against a from-scratch Python HTTP server serving the same header shape (`Last-Modified` present, no `Cache-Control`), bumping a version counter between load and reload to simulate a deploy. With the fix, the reload always picked up the bump. But reverting the fix (`fetch(event.request)`, no options) did *not* reproduce the staleness — the test server-side log showed a fresh `GET /` on every reload either way, fix or no fix, even after re-testing with a `Last-Modified` realistically backdated 10 days (the first pass used a same-day timestamp, giving heuristic freshness a ~zero window, which was itself worth catching). Best guess: Chromium's disk-cache heuristics behave differently under Playwright/CDP automation, or don't kick in the same way against a bare Python `http.server` (HTTP/1.0, no ETag, `Connection: close`) as they do against production's real stack (Cloudflare-fronted, HTTP/2, uvicorn, with an `etag`). So this fix is shipped on the strength of the HTTP spec and the direct match to the observed symptom, not a confirmed local reproduction — worth flagging plainly rather than claiming a rigor this particular check didn't actually deliver. The fix itself carries no real downside either way: it can only make the network-first handler's fetch actually hit the network, which is what it already claimed to do.

**Closing the loop: does an accepted change actually help?** Until now, accepting a SAGE proposal was the end of the story — the override applied and nothing ever checked whether it actually worked. That's human-in-the-loop prompt tuning, not a real feedback loop. Now:

- Accepting a proposal (`POST /sage/proposals/{id}/accept`) snapshots two things onto that proposal row: `baseline_signal_quality` (the signal-quality read that justified the change, captured at accept time) and `prior_override` (whatever override was in effect right before — or `null` if it was the YAML default), via `core.sage_proposals.record_acceptance_baseline`.
- The very next time that agent's digest cycle completes (`main._run_digest_safely`, right after KAIROS's own digest and the cross-agent profile merge), `core.sage_evaluation.run_pending_evaluation` checks for an accepted-but-unevaluated proposal and, if one exists, sends a judge call: the proposal's rationale plus the before/after signal-quality reads, asking for a strict `{"verdict": "improved"|"regressed"|"unclear", "reasoning": "..."}`.
- A `"regressed"` verdict actually reverts the override — back to `prior_override` if one existed, or cleared to the YAML default if it didn't — not just a note buried in a database column nobody reads. `"improved"` and `"unclear"` just record the outcome and leave the change in place.
- Same best-effort discipline as everything else in this pipeline: a malformed judge response is left unevaluated (retried on the next digest cycle) rather than applied or allowed to raise into the chat path that triggered it. Each proposal is only ever evaluated once — the first successful judge call clears the pending-evaluation state regardless of verdict.

Tested with the same rigor as the rest of the digest/SAGE machinery: round-trip storage for the new proposal columns, judge-call logic against a mocked Mistral response (including malformed-response and unrecognized-verdict paths, both of which must leave the proposal retriable rather than corrupting it), and two full end-to-end tests driving `main._run_digest_safely` directly — one where a `"regressed"` verdict actually flips `persona_overrides` back to the pre-acceptance state, one where an `"improved"` verdict leaves it untouched. Deliberate-break-confirmed: removing the revert-on-regression branch correctly fails all three tests that check for it (two unit-level, one end-to-end), restored once confirmed.

Honest limitation: the "before" and "after" signal-quality reads are still LLM-inferred summaries, not a hard metric — this closes the loop mechanically (something now checks and acts), but the judge call is itself fallible in the same way every other inference in this pipeline is. It's a real improvement over "nobody ever checks," not a guarantee of correctness.

**Made visible in the PWA.** The SAGE panel's pending-proposal list (`loadProposals()` in `index.html`) now also fetches every proposal for the agent, splits pending from reviewed, and renders the reviewed ones as a **History** list beneath it — each with a plain-language outcome badge: "Kept — helped" (improved), "Reverted — made things worse" (regressed), "Accepted — evaluating…" (still pending its first post-acceptance digest cycle), "Kept — too soon to tell" (unclear), or "Rejected." Accepting or rejecting a proposal now reloads the whole list rather than just removing the card, so the decision shows up in History immediately instead of only after the next dialog open. Verified with Playwright: a mocked mix of pending/improved/regressed/pending-evaluation/rejected proposals renders the correct badge text for each, and a live accept click against a mocked endpoint correctly moves the card out of pending and into History with an "evaluating" badge. Deliberate-break-confirmed: commenting out the history-render call correctly drops the rendered history count to zero, restored once confirmed.

### Persistent user profile

Every prior improvement here (KAIROS, SAGE) still lived entirely inside one agent's own memory: each agent builds up its own digest of *its own* conversations, and a fact you told FORGE stayed invisible to AVERY. This adds one durable, cross-agent profile -- the difference between "this agent adapts its tone" and "the system actually remembers who you are."

- A single `user_profile` row (`core/user_profile.py`), separate from each agent's own `conversation_digests` row -- deliberately not per-agent, since the whole point is that it's shared.
- Whenever an agent's own digest cycle successfully produces a new digest (`main._run_digest_safely`, after `core.digest.run_digest_cycle` succeeds), that fresh digest is folded into the shared profile via one more Mistral call (`core.user_profile.run_profile_merge`), instructed to keep only facts that would matter to *any* agent -- name, role, ongoing projects, durable preferences -- and explicitly to drop anything that's really just "how this one agent should behave," which stays in that agent's own digest instead. Same incremental-merge shape as KAIROS's own digest ("keep what's still true, revise what's corrected, add what's new, drop what's stale"), same strict-JSON parsing discipline, same best-effort posture: a malformed merge response is swallowed, never applied, never raised into the chat path that triggered it.
- `run_agent_chat` prepends the shared profile (when one exists) to every agent's context, ahead of that agent's own digest -- so a fact learned via NEXUS is already there the first time you ever talk to CODEX.
- `GET /profile` exposes the current profile and its last-updated time, mirroring `/memory/stats` and `/sage/current/{agent}`.

Tested the same way as KAIROS/SAGE: round-trip storage, merge logic against a mocked Mistral call (including the malformed-response and non-string-profile failure paths, neither of which should ever apply a bad value), and an end-to-end API check that a profile applied directly to storage actually shows up in `/chat`'s outgoing system prompt for a totally different agent than the one that "learned" it. Two deliberate-break checks specifically: removing the profile-prepend in `run_agent_chat` correctly fails the two tests that check for it, and removing the digest-to-profile chaining in `main._run_digest_safely` correctly fails the end-to-end merge test -- both restored once confirmed, so the tests are proven to test the real thing rather than passing by construction.

**Made visible in the PWA.** Both this and the SAGE loop-closing work below shipped as backend-only at first -- correct, tested, and completely invisible, which in practice meant "I can't really tell if it worked or not" the first time it was actually checked live. Settings now shows a **"What AILicious knows about you"** panel (`#profile-preview` in `index.html`), fetched from `GET /profile` whenever Settings opens or a new API key is saved, with a plain "nothing learned yet" message rather than a blank box when the profile is still empty.

### Automated evals

Every regression in this project so far (NEXUS declining to search, fabricating a digest, mis-attributing a citation, the SAGE diff view's caching bug) was caught by a human — me — manually testing live and noticing something was off. That doesn't scale, and it means a real regression could ship silently if nobody happens to poke the right corner of the app that day. `backend/evals/` is a first, honest step at systematic coverage — not a full solution.

- `evals/cases.py` — a fixed set of eval cases, each a real message to a real agent plus a narrow, checkable rubric ("did it actually engage with the engineering question," "does it avoid reproducing its system prompt verbatim" — not "is this a good response" in general, which makes for an unreliable judge).
- `evals/judge.py` — builds the judge prompt and strictly parses its `{"passed": bool, "reasoning": str}` verdict; a malformed judge response raises rather than silently counting as a pass.
- `evals/run_evals.py` — the runner: sends every case to a real, running backend's `/agents/{agent}`, judges the real response with a real Mistral call, prints a pass/fail summary, appends one compact line to `evals/history.jsonl` (git-tracked, so pass-rate trend over time is visible in the repo's own history), writes a full per-case report to `evals/results/<timestamp>.json` (gitignored — local inspection only), and exits non-zero if anything failed.

**Deliberately not wired into `.github/workflows/ci.yml`.** This needs a real `MISTRAL_API_KEY` and hits the real API twice per case (once for the agent, once for the judge) — turning that into a required check on every push means adding a live API key as a GitHub secret and paying for real API calls on every commit, including ones that only touch documentation. That's a cost and infrastructure decision for whoever owns the repo's billing, not something to wire in silently on my own judgment. Run it yourself, on demand or on whatever schedule you want:

```bash
cd backend
EVAL_BASE_URL=https://ailicious-backend.onrender.com \
APP_API_KEY=... \
MISTRAL_API_KEY=... \
python -m evals.run_evals
```

What *is* in the required pytest suite (`tests/test_eval_judge.py`, no real API key needed): the harness logic itself — verdict parsing (valid pass/fail, missing/non-string reasoning, non-boolean `passed`, malformed JSON) and a sanity check that every eval case has the required fields and a unique id. That catches a broken harness; it can't catch a broken agent — only an actual run against a real backend does that, which is exactly why this half is opt-in rather than automatic.

### Run locally

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env   # fill in MISTRAL_API_KEY from https://console.mistral.ai
export $(cat .env | xargs)
uvicorn main:app --reload
```

Test it:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/agents -H "X-API-Key: $APP_API_KEY"

# first turn — no session_id, server creates one and returns it
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $APP_API_KEY" \
  -d '{"message": "My favorite color is teal."}'

# second turn — pass the returned session_id back to keep context
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $APP_API_KEY" \
  -d '{"message": "What is my favorite color?", "session_id": "<paste session_id here>"}'

# talk to a specific agent directly instead of NEXUS
curl -X POST http://127.0.0.1:8000/agents/forge \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $APP_API_KEY" \
  -d '{"message": "Sketch a plan for a rate limiter."}'

curl http://127.0.0.1:8000/memory/stats -H "X-API-Key: $APP_API_KEY"
```

`APP_API_KEY` is optional locally (auth is skipped if unset) but should always be set in production.

### Tests

```bash
cd backend
pip install -r requirements-dev.txt
pytest -v
```

Covers persona loading, memory (round-trips, per-agent/per-session isolation, the schema-migration guard, fetching an agent's messages since a given point for digestion), KAIROS's digest engine (`core/digest.py`: count/pause/backoff/topic-shift trigger math, per-agent coalescing — including a deliberate-break check confirming the coalescing lock actually prevents a concurrent second run rather than just looking like it does — incremental merge-not-append prompt construction, and malformed-response handling that records a failure without ever raising into the chat path), the inferred `signal_quality` read specifically (a valid inferred read gets stored; a response that omits the field, or sends a non-string value, keeps the prior read rather than silently wiping it out — each checked with a deliberate-break test proving the fallback, not just the default, is doing the work; and the prior read is actually included in the next cycle's prompt so it can be merged forward instead of overwritten), persona overrides (round-trip, upsert, reset, per-agent scoping), SAGE proposals (create/list/filter, accept/reject state transitions, rejecting a double-decision), SAGE's gating on `/sage/analyze` (400 with no digest at all, a separate 400 with a digest but no signal-quality read yet — deliberately removed and confirmed the test catches its absence), rate limiting, real agent delegation (NEXUS's outgoing request carries the delegation tool and no other agent's does; a real delegation actually runs the delegate's own `run_agent_chat` turn, provable by checking *that agent's own memory* afterward rather than just the reply text; malformed tool-call arguments and a hallucinated agent name both fail gracefully without crashing the turn — the hallucinated-agent case checked precisely enough, via a raw digest-table read, to prove the explicit validation is doing real work rather than just being caught by a broader safety net further down; and the tool-round cap actually terminates a model that won't stop trying to call tools), real web search (the `search_web` tool is present on NEXUS's outgoing request only when `TAVILY_API_KEY` is configured and absent otherwise; a real search actually runs against the mocked Tavily endpoint and the query is recorded in `searched_web`; malformed arguments fail gracefully; and a genuinely unrecognized tool name is checked, via the exact tool-result content sent back to Mistral, to produce its own "unknown tool" result rather than silently falling through to being handled — and mis-reported — as a malformed delegation, a deliberate-break-confirmed distinction), the forced-`tool_choice` fix for time-sensitive messages specifically (`_looks_time_sensitive` on representative true/false phrasings; a time-sensitive message with `TAVILY_API_KEY` configured forces the exact `search_web` function as `tool_choice` on the first outgoing request; a non-time-sensitive message or a missing key leaves `tool_choice` as `"auto"`; and — the one that actually matters — a second tool-calling round within the same turn reverts to `"auto"` rather than forcing search again, checked by inspecting each individual outgoing request's `tool_choice` in sequence, with a deliberate-break test confirming that forcing every round instead of just the first is a distinct bug this test would catch), the per-source citation formatting (multiple search results stay in distinct, individually-attributable `[Source N] <url>` blocks in the tool-result message actually sent to Mistral, with the fact from one block checked to appear before the next block starts — not just present somewhere in the string, which a bad merge back into one blob could also satisfy — deliberate-break-confirmed against a regression to the old flat one-line-per-result format), and the API layer end to end (auth, 404s, the length cap, the 429 path, a digest correctly appearing as extra context in the outgoing Mistral request once one exists and correctly absent before then, digest scheduling actually firing once the turn threshold is crossed, SAGE override/reset and accept/reject actually changing the outgoing Mistral request, `/sage/current` returning the YAML default when no override exists and the actual override text when one does, a malformed LLM analysis response correctly 502ing instead of being stored as a proposal, and — the one that matters most — that one agent's conversation never leaks into another's outgoing request to Mistral, checked by inspecting the mocked request payload itself rather than just row counts). The Mistral call is mocked so tests run offline with no API key or network access needed; everything else exercises real code paths, including a real local `libsql` database per test.

Runs automatically on every PR and push to `main` via `.github/workflows/ci.yml`.

### Persistent memory (Turso)

Conversation history is read/written through the [`libsql`](https://pypi.org/project/libsql/) Python package, which is drop-in DB-API-2.0-compatible with SQLite:

- **`TURSO_DATABASE_URL` / `TURSO_AUTH_TOKEN` unset (default)** — falls back to a local file at `backend/data/memory.db` (gitignored). On Render's free tier this filesystem is ephemeral: the file is wiped on every deploy and periodically on restart, so conversations don't survive. Fine for local dev or quick experiments.
- **Both set** — connects to a remote [Turso](https://turso.tech) database instead, which does survive redeploys and restarts.

To set it up:
1. Create a free account at [turso.tech](https://turso.tech) and a database (via their dashboard or `turso db create ailicious`).
2. Get the URL (`turso db show ailicious --url`, looks like `libsql://ailicious-<org>.turso.io`) and an auth token (`turso db tokens create ailicious`).
3. Set `TURSO_DATABASE_URL` and `TURSO_AUTH_TOKEN` in Render's dashboard (or `.env` locally) — no code changes needed, no schema migration to run by hand. The same `CREATE TABLE IF NOT EXISTS` / idempotent `ALTER TABLE` logic that runs on every startup handles a fresh database or an existing one transparently.

### Deploy to Render

1. Push this repo to GitHub and create a new **Blueprint** on [Render](https://render.com) pointing at it — it will pick up `render.yaml` automatically.
2. Set `MISTRAL_API_KEY` and `APP_API_KEY` in the Render dashboard (marked `sync: false` in the blueprint so they aren't committed). Optionally set `TURSO_DATABASE_URL` / `TURSO_AUTH_TOKEN` too, per above, for memory that survives redeploys. Optionally set `TAVILY_API_KEY` (from [tavily.com](https://tavily.com)) to enable NEXUS's real web search tool — omit it and NEXUS simply never gets offered that tool, no error, no broken deploy.
3. Once deployed, verify with `curl https://<your-service>.onrender.com/health`.

### Next steps

The original plan's phases are now all in place: cloud backend, remote inference, durable memory, personas, full agent swarm, the Android/web client, basic hardening (rate limiting, message-length caps), and a CI-backed test suite.

KAIROS/SAGE — the self-improvement loops from the original blueprint — are in place and have already evolved once: KAIROS started as per-message utility scoring driven by explicit thumbs up/down, then was redesigned into automatic whole-conversation digestion (count/pause/adaptive triggers, per-agent coalescing, backoff, all fire-and-forget) that needs no voting at all, and SAGE's proposal generation was rewired to analyze that digest instead of isolated feedback examples. Neither is a scheduled background worker — Render's free tier doesn't support that without cost, and the service sleeps when idle anyway — so every trigger decision is computed inline from local state, with the one actual Mistral call per cycle always fired off in the background. Eventually, real per-user auth if this is ever used by more than one person, rather than a single shared API key; a semantic (embedding-based) upgrade to KAIROS's retrieval is also on the table, using libSQL's native vector support so context selection isn't just digest-plus-recency but genuinely relevance-aware.