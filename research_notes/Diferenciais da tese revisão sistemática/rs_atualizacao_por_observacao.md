# Systematic review: rule/observation-driven symbolic KB update when the abstract operator is unknown

Thesis differential under review: the agent holds Π = (Π⁺, Π⁻) (open-world known-true / known-false literals over a hidden complete STRIPS theory c), executes only low-level actions, cannot see which abstract operator fired, receives a partial truthful ψ = Φ(c,x) ⊆ c each step, and updates Π with rules ρ = (Γ_kb, Γ_obs, Γ_new) (if Γ_kb ⊆ Π and Γ_obs ⊆ ψ then apply Γ_new).

Search date: 2026-10-09 (one session). Reviewer: Claude research subagent.

---

## 1. Protocol (fixed before searching)

**Research question (PICO-like)**
- **P (population):** agents (planning, robotic, RL or LLM agents) that partially observe a symbolic (propositional or first-order fluent) state.
- **I (intervention):** a knowledge/belief update driven by observations and/or rules, where the agent does **not** know which abstract (symbolic) action/event occurred. Examples: hypothesizing unobserved/exogenous actions, update rules, filtering with an unknown action, or abstraction monitoring.
- **C (comparison):** standard progression/filtering where the executed action is known (Lin–Reiter progression, logical filtering with known actions, POMDP belief update with known a).
- **O (outcome):** sound (truthful, non-over-committing) tracking of the symbolic state. Formal guarantees or empirical tracking accuracy both count.

**Sources planned:** Consensus (Semantic Scholar/OpenAlex index), alphaXiv `discover_papers` + `get_paper_content`, general web search (WebSearch), and Semantic Scholar API / DBLP / publisher pages via fetch for snowballing.

**Time window:** 1990–2026.

**Inclusion criteria (all of these)**
- IC1: Symbolic (logical/propositional/relational) state or knowledge representation, or a neuro-symbolic system with explicit symbolic state.
- IC2: State/knowledge is updated from observations over time (not one-shot perception).
- IC3: At least one of the following: (a) the action/event causing change is unknown, unobserved, exogenous, or only partially observed; (b) update is rule-based or abductive; (c) there are two levels (low-level execution vs abstract operators).
- IC4: Peer-reviewed venue, or an arXiv preprint (flagged as such).

**Exclusion criteria**
- EC1: Purely continuous/metric state estimation (Kalman/particle filters over poses) with no symbolic layer.
- EC2: Belief revision of static knowledge with no dynamics/actions.
- EC3: Action-model learning where the goal is learning the model rather than tracking the state, unless it also tracks state (e.g., SLAF).
- EC4: Non-English, or no abstract accessible.
- EC5: Pure LLM "memory" or retrieval with no explicit state representation.

**Overlap verdict scale**
- *identical*: same problem (abstract operator unknown, partial truthful observations, rule-based KB update with open-world Π).
- *partial*: shares at least two core elements (e.g., unknown actions + symbolic state tracking from observations) but differs in mechanism or assumptions.
- *superficial*: same broad area (symbolic state tracking under partial observability) but known actions, or a different goal.

---

## 2. Search log

Tools actually used: Consensus (failed: monthly quota exhausted / rate limit), alphaXiv `discover_papers` (worked), WebSearch (worked until a session limit hit near the end), alphaXiv `get_paper_content` (worked, arXiv only). Semantic Scholar API, DBLP, OpenAlex, ojs.aaai.org, aima.eecs.berkeley.edu, papers.phmsociety.org and arxiv.org direct were **all unreachable**: proxy 403 / DNS failure; see Section 7.

"Results" counts the result links each tool returned, some of which are non-paper pages (index pages, slides, patents). "Screened" means title plus snippet/abstract was read. "Retained" means the record passed IC1–IC4 at title/abstract level.

| # | Query string (exactly as typed) | Tool | Results | Screened | Retained |
|---|---|---|---|---|---|
| Q1 | `belief update from observations unknown action` | Consensus | 0 (error: monthly quota exhausted) | 0 | 0 |
| Q2 | `explanatory diagnosis planning partial observations` | Consensus | 0 (rate-limit error) | 0 | 0 |
| Q3 | `logical filtering partially observable` | Consensus | 0 (rate-limit error) | 0 | 0 |
| Q4 | question: `belief update from observations when the action that occurred is unknown; logical filtering in partially observable domains; explanatory diagnosis as planning from partial observations`; keywords [belief update, observations, unknown action, logical filtering, explanatory diagnosis]; prioritize=historical | alphaXiv | 10 | 10 | 3 (2407.06622, 2609.10036, 2603.03704) |
| Q5 | question: `LLM or RL agents that track a symbolic world state (PDDL predicates) from partial observations for planning, neuro-symbolic state estimation, 2023-2026`; keywords [world state tracking, LLM, symbolic state, partial observability, planning]; after 2023-01-01 | alphaXiv | 12 | 12 | 6 (2607.28942, 2609.16884, 2608.04933, 2609.25766, 2602.03974, 2606.15654) |
| Q6 | `progression exogenous actions sensing situation calculus` | WebSearch | 10 | 10 | 4 |
| Q7 | `ramification static causal laws knowledge update` | WebSearch | 9 | 9 | 3 |
| Q8 | `state estimation symbolic planning robot knowledge base update` | WebSearch | 9 | 9 | 3 |
| Q9 | `open world planning knowledge update sensing` | WebSearch | 9 | 9 | 3 |
| Q10 | `symbolic state tracking reinforcement learning partial observability` | WebSearch | 9 | 9 | 1 (background only) |
| Q11 | `belief update from observations unknown action` | WebSearch | 10 | 10 | 5 |
| Q12 | `explanatory diagnosis planning partial observations` | WebSearch | 9 | 9 | 3 |
| Q13 | `logical filtering partially observable Amir Russell` | WebSearch | 9 | 9 | 4 |
| Q14 | `Sohrabi Baier McIlraith "Diagnosis as Planning Revisited" KR 2010 pdf` (snowball) | WebSearch | 9 | 9 | 3 |
| Q15 | question: `LLM agents maintaining an explicit symbolic world state / state representation updated from observations during long-horizon task planning (e.g., LLM-State, entity state tracking, belief tracking for embodied agents)`; after 2023-01-01 | alphaXiv | 12 | 12 | 5 |
| Q16 | `Hunter Delgrande "Belief Change with Uncertain Action Histories" JAIR 2015 abstract` (snowball) | WebSearch | 9 | 9 | 2 |
| Q17 | `Yu Wen Liu IJCAI 2013 diagnosis situation calculus observations` (snowball) | WebSearch | 9 | 9 | 2 |
| Q18 | `Delgrande Levesque KR 2012 "Belief Revision with Sensing and Fallible Actions"` (snowball) | WebSearch | 9 | 9 | 3 |
| Q19 | `Yu Wen Liu "Multi-agent epistemic explanatory diagnosis via reasoning about actions" IJCAI` | WebSearch | 10 | 10 | 1 |
| Q20 | `Bonet Geffner belief tracking planning with sensing width complexity JAIR` (snowball from Amir & Russell) | WebSearch | 10 | 10 | 3 |
| Q21 | `Shapiro Pagnucco iterated belief change exogenous actions situation calculus IJCAI 2004` (snowball) | WebSearch | 9 | 9 | 3 |
| Q22 | `Banihashemi De Giacomo Lespérance abstraction situation calculus action theories refinement mapping` | WebSearch | 10 | 10 | 3 |
| Q23 | `"LLM-State" open world state representation long-horizon task planning large language models` | WebSearch | 0 (error: session limit) | 0 | 0 |
| Q24 | `Embodied Agent Interface Li NeurIPS 2024 goal interpretation subgoal decomposition transition modeling LTL` | WebSearch | 0 (error: session limit) | 0 | 0 |
| Q25 | question: `LLM agents that keep and update an explicit symbolic or structured world state from observations during task planning, 2023-2025 (LLM-State, Embodied Agent Interface, entity tracking)`; 2023-01-01..2025-12-31 | alphaXiv | 13 | 13 | 5 |
| Q26 | question: `explaining observations by hypothesizing unobserved exogenous actions or events and updating the agent's knowledge base (abductive state estimation in reasoning about action)`; prioritize=historical | alphaXiv | 0 ("No papers found") | 0 | 0 |

Totals: 26 query strings (22 distinct, 4 tool-failures with no results). 196 records returned, 196 screened, 66 retained before deduplication.

Notes on query design: Q1–Q3 and Q11–Q13 are the same strings on different tools (Consensus failed). All eight suggested strings were run on at least one working tool: Q11, Q12, Q13, Q8, Q6, Q7, Q10, Q9.

---

## 3. Snowballing

Forward-citation databases (Semantic Scholar API, Google Scholar, DBLP) were **unreachable**: proxy 403 / DNS failure. Forward snowballing was therefore approximated by web search for papers that the search results describe as extending or citing the seed. Backward snowballing was done on reference lists that could actually be read (arXiv full text via alphaXiv).

| Seed | Backward (refs checked) | Forward (citing/extending works found) | Retained |
|---|---|---|---|
| **Amir & Russell 2003, "Logical Filtering", IJCAI** (PDF at aima.eecs.berkeley.edu unreachable; abstract via search) | Reference list not readable | Shahaf & Amir IJCAI 2007 "Logical Circuit Filtering"; Kumar & Russell ICAPS 2006 "On Some Tractable Cases of Logical Filtering"; Mombourquette, Muise & McIlraith AAAI 2017 "Logical Filtering and Smoothing"; Bonet & Geffner AAAI 2012 / IJCAI 2013 / JAIR 2014 belief tracking (found via Q20) | 5 (all assume known actions; Mombourquette et al. closest) |
| **Sohrabi, Baier & McIlraith 2010, "Diagnosis as Planning Revisited", KR** (PDF not reachable) | Not readable; abridged report at PHM Society also unreachable | Baier, Mombourquette & McIlraith KR 2014 "Diagnostic problem solving via planning with ontic and epistemic goals"; Haslum & Grastien ICAPS 2011 diagnosis of DES as planning (ANU record); "Diagnostic Reasoning for Robotics Using Action Languages" (ouci.dntb.gov.ua) | 3 |
| **Yu, Wen & Liu 2013, IJCAI** (identified as "Multi-Agent Epistemic Explanatory Diagnosis via Reasoning About Actions", IJCAI-13 pp. 1183–1190) | Abstract only; it builds on McIlraith's explanatory diagnosis (McIlraith 1995/1998 Toronto pages surfaced in Q17) | Forward not reachable | 2 (Yu 2013; McIlraith explanatory diagnosis) |
| **Delgrande & Levesque 2012, "Belief Revision with Sensing and Fallible Actions", KR** (PDF link at www2.cs.sfu.ca surfaced; fetch not attempted because the fetch tool's DNS was failing) | Search summary: builds on Shapiro et al. plausibility-ordered situations and Reiter BATs | Classen & Delgrande KR 2021; Classen & Delgrande KR 2022 (projection of belief with nondeterministic actions and fallible sensing); Delgrande & Levesque 2019 follow-up (mentioned in snippet, not located) | 3 |
| **Hunter & Delgrande 2011, "Iterated Belief Change Due to Actions and Observations", JAIR** (arXiv 1401.3867; full text retrieved; intro + reference list read) | Refs checked: Boutilier IJCAI 1995 "Generalized update"; Shapiro & Pagnucco ECAI 2004 "Iterated belief change and exogenous actions in the situation calculus"; Shapiro, Pagnucco, Lespérance & Levesque KR 2000; Lang NMR 2006 "About time, revision, and update"; Katsuno & Mendelzon KR 1991; Peppas et al. ECAI 1996; Son & Baral AIJ 2001; Lobo et al. TPLP 2001; Hunter & Delgrande IJCAI 2005 / AAAI 2006 | Hunter & Delgrande JAIR 2015 "Belief Change with Uncertain Action Histories"; "Using ranking functions to determine plausible action histories" (BCIT repository) | 6 (Boutilier 1995, Shapiro & Pagnucco 2004, Lang 2006, H&D 2006, H&D 2015, ranking-functions paper) |
| **Dupin de Saint-Cyr & Lang, "Reasoning about unpredicted change and explicit time"** (arXiv 2407.06622, full text read) | Refs checked: Lifschitz & Rabinov IJCAI 1989 "Things that change by themselves"; Sandewall 1994 *Features and Fluents*; Boutilier IJCAI 1995; Cordier & Thiébaux DX 1994 event-based diagnosis; Console et al. ECAI 1992; Friedrich & Lackinger IJCAI 1991; Dean & Kanazawa 1989 | — | 3 (Lifschitz & Rabinov 1989, Cordier & Thiébaux 1994, Boutilier 1995), titles only |
| **Banihashemi, De Giacomo & Lespérance, "Abstracting Situation Calculus Action Theories"** (arXiv 2410.14712 / AIJ 348, 2025; intro + Sec. 7 read) | Mentions Gabaldon on exogenous actions and online executions (title not extracted) | IJCAI 2023 "Abstraction of Nondeterministic Situation Calculus Action Theories" | 2 |

---

## 4. Selection flow (PRISMA-style)

- **Identified:** 196 records from the database/tool searches, plus 14 from snowballing (reference-list titles and forward/extension hits not already present) = **210**.
- **After deduplication and removal of non-paper pages** (index pages, slides, patents, tutorial pages, the same paper under several URLs): **≈ 95 unique records** (approximate; deduplication was done by hand).
- **Screened (title/abstract):** 95.
- **Excluded at screening:** ≈ 46. Reasons: EC1 continuous-only state estimation (e.g., MAPF under map uncertainty, CAR-DESPOT, path planning); EC2 static revision; EC5 LLM memory without explicit state; off-topic (video benchmarks, UAV agents, patents).
- **Eligible:** 49.
- **Full text (or substantial portion) read:** **6**. These were Hunter & Delgrande 2011 (intro + references); Dupin de Saint-Cyr & Lang (complete); Banihashemi et al. 2410.14712 (intro + Sec. 7 "Monitoring and Explanation"); Belief-State Engine 2609.10036 (alphaXiv structured report); BB-WM 2609.00455 (alphaXiv structured report); CoCo-TAMP 2603.03704 (abstract + intro preview).
- **Included in synthesis table:** **47** (6 at full text, 41 at abstract/snippet level). This includes the previously known works that were re-located by this search.

---

## 5. Included studies

R = what was read: FT = full text; PT = part of the full text; AR = alphaXiv AI-generated structured report of the full text; AB = abstract or search-snippet summary only; REF = title known only from a reference list.

### 5a. Reasoning about action / belief change / diagnosis (1990–2022)

| # | Citation | R | Verdict | One-line justification |
|---|---|---|---|---|
| 1 | Banihashemi, De Giacomo, Lespérance. "Abstraction in Situation Calculus Action Theories", AAAI 2017. https://ojs.aaai.org/index.php/AAAI/article/view/10693 ; extended: "Abstracting Situation Calculus Action Theories", AIJ 348 (2025) / arXiv 2410.14712 https://www.alphaxiv.org/abs/2410.14712 | PT (Sec. 7) | **partial (closest structural match)** | Two-level setting: abstract actions are refined into low-level programs, and Sec. 7 "Monitoring and Explanation" infers which high-level action sequence explains a low-level situation. But it assumes complete low-level information and states that the incomplete-information case "may be different for different models… We leave this problem for future work". There are no truthful partial observations ψ and no open-world Π. |
| 2 | Banihashemi, De Giacomo, Lespérance. "Abstraction of Nondeterministic Situation Calculus Action Theories", IJCAI 2023. https://ijcai.org/proceedings/2023/347 | AB | partial | Abstract actions split into agent actions and environment reactions, which matches the thesis's "abstract operators fire as side effects". No partial-observation KB update. |
| 3 | Sohrabi, Baier, McIlraith. "Diagnosis as Planning Revisited", KR 2010, pp. 26–36. https://bibbase.org/network/publication/sohrabi-baier-mcilraith-diagnosisasplanningrevisited-2010 ; https://ing.uc.cl/publicaciones/diagnosis-as-planning-revisited | AB | partial | Hypothesizes unobserved actions/faults that explain observations, extended to incomplete information. It is offline explanation generation by planning, not an incremental rule-based KB update. |
| 4 | Baier, Mombourquette, McIlraith. "Diagnostic Problem Solving via Planning with Ontic and Epistemic Goals", KR 2014. https://bibbase.org/network/publication/baier-mombourquette-mcilraith-diagnosticproblemsolvingviaplanningwithonticandepistemicgoals-2014 | AB | superficial | Goal is planning to discriminate diagnoses, not state tracking. |
| 5 | Haslum & Grastien. Diagnosis of discrete event systems as planning (partially ordered observations, PDDL), ICAPS 2011. https://openresearch-repository.anu.edu.au/items/fec7b7d4-7f2b-42fe-a0ce-891e7b2ccf86 | AB | partial | Infers unobserved event sequences that are consistent with observations. Closed-world planning encoding; no open-world KB. |
| 6 | Yu, Wen, Liu. "Multi-Agent Epistemic Explanatory Diagnosis via Reasoning About Actions", IJCAI 2013, pp. 1183–1190. https://www.ijcai.org/Abstract/13/178 ; https://mlanthology.org/ijcai/2013/yu2013ijcai-multi | AB | partial | Explanatory diagnosis (conjecturing actions to explain observations) in DEL, with partially observable actions modelled as Kripke action models. Undecidable in general, so restricted to decidable fragments. Uses no rules of the Γ_kb/Γ_obs form. |
| 7 | McIlraith. Explanatory diagnosis in the situation calculus (Toronto technical pages). https://www.cs.toronto.edu/~sheila/publications/ss95.pdf ; https://www.cs.toronto.edu/~sheila/publications/aaai97.pdf | AB (snippet) | partial | Origin of conjecturing actions to explain observations. Exact titles/venues of these two PDFs were not verified in tool output. |
| 8 | Hunter & Delgrande. "Iterated Belief Change Due to Actions and Observations", JAIR 40 (2011) 269–304. https://jair.org/index.php/jair/article/view/10690 ; arXiv 1401.3867 | PT | superficial (contrast) | Explicitly assumes "effects of actions are completely specified and infallible… perfect knowledge of the actions executed". This is the C (comparison) arm. It also notes that hidden exogenous actions are a second source of error, but sets that aside. |
| 9 | Hunter & Delgrande. "Belief Change with Uncertain Action Histories", JAIR 53 (2015) 779–824, doi:10.1613/JAIR.4558. https://mlanthology.org/jair/2015/hunter2015jair-belief ; https://jair.org/index.php/jair/article/download/10956/26097/20441 | AB | **partial (high)** | The agent is uncertain about which actions occurred. Ranking functions over actions and states handle exogenous and failed actions alongside observations. Plausibility-based (ranking) and semantic (sets of worlds), not literal-level rules over Π⁺/Π⁻. |
| 10 | Hunter & Delgrande. "Belief Change in the Context of Fallible Actions and Observations", AAAI 2006. https://mlanthology.org/aaai/2006/hunter2006aaai-belief | AB | partial | Earlier version of #9 (fallible actions). |
| 11 | Hunter & Delgrande. "Using ranking functions to determine plausible action histories". https://repository.lib.bcit.ca/node/1652 | AB (title) | partial | Same line as #9; venue not verified. |
| 12 | Nance et al. "Reasoning About Partially Observed Actions", AAAI 2006. https://mlanthology.org/aaai/2006/nance2006aaai-reasoning | AB (snippet) | **partial (high)** | The action type is observed but its arguments are unknown. A logical belief state is updated using new constants for unknown objects, and the authors note that the naive approach is incorrect. This is the closest in the logical-filtering line to "agent does not know which operator instance fired". Co-authors not verified (slug suggests first author Nance) [co-authors from memory - unverified: Vogel, Amir]. |
| 13 | Amir & Russell. "Logical Filtering", IJCAI 2003. https://aima.eecs.berkeley.edu/~russell/papers/ijcai03-filter.pdf | AB | partial | Logical belief-state update from actions and observations in partially observable (nondeterministic) STRIPS domains, with compact representations. Actions are known; observations are formulas. |
| 14 | Shahaf & Amir. "Logical Circuit Filtering", IJCAI 2007. https://www.ijcai.org/Abstract/07/420 | AB | superficial | Tractable exact filtering for deterministic domains; actions known. |
| 15 | Kumar & Russell. "On Some Tractable Cases of Logical Filtering", ICAPS 2006. https://aima.eecs.berkeley.edu/~russell/papers/icaps06-filter.pdf | AB | superficial | Tractable classes of transition constraints; known actions. |
| 16 | Mombourquette, Muise, McIlraith. "Logical Filtering and Smoothing: State Estimation in Partially Observable Domains", AAAI 2017. https://ojs.aaai.org/index.php/AAAI/article/view/11031 ; https://mlanthology.org/aaai/2017/mombourquette2017aaai-logical | AB | partial | Approximate (weaker) filtering plus smoothing that refines past beliefs from new observations, comparable to sound-but-incomplete Π. Actions known. |
| 17 | Bonet & Geffner. "Belief Tracking for Planning with Sensing: Width, Complexity and Approximations", JAIR 50 (2014) 923–970. https://mlanthology.org/jair/2014/bonet2014jair-belief ; AAAI 2012: https://ojs.aaai.org/index.php/AAAI/article/view/8365 ; IJCAI 2013 / arXiv 1909.13778 | AB | superficial–partial | Factored/beam belief tracking (sound approximations), comparable to the thesis's literal-level Π. Actions known; it is a planning-with-sensing setting. |
| 18 | Delgrande & Levesque. "Belief Revision with Sensing and Fallible Actions", KR 2012. https://www2.cs.sfu.ca/~jim/publications/KR12.pdf | AB | partial | The agent may execute a different action than intended (e.g., the wrong button), so the actual action is uncertain. Plausibility-ordered situations in the situation calculus. Not rule-based; no two-level abstraction. |
| 19 | Classen & Delgrande, KR 2021 and KR 2022 (projection of belief with nondeterministic actions and fallible sensing). https://proceedings.kr.org/2021/19/kr2021-0019-classen-et-al.pdf ; https://proceedings.kr.org/2022/40/kr2022-0040-classen-et-al.pdf | AB | partial | Generalize #18; progression with nondeterministic actions and fallible sensors. |
| 20 | Shapiro, Pagnucco, Lespérance, Levesque. "Iterated Belief Change in the Situation Calculus", KR 2000, pp. 527–538; journal version AIJ 175(1) 2011. Ref in #8; snippet https://datalearner.com/academic/journal-papers/0004-3702/volumes-and-issues/359/paper-detail/68780 | REF/AB | superficial | Sensing as revision with plausibility over initial situations; actions known. |
| 21 | Shapiro & Pagnucco. "Iterated Belief Change and Exogenous Actions in the Situation Calculus", ECAI 2004, pp. 878–882 (from #8 reference list). Online copy not located. | REF | **partial (high, unread)** | Explains inconsistent sensing by postulating exogenous actions, which is close to "observation implies an unknown operator happened". Only the title and the one-line description in #8 were seen. |
| 22 | Klassen, McIlraith, Levesque. KR 2018 (plausibility levels; belief change about predicted and unpredicted exogenous actions). https://www.cs.toronto.edu/~toryn/papers/KR-2018.pdf | AB (snippet) | partial | Handles unpredicted exogenous actions in belief change. Exact title not verified in tool output. |
| 23 | Ma, Liu, Miller. "Belief change with noisy sensing in the situation calculus", UAI 2011. https://proceedings.mlr.press/r9/ma11a.html | AB | superficial | Noisy sensing; observations are not truthful (the thesis has truthful ψ). |
| 24 | Boutilier. "Generalized Update: Belief Change in Dynamic Settings", IJCAI 1995, pp. 1550–1556 (from refs of #8 and #25). | REF | **partial (high, unread)** | Per #25: explains observations by events ranked by plausibility, with a two-time-point state change caused by unknown events. This is the classic "update by unknown event" framework. |
| 25 | Dupin de Saint-Cyr & Lang. "Reasoning about unpredicted change and explicit time", arXiv 2407.06622 (2024 upload of an older IRIT paper; original venue not verified). https://www.alphaxiv.org/abs/2407.06622 | FT | **partial (high)** | Passive agent with truthful observations (Sandewall's K). Changes are explained by minimal "surprises" (unexplained fluent flips), linked to model-based diagnosis. Fluents are independent and there are explicitly no ramifications/state constraints; no operators or rules. |
| 26 | Lifschitz & Rabinov. "Things that change by themselves", IJCAI 1989 (from refs of #25; just outside the 1990 window). | REF | superficial | Minimizing unexplained change; background. |
| 27 | Cordier & Thiébaux. "Event-based diagnosis for evolutive systems", DX 1994 (from refs of #25). | REF | partial (unread) | Infers event sequences from observations in evolving systems. |
| 28 | Lang. "About time, revision, and update", NMR 2006 (from refs of #8). | REF | partial (unread) | Treats update as progression and relates it to revision under action/observation sequences. |
| 29 | Roos & Witteveen. "Models and Methods for Plan Diagnosis". https://dke.maastrichtuniversity.nl/nico.roos/wp-content/uploads/2015/04/RW06MBS.pdf ; "Diagnosis of Plans and Agents" (CEEMAS 2005) https://dke.maastrichtuniversity.nl/nico.roos/wp-content/uploads/2015/04/RW05CEEMAS.pdf ; Witteveen et al. https://ir.cwi.nl/pub/14873/ | AB | superficial | Diagnoses abnormal plan steps from partial states; actions are known (the plan), so the unknown is the failure. |
| 30 | Çoruhlu & Gökay (Sabancı). "Explainable robotic plan execution monitoring under partial observability". https://research.sabanciuniv.edu/id/eprint/44349/ | AB | superficial | Hypotheses revised as partial observations arrive; robot fault hypotheses. |
| 31 | Vassos & Levesque. "Progression of Situation Calculus Action Theories with Incomplete Information", IJCAI 2007. https://www.ijcai.org/Abstract/07/327 | AB | superficial (C arm) | Progression with sensing and incomplete knowledge; known actions. |
| 32 | De Giacomo & Levesque. Projection using regression and sensors, IJCAI 1999. https://www.diag.uniroma1.it/~degiacomo/papers/1999/DeLe99ijcai.pdf | AB | superficial | Known actions. |
| 33 | Schwering, Lakemeyer, Pagnucco. Belief revision and progression with conditional beliefs, IJCAI 2015. https://www.ijcai.org/Abstract/15/453 | AB | superficial | Revisable sensing; known actions. |
| 34 | Eppe & Bhatt. "Tractable Epistemic Reasoning with Functional Fluents, Static Causal Laws and Postdiction", arXiv 1403.0034. https://arxiv.org/pdf/1403.0034 ; also mentions Tu et al. 2007 A_k^c (0-approximation with static causal laws) | AB | partial | An approximate (0-approximation-style) knowledge state with known-true/known-false fluents plus static causal laws and postdiction. This is technically the closest representation to Π⁺/Π⁻ plus rules. Actions known. |
| 35 | McCain & Turner. "A Causal Theory of Ramifications and Qualifications", IJCAI 1995. https://mlanthology.org/ijcai/1995/mccain1995ijcai-causal | AB | superficial | Ramification background for rule-triggered indirect effects. |
| 36 | Babaian & Schmolze. PSIPLAN / PSIPLAN-S (AIPS 2000; LMCS 2006; journal 2009). https://cdn.aaai.org/AIPS/2000/AIPS00-031.pdf ; https://lmcs.episciences.org/2247/pdf ; https://philpapers.org/rec/BABPRA | AB | superficial | Open-world knowledge-state update with sensing, correct and complete. Actions known. |
| 37 | Petrick & Bacchus. PKS knowledge-level planning. https://www.cs.nmsu.edu/~tson/classes/spring04-579/petricbacchus.pdf | AB | superficial | Knowledge state as databases (Kf etc.) updated by actions, similar in spirit to Π. Actions known. |

### 5b. Robotics / neuro-symbolic / LLM / RL (2023–2026)

| # | Citation | R | Verdict | One-line justification |
|---|---|---|---|---|
| 38 | Kumar, Kumar, Ahuja, Jha. "Towards a Belief-Based World Model for LLM Agents", arXiv 2609.00455 (2026). https://www.alphaxiv.org/abs/2609.00455 | AR | partial | Hand-specified symbolic belief state (location, inventory, receptacle contents) plus categorical beliefs, updated from observations by explicit rules ("presence/absence renormalization"). The agent's own text actions are known and it is single-level, with no hidden abstract operators. |
| 39 | Chattopadhayay & Halder. "Belief-State Engine: Augmenting LLMs for Principled Planning Under Partial Observability", arXiv 2609.10036 (2026). https://www.alphaxiv.org/abs/2609.10036 | AR | superficial | External exact Bayes filter with known T, Z and known actions feeding an LLM policy. The relevant point is the separation of belief tracking from the LLM, plus a soundness theorem. |
| 40 | Kim, Arora, Martín-Martín, Stone, Abbatematteo, Sung. CoCo-TAMP: "LLM-Guided State Estimation for Partially Observable TAMP", arXiv 2603.03704 (2026). https://www.alphaxiv.org/abs/2603.03704 | AB+intro | superficial | LLM common-sense priors shape the belief over object locations; probabilistic, not rule-based KB update. |
| 41 | Chen et al. (NUS). "LLM-State: Open World State Representation for Long-horizon Task Planning with LLM", arXiv 2311.17406. https://www.alphaxiv.org/abs/2311.17406 | AB (snippet) | partial | Explicitly tracks key objects/attributes in an open-world state updated from observations and action outcomes. Updates are done by the LLM, with no soundness guarantee. First-author name not verified in tool output. |
| 42 | StateAct: "Enhancing LLM Base Agents via Self-prompting and State-tracking", arXiv 2410.02810 (Imperial). https://www.alphaxiv.org/abs/2410.02810 | AB (snippet) | superficial | Prompted textual state-tracking; not symbolic/sound. |
| 43 | PDDL-Mind: "LLMs are Capable on Belief Reasoning with Reliable State Tracking", arXiv 2604.17819 (USC, 2026). https://www.alphaxiv.org/abs/2604.17819 | AB (snippet) | partial | Uses PDDL-based state tracking to support belief (ToM) reasoning: the state is tracked symbolically from narrated events. Actions are given in text. |
| 44 | Mimir: "Neuro-Symbolic Memory System with Dynamic Grounding for Embodied Agents", arXiv 2608.04933 (2026). https://www.alphaxiv.org/abs/2608.04933 | AB (snippet) | superficial | Scene belief plus execution progress under partial observability. |
| 45 | NeSyFS: "Neuro-symbolic Fast-Slow Thinking Framework for LLM Agent under Partial Observability", arXiv 2607.28942 (Georgia Tech, 2026). https://www.alphaxiv.org/abs/2607.28942 | AB (snippet) | superficial | Neuro-symbolic agent under PO; details not read. |
| 46 | "Bridging Learned Visual Perception and Symbolic Belief-Space Planning", arXiv 2609.16884 (Technion, 2026) https://www.alphaxiv.org/abs/2609.16884 ; "Seeing is Believing: Belief-Space Planning with Foundation Models as Uncertainty Estimators", arXiv 2504.03245 https://arxiv.org/pdf/2504.03245 ; S3E symbolic state estimation with VLMs (ICRA 2025 workshop) https://dyalab.mines.edu/2025/icra-workshop/5.pdf | AB | superficial | Ground PDDL-like belief predicates from perception (an analogue of Φ). The symbolic update is predicate re-evaluation, not inference over unknown operators. |
| 47 | Other 2026 hits screened at title level and kept as background: PO-PDDL 2606.15654; Neurosymbolic action model learning under PO 2609.25766; Active Epistemic Control 2602.03974; "Beyond Memory: Explicit Belief States" 2610.01415; SKILL.state 2608.26263; World State Generator 2609.24744; DR. WELL 2511.04646; LLM-empowered state representation for RL 2407.13237. URLs: https://www.alphaxiv.org/abs/<id> | AB (title/snippet) | superficial | Explicit state/belief for LLM or RL agents; none reported (in snippets) inferring hidden abstract operators from passive observation. |

**Previously known works not re-located in this search** (WebSearch hit its session limit before Q23/Q24; none of the alphaXiv queries returned them): ROSPlan; Hanheide et al. AIJ 2017; Li et al. NeurIPS 2024 (presumably "Embodied Agent Interface"). They remain in the prior non-systematic list as [from memory - unverified in this protocol]. Ramírez & Geffner plan recognition as planning (2009/2010) and Amir & Chang SLAF (JAIR 2008) are also relevant comparators but were **not returned by any tool here** [from memory - unverified].

---

## 6. LLM-based and RL agents (2023–2026): specific findings

- No retrieved 2023–2026 LLM/RL paper infers *which hidden abstract operator fired* from passive partial observations. All located systems either know the agent's own (single-level) action, or re-ground predicates directly from perception. [#38–#47]
- The closest is BB-WM (#38). It is a hand-coded symbolic belief state with explicit, rule-like observation updates (presence/absence renormalization) exposed to the LLM through a query interface. Memory-only (deterministic) tracking underperformed belief tracking in its ALFWorld ablation, which per the report suggests explicit uncertainty matters. [https://www.alphaxiv.org/abs/2609.00455]
- The Belief-State Engine (#39) proves soundness of composing an exact external Bayes filter with an LLM policy. This is a precedent for the thesis's argument of keeping tracking outside the learned policy, but it requires known T, Z and known actions. [https://www.alphaxiv.org/abs/2609.10036]
- In LLM-State (#41) and StateAct (#42), the LLM itself performs the state updates, with no soundness guarantee. This contrasts with the thesis's sound rule application.
- Perception-grounding works (#46) correspond to the thesis's Φ(c,x), not to Π-update rules.
- RL search (Q10) returned only generic belief-representation work (e.g., Wang et al. ICML 2023, https://proceedings.mlr.press/v202/wang23p/wang23p.pdf). Nothing symbolic. [screened, excluded EC1/EC5]

---

## Synthesis: overlap with the thesis differential

**Verdict:** no included study is *identical*. The differential holds as the **combination** of these elements:
1. Two-level setting where abstract operators fire as side effects of known low-level actions, and the agent does not know which operator fired.
2. Truthful partial observations ψ ⊆ c.
3. Open-world literal-level Π⁺/Π⁻.
4. Update by condition–action rules ρ = (Γ_kb, Γ_obs, Γ_new) rather than model-theoretic filtering.

Each element appears separately in the literature:
- (1) The two-level part matches Banihashemi et al. (#1, #2). They explicitly leave monitoring under incomplete low-level information as future work, which is a direct, citable gap statement.
- "Unknown action" (part of 1) appears in Hunter & Delgrande 2015 (#9), Nance et al. 2006 (#12), Delgrande & Levesque 2012 (#18), Shapiro & Pagnucco 2004 (#21), Boutilier 1995 (#24), Dupin de Saint-Cyr & Lang (#25), explanatory diagnosis (#3, #5, #6, #7), and Cordier & Thiébaux (#27). These are all semantic, plausibility-based or abductive, not literal-rule-based.
- (3) The Π⁺/Π⁻ representation matches 0-approximation / PKS / PSIPLAN (#34, #36, #37) and the approximate filtering of #16 and #17. All of these assume known actions.
- (4) Rule-based observation updates appear in an ad hoc form in BB-WM (#38), without formal soundness and with known actions.

**Most important additions vs. the earlier non-systematic list:** Hunter & Delgrande JAIR 2015 (#9), Nance et al. AAAI 2006 (#12), Shapiro & Pagnucco ECAI 2004 (#21), Boutilier IJCAI 1995 (#24), Dupin de Saint-Cyr & Lang (#25), Banihashemi et al. Sec. 7 monitoring gap (#1), Bonet & Geffner JAIR 2014 (#17), Mombourquette et al. AAAI 2017 (#16), Eppe & Bhatt (#34), and BB-WM 2026 (#38).

**Inferences (reviewer's, not sourced):**
- The thesis rules can be positioned as a sound, tractable, syntactic approximation of filtering over the disjunction of possible abstract operators. This would explain why ρ is sound when its Γ_obs conditions entail the operator's occurrence. A formal comparison to Nance et al. and to Hunter & Delgrande's ranking semantics would strengthen the differential.
- The truthful-ψ assumption (Sandewall's K, as in #25) separates the thesis from the noisy-sensing line (#23, #39) and should be stated explicitly as a scope restriction.

---

## 7. Limitations of the search

- **Consensus unusable:** the monthly quota was exhausted on the first call and the next two calls were rate-limited, so no Consensus results exist. Coverage relied on alphaXiv (arXiv only; weak for KR/IJCAI/AAAI/JAIR papers not on arXiv; one historical query returned 0 results) and WebSearch.
- **WebSearch session limit** was hit on Q23/Q24, so the targeted re-verification of LLM-State, Embodied Agent Interface (Li et al. NeurIPS 2024), ROSPlan and Hanheide et al. AIJ 2017 through web search could not be done. LLM-State was found via alphaXiv instead.
- **Forward citations not checked properly:** the Semantic Scholar API, DBLP and OpenAlex were blocked (proxy CONNECT 403 / DNS ENOTFOUND). Google Scholar was not attempted. Forward snowballing is therefore approximate (search-surfaced extensions only).
- **Full texts unreachable** for all non-arXiv seeds: Sohrabi et al. KR 2010, Amir & Russell IJCAI 2003, Yu et al. IJCAI 2013, Delgrande & Levesque KR 2012, Hunter & Delgrande JAIR 2015, Nance et al. AAAI 2006, Boutilier 1995, Shapiro & Pagnucco 2004. Their verdicts rest on abstracts or search summaries and on reference-list mentions, and should be confirmed by reading the PDFs.
- **WebSearch summaries are LLM-generated:** details taken from them (e.g., the Nance et al. "new constants" mechanism, Delgrande & Levesque "wrong button") are snippet-level.
- **alphaXiv "reports" (AR) are AI-generated** summaries of the full text, not the text itself.
- Some bibliographic fields are unverified and flagged inline: Dupin de Saint-Cyr & Lang original venue, Klassen et al. KR 2018 exact title, McIlraith PDF titles, Nance co-authors, LLM-State first author.
- **Screening and counting:** screening was done by a single reviewer (no double screening). Deduplication counts in Section 4 are approximate.
- Many 2026 arXiv preprints were screened at title/snippet level only.
