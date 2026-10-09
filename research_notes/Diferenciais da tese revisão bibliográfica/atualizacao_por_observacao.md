# Observation-triggered knowledge-base update rules (unknown abstract action): closest prior work

Thesis claim under test: the agent keeps Π = (Π⁺, Π⁻). Hand-written rules ρ = (Γ_kb, Γ_obs, Γ_new) fire when Γ_kb ⊆ Π and Γ_obs ⊆ ψ. Firing a rule updates Π at the abstract level. The agent does not know which abstract operator occurred, because operators are triggered as side effects of low-level transitions.

Overlap scale used below:
- **IDENTICAL**: same problem and same mechanism.
- **PARTIAL**: same problem but a different mechanism, or the same mechanism under a different assumption.
- **SUPERFICIAL**: shares vocabulary only.

Effort: about 14 web searches. Abstracts were checked through search snippets, but most full texts were NOT read. Every verdict is my own assessment from the abstracts plus general knowledge of these papers.

## Q1. Updating knowledge/belief from observations when the action that occurred is unknown or not executed by the agent (situation calculus, diagnosis, plan recognition, logical filtering, revision vs update)

### Takeaway
The problem itself is well studied. The closest framings are:
- explanatory diagnosis (Sohrabi, Baier & McIlraith 2010; Yu, Wen & Liu 2013);
- belief change under fallible or unknown actions in the situation calculus (Delgrande & Levesque 2012; Shapiro et al. 2011);
- logical filtering (Amir & Russell 2003).

All of these derive the update from an action theory, either by abducing the unseen actions or by progressing over every possible action. None of them uses hand-written, local, pattern-style rules that map (prior KB, observation) directly to literal changes. So the overlap is PARTIAL at the problem level and only weak at the mechanism level. The thesis should present its rules as a compiled, sound but incomplete shortcut for explanatory diagnosis / filtering, not as a new problem.

### Cited Findings
- **Sohrabi, Baier & McIlraith (2010), "Diagnosis as Planning Revisited", KR 2010, pp. 26–36.** Gives a formal characterization of diagnosis of discrete dynamical systems in the situation calculus. Computing these dynamical diagnoses corresponds to generating plans, i.e. finding sequences of (unobserved) actions that explain the observations. The paper extends this to incomplete information and preferences. — [bibbase entry](https://bibbase.org/network/publication/sohrabi-baier-mcilraith-diagnosisasplanningrevisited-2010); [PHM abridged report](https://www.papers.phmsociety.org/index.php/phmconf/article/view/1958); [PUC Chile listing](https://ing.uc.cl/publicaciones/diagnosis-as-planning-revisited)
  - **Verdict: PARTIAL (same problem, different mechanism).** It covers the thesis's exact situation: the abstract state changed through actions the agent did not knowingly execute, and the agent sees only partial observations. Its mechanism is abduction of explaining action sequences by planning. The thesis's rules do no search; they are local triggers. Each thesis rule can be read as a precompiled explanatory diagnosis: "the only explanation of at(R1) → at(R2) is move(R1,R2), whose effect implies connected(R1,R2)".
  - **What stays distinctive:** no planner/abduction at runtime; a two-level (c, x) structure in which the "faults/exogenous events" are side effects of the agent's own low-level actions; and the use of the KB as a precondition on rule firing.
- **Yu, Wen & Liu (2013), "Multi-Agent Epistemic Explanatory Diagnosis via Reasoning About Actions", IJCAI 2013, pp. 1183–1190.** The problem is to find a sequence of actions that explains an observation. It is extended to multiple agents, where actions are only partially observable to other agents and observations may concern (common) knowledge. The formalization is in dynamic epistemic logic, using Kripke models of actions for partial observability. The general problem is undecidable, and decidable fragments come from restricting to finite epistemic states or action sequences. — [IJCAI abstract](https://www.ijcai.org/Abstract/13/178); [mlanthology](https://mlanthology.org/ijcai/2013/yu2013ijcai-multi)
  - **Verdict: PARTIAL.** It shares the core idea: actions that are not (fully) observed are explained from observations, then knowledge is updated. Its scope is multi-agent and higher-order epistemic, and it computes explanations by search over action models. The thesis is single-agent, first-order knowledge only (literals in Π⁺/Π⁻), and rule-based.
  - **Note:** the examiners cited this paper. The thesis should state explicitly that its rules realize a restricted, single-agent, non-search form of explanatory diagnosis.
- **Delgrande & Levesque (2012), "Belief Revision with Sensing and Fallible Actions", KR 2012.** The agent has incomplete and possibly inaccurate knowledge. It "may inadvertently execute the wrong" action (e.g. push an unintended button). The framework is built on the situation calculus with a plausibility ordering over situations, and it integrates fallible actions, belief revision via informing actions, and sensing. — [PDF (SFU)](https://www2.cs.sfu.ca/~jim/publications/KR12.pdf)
  - **Verdict: PARTIAL; conceptually the closest formal semantics.** The action that actually occurred differs from what the agent believes, and sensing reveals this. That matches "agent executes a low-level a; an unknown abstract operator fires".
  - **Differences:** they use plausibility-ranked situations and full situation calculus reasoning. The thesis uses explicit rules and a three-valued literal KB, with no plausibility ordering.
- **Shapiro, Pagnucco, Lespérance & Levesque, "Iterated Belief Change in the Situation Calculus"** (KR 2000; journal version reportedly AIJ 2011 — **journal details not verified**). Adds a plausibility-based belief operator to Scherl–Levesque knowledge and distinguishes belief update from iterated belief revision under sensing. — [KR PDF (UNSW)](https://cgi.cse.unsw.edu.au/~morri/Papers/KR-151.pdf)
  - Later work notes that it assumes accurate sensing: conflicting sensing leads to inconsistent belief. Ma, Liu & Miller (UAI 2011) extend it to noisy sensing. — [Ma et al. UAI 2011](https://proceedings.mlr.press/r9/ma11a.html)
  - **Verdict: PARTIAL/SUPERFICIAL.** Same goal (belief change from sensing over action theories). The actions are known, and the mechanism is semantic, not rule-based.
- **Iwan & Lakemeyer (2002), "What Observations Really Tell Us"**, AAAI Workshop Technical Report WS-02-05 (a KI 2003 version is mentioned in one bibliography, **not verified**). Shows that a naive formalization of observations made during a course of actions gives unintended results and is sensitive to the form of the successor state axioms. Proposes a proper encoding. Origin: diagnosing plan execution failures. — [AAAI page](https://aaai.org/?p=82602); [studylib copy](https://studylib.net/doc/13662850/what-observations-really-tell-us-gero-iwan)
  - **Verdict: SUPERFICIAL/PARTIAL.** Relevant as a warning: the meaning of an observation depends on the action theory. Thesis rules that hard-code the inference may be unsound if the domain dynamics differ from what the rule author assumed. This supports the need for a soundness condition (see Q4).
- **De Giacomo & Levesque (1999), "Progression and Regression Using Sensors", IJCAI 1999.** Projection with sensing in the situation calculus. Generalized regression is correct whenever applicable; the paper gives conditions under which projection can combine sensing and regression. — [PDF](https://www.diag.uniroma1.it/~degiacomo/papers/1999/DeLe99ijcai.pdf)
  - Related progression work: Vassos & Levesque (IJCAI 2007), progression with incomplete info. — [mlanthology](https://mlanthology.org/ijcai/2007/vassos2007ijcai-progression)
  - **Verdict: SUPERFICIAL.** The action sequence is known; sensing only filters. I did not find a specific De Giacomo & Levesque paper titled "explaining observations" (**gap**).
- **Amir & Russell (2003), "Logical Filtering"**, AAAI Spring Symposium on Logical Formalization of Commonsense Reasoning (also IJCAI 2003 per Russell's page — **venue ambiguity, check DBLP**). The belief state is a logical formula over possible world states. The paper gives algorithms that keep it compact over time for classes of environments. — [PDF](https://aima.eecs.berkeley.edu/~russell/papers/mini03s-filter.pdf); [AAAI page](https://aaai.org/?p=67332)
  - Follow-ups: Shirazi & Amir, "First-Order Logical Filtering", IJCAI 2005 ([link](https://mlanthology.org/ijcai/2005/shirazi2005ijcai-first)); Shahaf & Amir, "Logical Circuit Filtering", IJCAI 2007 ([link](https://mlanthology.org/ijcai/2007/shahaf2007ijcai-logical)).
  - **Verdict: PARTIAL.** It is exact state estimation, progress(belief, action) then filter(observation), at the logical level. Standard filtering assumes the executed action is known (as an input to the transition). The thesis's unknown abstract operator corresponds to filtering with a disjunctive/nondeterministic transition over all operators that the low-level transition could have triggered. Logical filtering can model that. The thesis instead hand-writes the outcome of that filtering for selected cases. A good positioning: "thesis rules = a sound, incomplete, cheap approximation of logical filtering in a literal (0-approximation) belief representation".
- **Amir & Chang (2008), "Learning Partially Observable Deterministic Action Models", JAIR 33:349–402.** SLAF (simultaneous learning and filtering): exact learning of STRIPS-style action models from partial observations, polynomial for common action classes. — [DOI 10.1613/jair.2575](https://www.doi.org/10.1613/JAIR.2575); [arXiv 1401.3437](https://arxiv.org/pdf/1401.3437)
  - **Verdict: SUPERFICIAL for the update rules.** SLAF assumes the actions are observed/known and the model is unknown; the thesis is the reverse (model known, action unknown). It is more relevant if the thesis ever learns its rules or operators.
- **Ramírez & Geffner, plan/goal recognition as planning.** IJCAI 2009 ([abstract](https://www.ijcai.org/Abstract/09/296)), AAAI 2010 ([link](https://ojs.aaai.org/index.php/AAAI/article/view/7745)), IJCAI 2011 "Goal Recognition over POMDPs" ([abstract](https://ijcai.org/Abstract/11/335)). These infer goals from observed action sequences with gaps.
  - The fluent-observation variant: Amado, Pereira, Meneguzzi et al., "Partial-Order, Partially-Seen Observations of Fluents or Actions for Plan Recognition as Planning" (arXiv 1911.05876). — [arXiv](https://arxiv.org/pdf/1911.05876)
  - **Verdict: SUPERFICIAL.** The inference target is the goal/plan, not the current symbolic state. The mechanism uses a planner. The fluent-observation variant is closer: it infers from observed fluents without seeing actions. The goal is still recognition, not KB maintenance.
- **Katsuno & Mendelzon (1991), "On the Difference between Updating a Knowledge Base and Revising It", KR 1991, pp. 387–394.**
  - **Update** = the world changed: update each model toward its closest models.
  - **Revision** = new info about a static world.
  - [bibsonomy](https://bibsonomy.org/bibtex/8e79694b85966562ce7429ace21e6d98)
  - **Verdict: SUPERFICIAL (framing only).** The thesis rules mix both. Γ_obs ⊆ ψ adds observed literals (revision/expansion of knowledge about the current state). Γ_new removes literals such as on(X,R1) because the world changed (update). The thesis can use the KM distinction to justify why a rule must explicitly delete outdated literals rather than simply adding observations.

### Inferences
- The problem the thesis addresses has long been formalized in KR: tracking the abstract state when the action that changed it is unknown. Explanatory diagnosis is the closest, especially the line from McIlraith → Sohrabi/Baier/McIlraith → Yu/Wen/Liu. Claiming novelty for the problem would be hard to defend.
- What remains defensible as distinctive:
  1. **The mechanism:** declarative, local, rule-based update over a literal KB, conditioned jointly on prior knowledge and the current observation. There is no abductive search, no planner, and no full action theory used at runtime.
  2. **The setting:** a two-level (c, x) model in which the unknown abstract operators are side effects of the agent's own low-level actions (events). They are not exogenous actions by nature or other agents.
  3. **Integration** with an RL/low-level control agent.
- The cleanest positioning: the rules are a compiled, sound-but-incomplete approximation of logical filtering / explanatory diagnosis under the 0-approximation (literal) belief representation.

### Gaps
- I did not locate a specific De Giacomo/Levesque "explaining observations with exogenous actions" paper. Reiter's book (Knowledge in Action, 2001) treats exogenous actions; this was not verified in this session.
- The journal version of Shapiro et al. (AIJ 2011) was not verified.
- The venue of Amir & Russell 2003 is ambiguous (AAAI Spring Symposium vs IJCAI-03).
- Ramifications/static causal laws and PDDL derived predicates were not searched in this session. From general knowledge (unverified here), derived predicates compute literals from the state via axioms but are not conditioned on the previous KB, so they cannot express "knew R1, now see R2 ⇒ connected(R1,R2)". That temporal (two-time-point) condition is what separates thesis rules from static axioms.

## Q2. Rule- or knowledge-based symbolic belief tracking in robotics / cognitive architectures

### Takeaway
Robotic planning systems update symbolic KBs from perception, but usually by direct, per-predicate sensor-to-fact mappings (ROSPlan) or by modelling epistemic effects of known actions (Hanheide et al.; Kaelbling & Lozano-Pérez). The thesis's rules condition on both the prior KB and the observation to infer non-observed facts. That makes them a slightly richer mechanism than ROSPlan sensing, but similar in spirit to engineering practice. Overlap: PARTIAL/SUPERFICIAL.

### Cited Findings
- **ROSPlan sensing interface.** A YAML file maps each PDDL predicate to a ROS topic, message type and optional comparison or custom Python function. Examples: `battery_getting_low` from a threshold on the battery topic, `robot_at` from odometry. The KB is otherwise updated by applying at-start/at-end effects of the dispatched actions. — [ROS Answers](https://answers.ros.org/question/360522/rosplan-battery-monitor/); [ROSCon 2019 slides](https://roscon.ros.org/2019/talks/roscon2019_rosplan.pdf)
  - A 2025 survey states that ROSPlan struggles with partial observability and noisy sensing. — [arXiv 2505.04493](https://arxiv.org/pdf/2505.04493)
  - **Verdict: PARTIAL (mechanism-level, engineering).** These are hand-written rules from observation to KB facts, so they are very close in spirit to Γ_obs → Γ_new. As far as the snippets show, they are not conditioned on prior KB content and have no formal semantics or soundness.
  - **Differences:** the thesis adds Γ_kb (temporal, two-time-point inference), a formal open-world three-valued KB, and a soundness notion. I did not verify whether custom Python sensing functions can query the KB; they probably can, but informally.
- **Hanheide et al. (2017), "Robot Task Planning and Explanation in Open and Uncertain Worlds", Artificial Intelligence 247:119–150** (online 2015; DOI 10.1016/j.artint.2015.08.008).
  - The approach has three knowledge layers (instance, default/probabilistic, diagnostic). Actions have epistemic effects and assumptions. It covers planning under uncertainty, open worlds, explaining task failure, and verifying explanations. It was evaluated on a mobile robot in object search and room categorization.
  - [Lincoln repository](https://repository.lincoln.ac.uk/articles/journal_contribution/Robot_task_planning_and_explanation_in_open_and_uncertain_worlds/24343243); [DOI](https://doi.org/10.1016/j.artint.2015.08.008)
  - **Verdict: PARTIAL.** It shares open-world belief maintenance, and failure explanation from observations is a diagnosis-like step. Its knowledge changes come from epistemic effects of known actions plus probabilistic reasoning. The thesis's trigger is observation plus prior knowledge with the operator unknown. Hanheide's explanation of failures is the closest analogue to "inferring what happened"; that is an inference, as I did not read the full text.
- **Kaelbling & Lozano-Pérez (2013), "Integrated Task and Motion Planning in Belief Space", IJRR 32(9–10):1194–1227.** Planning in belief space with a vocabulary of logical expressions (fluents) that describe sets of belief states, and symbolic operators producing task-oriented perception. Tested on a PR2. — [author PDF](https://lis.csail.mit.edu/pubs/tlp/IJRRBelFinal.pdf); [DSpace](https://dspace.mit.edu/handle/1721.1/87038)
  - **Verdict: SUPERFICIAL/PARTIAL.** It has symbolic belief fluents over a continuous low-level estimator, which is analogous to the (c, x) split. The underlying belief update is probabilistic (Bayesian filtering at the low level). Belief fluents are evaluated from the distribution, not derived by KB+observation rules about a hidden symbolic state.
- **Talamadupula, Benton, Kambhampati, Schermerhorn & Scheutz (2010), "Planning for Human-Robot Teaming in Open Worlds", ACM TIST 1(2), Art. 14.** Open world quantified goals: sensing new objects (e.g. rooms or victims in USAR) creates new facts and goals during execution with replanning. — [PDF](https://hrilab.tufts.edu/publications/talamadupulaetal10tist.pdf)
  - **Verdict: SUPERFICIAL.** It shares the open-world framing and the idea that sensing adds facts. Its updates are direct observation insertion; it does not infer unobserved facts from KB plus observation.
- **"Jiang et al. 2019 open-world planning"**: I could not verify a paper matching this description.
  - Closest found: Jiang, Zhang, Khandelwal & Stone (2019), "Task Planning in Robotics: an Empirical Comparison of PDDL- and ASP-based Systems", FITEE 20(3):363–373. — [link](https://www.cs.utexas.edu/~pstone/Papers/bib2html/b2hd-FITEE19-jiang.html)
  - Also found: Ding, Zhang, Zhang et al., COWP / "Robot Task Planning and Situation Handling in Open Worlds" (arXiv 2210.01287). — [arXiv](https://arxiv.org/html/2210.01287v2)
  - **Verdict: SUPERFICIAL.** Both concern planning, not observation-driven KB inference.
- **Banihashemi, De Giacomo & Lespérance (2017), "Abstraction in Situation Calculus Action Theories", AAAI 2017.**
  - A refinement mapping maps each high-level action to a ConGolog program over the low-level theory, and each high-level fluent to a low-level state formula. Soundness and completeness of the abstraction are defined via bisimulation, and exogenous actions are included. — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/10693); [ext. PDF](https://www.eecs.yorku.ca/~lesperan/papers/AAAI17extVersion.pdf)
  - Extension: IJCAI 2018, "Abstraction of Agents Executing Online and their Abilities in the Situation Calculus", which handles sensing during execution. — [IJCAI](https://www.ijcai.org/proceedings/2018/235)
  - Extended preprint: arXiv 2410.14712 (2024). — [arXiv](https://www.arxiv.org/abs/2410.14712)
  - **Verdict: PARTIAL; this is the closest two-level formal work.** Both have a high-level symbolic theory over a low-level one, with abstract actions realized by low-level behaviours, plus a formal soundness notion.
  - **Key difference:** there, high-level fluents are defined by low-level formulas, so the abstract state is a function of the low-level state. In the thesis, c is hidden and independent of x and is only partially revealed by Φ(c, x). That is why the thesis needs belief tracking at all. The thesis should cite this line and make the contrast explicit.

### Inferences
- In robotics, KB-update-from-perception is ubiquitous but mostly ad hoc (ROSPlan-style per-predicate sensing). The thesis's formal contribution is a declarative rule language, conditioned on both KB and observation, with an open-world three-valued KB and a soundness criterion. That is a modest but defensible differential over these systems.
- The Banihashemi et al. abstraction line is the strongest candidate an examiner might raise against the "two-level" aspect. The defense rests on the hidden c (not a function of x) and on the event-triggered abstract operators.

### Gaps
- KnowRob was not searched in this session. From general knowledge, KnowRob uses computable predicates/virtual KBs over perception data. That is likely SUPERFICIAL/PARTIAL, but it is unverified here.
- I did not verify whether ROSPlan's sensing functions can condition on existing KB facts.

## Q3. Neuro-symbolic RL that tracks symbolic state from observations

### Takeaway
Li et al. (NeurIPS 2024) is the closest neuro-symbolic RL work. It treats the uncertain interpretation of propositional symbols under partial observability as a POMDP and tracks a belief over the reward machine state. It addresses symbolic-state tracking from low-level observations, but it tracks automaton states rather than a KB of literals about a hidden world, and it uses learned/probabilistic labelling rather than KB-conditioned rules. Overlap: PARTIAL.

### Cited Findings
- **Li, Chen, Klassen, Vaezipoor, Toro Icarte & McIlraith (2024), "Reward Machines for Deep RL in Noisy and Uncertain Environments", NeurIPS 2024** (arXiv 2406.00120). Ground-truth interpretations of the domain-specific vocabulary are "elusive" due to partial observability and noisy sensing. The problem is cast as a POMDP, and a suite of RL algorithms is proposed that exploit task structure under uncertain interpretations. Naive approaches have pitfalls. — [NeurIPS](https://papers.neurips.cc/paper_files/paper/2024/hash/c71769e2715835d37c3e25cc1173bd62-Abstract-Conference.html); [arXiv](https://arxiv.org/abs/2406.00120v3)
  - **Verdict: PARTIAL.** It shares the question of how to track symbolic task-relevant state when the labelling function is not directly observable.
  - **Differences:**
    - Their uncertainty is about whether propositions are currently true (noisy labels), handled probabilistically. The thesis's uncertainty is about a hidden persistent symbolic state c, with truthful but partial observations.
    - Their tracked object is the RM state. The thesis tracks a KB over a rich relational theory (connected, on, holding).
    - The thesis infers non-observed facts (e.g. connected(R1,R2)) from temporal KB+observation patterns. Li et al. do not do this (inference from abstract, not full text).
- **Bonet & Geffner belief-tracking line** (already known, listed for completeness):
  - AAAI 2012, width-based factored belief tracking — [PDF](https://www-i6.informatik.rwth-aachen.de/~hector.geffner/www.dtic.upf.edu/~hgeffner/bel-tracking-aaai-2012.pdf)
  - IJCAI 2013, causal belief decomposition — [abstract](https://ijcai.org/Abstract/13/336)
  - JAIR 2014, "Belief Tracking for Planning with Sensing: Width, Complexity and Approximations". Beam tracking is a sound approximation, exponential in causal width, and real-time on Battleship, Minesweeper and Wumpus. — [JAIR](https://jair.org/index.php/jair/article/download/10901/25996)
  - **Verdict: PARTIAL.** These are sound approximate belief trackers derived automatically from the model. The thesis rules are hand-written; they could be positioned as a hand-specified analogue of a sound, incomplete tracker.

### Inferences
- The neuro-symbolic RL literature tracks automaton or label beliefs probabilistically. I found no RL work using KB-plus-observation symbolic rules to maintain an open-world literal KB. This supports the differential within RL, with the caveat that this search was shallow.

### Gaps
- I did not search specifically for "symbolic state estimation" rule systems in RL, e.g. Minecraft/Crafter agents, or LLM-agent world-state trackers from 2023–2026. These are a likely source of informal overlap (LLM agents maintaining a textual world state from observations) and are worth a dedicated search.

## Q4. Formal treatment of soundness of observation-driven update rules relative to a hidden true state

### Takeaway
Soundness of belief tracking relative to the true state is formalized in several places:
- logical filtering (exactness);
- Bonet & Geffner beam tracking ("sound approximation");
- 0-approximation semantics;
- abstraction soundness via bisimulation (Banihashemi et al.).

I found no paper that defines soundness for a set of user-written (KB-condition, observation-condition, effect) rules over a hidden symbolic state. A soundness theorem for ρ-rules would be one of the more clearly distinctive contributions, provided the thesis proves one.

### Cited Findings
- Beam tracking is described as "a sound approximation scheme". — [Bonet & Geffner JAIR 2014](https://jair.org/index.php/jair/article/download/10901/25996)
- Logical filtering maintains a formula that exactly represents the set of possible states. — [Amir & Russell 2003](https://aima.eecs.berkeley.edu/~russell/papers/mini03s-filter.pdf)
- Sound/complete abstractions are defined via bisimulation between high- and low-level models. — [Banihashemi et al. AAAI 2017](https://ojs.aaai.org/index.php/AAAI/article/view/10693)
- Iwan & Lakemeyer show that naive observation encodings yield unintended conclusions depending on the form of the successor state axioms. Hand-coded observation semantics can therefore be unsound. — [AAAI page](https://aaai.org/?p=82602)

### Inferences
- A natural soundness condition for a rule ρ:
  - Precondition: for every (c, x, a) with Γ_kb true in the previous c (consistent with the previous Π) and Γ_obs ⊆ Φ(c', x').
  - Requirement: every abstract operator sequence that the event semantics can trigger on (x, a, x') yields a c' satisfying Γ_new.
  - This is exactly "Γ_new is entailed by progression/filtering under all explanations". It ties the thesis rules formally to explanatory diagnosis and logical filtering, which is the honest way to present them: a compiled entailment, checkable offline.

### Gaps
- I found no published formal soundness result specifically for KB-and-observation-conditioned update rules. That is absence of evidence from a limited search (about 14 queries), not proof of novelty.
