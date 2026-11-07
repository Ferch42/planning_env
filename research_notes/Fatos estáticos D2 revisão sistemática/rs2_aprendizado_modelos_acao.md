# Systematic review, facet RS2: learning action models under partial observability, and how static predicates are handled (thesis differential D2)

Review date: 2026-11-07. One reviewer (a research agent). This facet extends two earlier notes and does not repeat them:
- `../Diferenciais da tese revisão sistemática/rs_fatos_estaticos.md` (protocol-based, 2026-10-09). It covered OGAMUS, James et al. 2020, HiP-MDP, CTP, K-replanner and SDR, and graded claims (a), (b) and (c).
- `../Diferenciais da tese revisão bibliográfica/fatos_estaticos.md` (no protocol).

This note goes deeper into the **action-model-learning (AML) literature**. Most of the items in it were missing from the earlier notes or were assessed there from snippets only.

**Reading labels:**
- **[full text]**: the PDF was read in this session through alphaXiv `answer_pdf_queries`, with targeted queries returning the relevant pages.
- **[abstract/snippet]**: only the abstract or search snippets were seen.
- **[cited by X]**: the claim comes from the related-work section of paper X, which was read in full.
- **[memory, unverified]**: background knowledge that was not confirmed in this session.

**The D2 claim under test.** Given fixed lifted templates (action schemas), the values of the static predicates select which transition system holds. The agent does not know these values. It identifies them by interacting with the environment, abducing preconditions from successes and failures, and tracks what it knows as a pair (Π⁺, Π⁻) updated by observation rules. The claim is "identifying static facts from interaction, given known templates, with identifiability guarantees, as the dual of action-model learning (where the schemas are unknown and the static facts are usually given)".

---

## Q0. Protocol, search log and PRISMA counts

### Takeaway
The protocol was fixed before searching. I ran 17 distinct search strings on two tools (12 on WebSearch, 5 on alphaXiv `discover_papers`) and snowballed from 4 seeds. In the PRISMA-style flow, about 216 records were identified and 32 studies were included; 9 of those were read in full text. No study matched D2 exactly (no IDÊNTICO verdict).

### Protocol (written before searching)
- **Research question (RQ-RS2).** In learning planning action models (STRIPS/PDDL, lifted) from traces or interaction, under full or partial observability:
  - (i) how are static predicates and static facts treated: learned, given, or ignored?
  - (ii) does any work fix the schemas and learn only the instance-level static facts ("learning the problem, not the domain")?
  - (iii) which works learn online by interaction, and which offline?
  - (iv) what guarantees are given (safety/soundness, completeness, convergence, identifiability, sample complexity)?
- **Inclusion criteria.**
  - Peer-reviewed paper or arXiv preprint, 1998–2026.
  - Proposes a method or theory in which an agent or learner infers action models, or persistent/static relational facts, from observations or interaction.
  - Symbolic (relational or propositional) target.
- **Exclusion criteria.**
  - Purely continuous control or motion planning with no symbolic structure.
  - Duplicates.
  - Non-English texts.
  - LLM-only pipelines with no explicit symbolic model of preconditions and facts.
  - Patents and aggregators as the sole source.
- **Tools.**
  - WebSearch (standard mode, plus extended mode once).
  - alphaXiv `discover_papers`, and `answer_pdf_queries` for full text.
  - WebFetch was attempted on `ojs.aaai.org` and `ai.dmi.unibas.ch` and was blocked (ENOTFOUND), so AAAI/ICAPS PDFs hosted there could not be opened.
  - Consensus was not used, because its quota had been exhausted in the previous review. The Semantic Scholar and DBLP APIs were known to be blocked from the previous review and were not retried.

### Search log (all run 2026-11-07)
"Hits" is the number of records the tool returned. "Kept" is the number passed to eligibility.

| # | Exact string | Tool | Hits | Kept |
|---|---|---|---|---|
| S1 | question "action model learning from traces under partial observability, how static predicates are learned or assumed given (SLAF, ARMS, LOCM, FAMA, SAM, OffLAM, ROSAME, Latplan)"; keywords [action model learning, static predicates, partial observability, SAM, FAMA, LOCM, OffLAM] | alphaXiv | 10 | 6 (2411.14995, 2508.21449, 2605.18627, 2609.25766, 2605.13282, 2608.30955) |
| S2 | question "learning the initial state or instance-level facts (map, connectivity) of a planning problem when the action schemas are known, by interacting with the environment; agent interrogation; active learning of action models"; keywords [agent interrogation, active learning, action models, initial state] | alphaXiv | 10 | 2 (2303.04912 screened out as continuous/bilevel; 2602.03974 kept as context) |
| S3 | `OffLAM offline learning action models partial observability Lamanna static predicates` | WebSearch | 10 | 3 (OffLAM AIJ 2025, PI-SAM AAAI 2024, STRIPS Action Discovery) |
| S4 | `SLAF simultaneous learning and filtering Amir Chang partially observable action models` | WebSearch | 10 | 2 (SLAF JAIR 2008, Shahaf et al. AAAI 2006) |
| S5 | `FAMA learning action models minimal observability static predicates input` | WebSearch | 10 | 2 (FAMA AIJ 2019, 2402.10726) |
| S6 | `Lindsay "Reuniting the LOCM family" static relationships` | WebSearch | 17 | 0 (all off-topic; re-run as S14) |
| S7 | `Walsh Littman efficient learning of action schemas and web-service descriptions AAAI 2008 KWIK` | WebSearch | 10 | 1 |
| S8 | `Le Juba Stern "Learning safe action models with partial observability" PI-SAM static` | WebSearch | 10 | 1 |
| S9 | `Verma Marpally Srivastava "Asking the right questions" agent interrogation action models static predicates` | WebSearch | 10 | 1 (AIA AAAI 2021, 1912.12613) |
| S10 | `infer unknown initial state facts from action failures known PDDL domain learning instance problem not domain` | WebSearch (extended) | 9 | 1 (conditional-effects SAM 2403.15251, context only) |
| S11 | `Gregory Cresswell LOP static relations optimal plans induce static predicates abstract` | WebSearch | 10 | 1 (LOP ICAPS 2015 / IJCAI 2016) |
| S12 | `OLAM online learning action models PDDL Lamanna static predicates "always true" preconditions` | WebSearch | 10 | 1 (OLAM IJCAI 2021) |
| S13 | `Lamanna "Planning for Learning Object Properties" AAAI 2023 PDDL known domain learn object properties perception` | WebSearch | 10 | 1 |
| S14 | `Lindsay 2021 KEPS LOCM static relationships identifying static predicates action traces` | WebSearch | 10 | 1 (KEPS 2021 paper 12) |
| S15 | `Bachor Behnke learning planning domains non-redundant fully observed traces complexity static` | WebSearch | 10 | 2 (AAAI 2024; ICAPS 2025 follow-up) |
| S16 | question "plan recognition or diagnosis that infers unknown initial state facts (incomplete initial state) from observed action executions given a known domain model"; keywords [incomplete initial state, plan recognition, diagnosis, known domain model] | alphaXiv | 15 | 2 (2203.03485 self-directed AML; 2303.14272 open-world model adaptation), both context |
| S17 | question "OffLAM: lifted action models learning from partial traces…"; keywords [OffLAM, partial traces, lifted action models] | alphaXiv | 10 | 1 new (2604.19043, Xi et al., unsupervised visual traces) |

### Snowballing (seeds, and what each one yielded)
- **Seed A: SIFT** (Gösgens, Jansen & Geffner, [arXiv 2411.14995](https://www.alphaxiv.org/abs/2411.14995)) [full text].
  - Backward: LOCM, LOCM2, LOP, Lindsay 2021, Bonet & Geffner 2020, Rodriguez et al. KR 2021, Latplan.
  - Forward (found via S1): SYNTH ([2508.21449](https://www.alphaxiv.org/abs/2508.21449)) and SYNTH+/SIFT+ ([2605.18627](https://www.alphaxiv.org/abs/2605.18627)).
- **Seed B: Balyo et al.**, "Planning Domain Model Acquisition from State Traces without Action Parameters" ([arXiv 2402.10726](https://www.alphaxiv.org/abs/2402.10726)) [full text]. Its related-work section is a 6-group mini-survey. Backward from it: OBSERVER, EXPO, OpMaker, TRAIL, ARMS, LAMP, AMAN, ADMN, LOUGA, ALICE, SLAF, AMLSI, LSSS, LPROB, FAMA, SAM.
- **Seed C: STRIPS Action Discovery** (Suárez-Hernández et al., [arXiv 2001.11457](https://www.alphaxiv.org/abs/2001.11457)) [full text]. Backward from it: FAMA, Aineto et al. ICAPS 2018, LOCM family, Amado et al. 2018 (Latplan-based action learner).
- **Seed D: NeSyAM** (Kikaj et al., [arXiv 2609.25766](https://www.alphaxiv.org/abs/2609.25766)) [full text]. A forward citation of ROSAME and OffLAM. Backward from it: ROSAME-I (Xi, Gould & Thiébaux ICAPS 2024), Xi et al. ICAPS 2026, Lamanna & Serafini ICAPS 2024 (noisy traces), Jin et al. 2022.
- The FAMA, SAM and LOP seeds were already snowballed (forward) in the previous review, so here they were reached backward from Seeds A–D.

### PRISMA-style flow (approximate; deduplicated by hand by title)
- **Identified:** 191 records from database searches (sum of Hits), plus about 25 new records from the reference lists of the 4 seeds. Total about 216.
- **Duplicates removed:** about 62. For example, 2411.14995, 2508.21449 and 2609.25766 each appeared 3–4 times, and OffLAM appeared 3 times.
- **Screened** on title and abstract: about 154.
- **Excluded at screening:** about 116. Reasons: off-topic (S6 genealogy pages); continuous or robotics-only; LLM-only; aggregators; goal recognition without fact inference.
- **Assessed for eligibility:** 38.
  - Full text (9): 2402.10726, 2001.11457, 2508.21449, 2609.25766, 2411.14995, 2404.09631, 1912.12613, 2605.18627, plus the reference list of OffLAM as reported by citing papers.
  - Abstract or snippet: the rest.
- **Included:** 32 (table in Q5). 6 of them were carried over from the earlier notes as cross-references: LOP, SAM, FAMA, Bolander & Gierasimczuk, OGAMUS and Bonet & Geffner 2020.

### Gaps
- Hit counts are search-engine pages of about 10 results, not database totals, so recall is not measurable.
- The ICAPS and AAAI OJS PDFs could not be opened. LOP, Lindsay 2021, PI-SAM, OLAM, OffLAM and Lamanna AAAI 2023 were therefore assessed from abstracts or from citing papers.

---

## Q1. How do the main AML systems treat static predicates? Which learn them, and which assume them given?

### Takeaway
Almost all state-based learners (ARMS, SLAF, FAMA, SAM, PI-SAM, OLAM, OffLAM, VSLAM, ROSAME, NeSyAM, SYNTH) assume that the static facts of each training instance are **given**, inside observed or initial states. What they learn is only whether a static predicate appears as a *precondition of a schema*. Only the action-trace-only family (LOCM2 → LOP → Lindsay 2021 → SIFT) *infers* static relations. Even there, they are a by-product of schema learning, inferred offline from positive traces.

SIFT, and SYNTH+ after it, say explicitly that static predicates are "not a true property of the domain but of the instances". That supports D2's framing, but they leave learning them as future work.

### Cited Findings
- **SIFT** [full text].
  - Static predicates "just control the grounding of the actions … while it's possible to learn them from traces, it is not strictly necessary".
  - Instead it introduces one static predicate p_a per lifted action, and "atoms p_a(o) are set to true in I_T iff the ground action a(o) appears in a trace". Its soundness and completeness theorem is stated "for … a hidden instance P = ⟨D, I⟩ with no static predicates, as static predicates can be treated separately". — [Gösgens, Jansen & Geffner, ICAPS 2025, arXiv 2411.14995](https://www.alphaxiv.org/abs/2411.14995)
  - SIFT's features "do not see" static predicates, or predicates that behave as static in the traces. Its listed future work includes "the derivation of static predicates of lower arity". — [arXiv 2411.14995](https://www.alphaxiv.org/abs/2411.14995)
- **SYNTH+** [full text].
  - "static predicates, namely, those which are not affected by the actions do not play any role in these definitions as they are not a true property of the domain but of the instances".
  - In its Sokoban experiment, "only the static predicates defining the grid are observed". Static predicates are thus an *input*.
  - It also defines **local observability**: an agent observes only adjacent(x, x′) atoms for its own cell.
  - Conclusion: "Future work includes effective methods for learning static predicates". — [Gösgens, Jansen & Geffner, arXiv 2605.18627 (2026)](https://www.alphaxiv.org/abs/2605.18627)
- **SYNTH (STRIPS+)** [full text].
  - It defines atoms as "static" if they appear in no effect: "Static atoms appear in the initial situation and action preconditions, and affect just the action groundings".
  - States are fully observed, so the static atoms are observed.
  - In NAVIG, the implicit argument z′ is determined because "a single (static) true atom above(z′, z)" exists. — [Jansen, Gösgens & Geffner, KR 2025, arXiv 2508.21449](https://www.alphaxiv.org/abs/2508.21449)
- **LOP.**
  - "static predicates can be seen as restrictions on the valid groundings of actions". Without them, LOCM2 domains "produce plans that are typically shorter than the true optimal solutions".
  - LOP finds "a set of minimal static predicates for each operator that preserves the length of the optimal plan". This requires optimal, goal-directed plans, though it is still useful with suboptimal ones. — [Gregory & Cresswell, ICAPS 2015](https://ojs.aaai.org/index.php/ICAPS/article/view/13729) [abstract/snippet]
- **Lindsay 2021.** An approach "for identifying static predicates, which does not rely on unit-cost optimal plans". It takes correct actions as input and uses a Boolean test on the examples to validate static relationships. It is reported to be more accurate than LOP. — [Lindsay, KEPS 2021](https://ai-icaps.dmi.unibas.ch/workshops/KEPS/Papers/KEPS_2021_paper_12.pdf) [abstract/snippet]
- **LOCM.**
  - Balyo et al. say LOCM "learns predicates, including static predicates", and that LOP adds "preconditions with static predicates" [cited by 2402.10726](https://www.alphaxiv.org/abs/2402.10726).
  - Gregory & Cresswell's own motivation for LOP is that LOCM2 domains lack the static restrictions [LOP](https://ojs.aaai.org/index.php/ICAPS/article/view/13729).
  - The two sources conflict. The more reliable reading is that LOCM/LOCM2 do not induce static preconditions in general, and LOP was built to fill that gap.
- **STRIPS Action Discovery (UDAM).**
  - Its input contains full initial states, including static facts. The visitall example has conn[p1,p2] and conn[p2,p1] in s0.
  - It "solves the problem of programming static predicates by initializing partial action schemas with all possible predicate-parameter combinations and applying actions that remove preconditions".
  - So it learns *which* static preconditions schemas have; the static facts themselves are given. — [Suárez-Hernández et al., arXiv 2001.11457](https://www.alphaxiv.org/abs/2001.11457)
- **FAMA.**
  - It learns from initial and final states with unobserved intermediate states and actions [cited by 2402.10726](https://www.alphaxiv.org/abs/2402.10726).
  - It assumes action signatures are observable in traces or in a partial model [cited by 2001.11457](https://arxiv.org/pdf/2001.11457).
  - Initial states, and hence static facts, are given. — [Aineto, Jiménez & Onaindia, AIJ 275, 2019](https://riunet.upv.es/bitstream/10251/144560/7/Aineto%3bJim%c3%a9nez-Celorrio%3bOnaindia%20-%20Learning%20action%20models%20with%20minimal%20observability.pdf) [abstract/snippet]
- **SLAF.**
  - It assumes "a fully observable sequence of executed actions, and partially observed states, but without noise" [cited by 2402.10726](https://www.alphaxiv.org/abs/2402.10726).
  - Its output formula "represents the possible transition relations and world states at every time step". So it *jointly* filters unknown state facts (static ones included, in principle) and the transition relation. It makes no static/fluent distinction. — [Amir & Chang, JAIR 33, 2008](https://arxiv.org/pdf/1401.3437) [abstract/snippet]
- **ARMS.** "the initial and final states are known, but intermediate states may be incomplete". It uses weighted MAX-SAT. [cited by 2402.10726](https://www.alphaxiv.org/abs/2402.10726)
- **SAM / ESAM.**
  - They learn from complete traces, and the action parameters are given [cited by 2402.10726](https://www.alphaxiv.org/abs/2402.10726).
  - Static atoms are in the observed states and become preconditions by intersection. This is from SAM's learning rule as described in [2403.15251](https://arxiv.org/html/2403.15251v1): "A literal l only appears in pre(a) for an action a if ¬l has never been observed in the pre-state".
- **PI-SAM / EPI-SAM (Le, Juba & Stern, AAAI 2024).** PI-SAM applies the SAM rules "only … when a literal is observed in the states immediately before and after an action is applied". It is polynomial and outputs a safe PDDL model; EPI-SAM uses the ignored observations to learn more. Static predicates get no special treatment in the abstract. — [AAAI 38(18)](https://ojs.aaai.org/index.php/AAAI/article/view/29995) [abstract/snippet]
- **OffLAM (Lamanna et al., AIJ 339, 2025).**
  - It requires an input model specifying "the predicates and operators signature, plus a list of trajectory files" [AMLGym docs](https://amlgym.readthedocs.io/en/latest/passive_algorithms/offlam.html).
  - It handles partially observable states *and* actions, but "cannot learn preconditions over action arguments that are not observable and do not appear in the effects" [cited by 2605.18627](https://www.alphaxiv.org/abs/2605.18627). It also "comes with no guarantees" [cited by 2508.21449](https://www.alphaxiv.org/abs/2508.21449).
  - Its treatment of static predicates was not verified.
- **NeSyAM vs ROSAME-I (visual traces).**
  - NeSyAM states an identifiability limitation directly relevant to static preconditions: "If a predicate is true before every execution and remains true afterward, both pre and not-involved explain the transition equally well".
  - ROSAME-I resolves this with "an explicit bias for pre conditions", while NeSyAM relies only on trace evidence. The suggested fix is "traces in which uninvolved predicates are sometimes false". — [Kikaj, De Smet, Marra & De Raedt, arXiv 2609.25766 (2026)](https://www.alphaxiv.org/abs/2609.25766)
- **Balyo et al. L1.**
  - Fully observable states; the predicates and their types are given.
  - It notes a "rigid predicate typing" artefact in Barman.
  - It flags Woodworking actions whose saw parameter appears in no precondition or effect ("the domain uses pure object's existence … as a 'hidden' precondition"), which cannot be typed. — [arXiv 2402.10726 (KR 2024)](https://www.alphaxiv.org/abs/2402.10726)
- **LSSS (Bonet & Geffner, ECAI 2020).** "It learns the predicates of the operators and the initial state by using the satisfiability paradigm", from full reachable state graphs. So instance-level facts, static ones included, are learned together with the domain, offline. [cited by 2402.10726](https://www.alphaxiv.org/abs/2402.10726)

### Inferences
- There is a clear **asymmetry** in the literature.
  - State-based AML takes static facts as input and asks *whether a static predicate is a precondition*. This is the "schemas unknown, statics given" setting that D2 calls the dual.
  - Trace-only AML (LOP, Lindsay, SIFT) recovers static relations, but offline, from positive evidence, and as part of learning the schemas.
- D2's "dual" framing is therefore well supported. The most authoritative recent group (Geffner's) explicitly classifies static predicates as instance properties and lists learning them as open future work. That is direct support for the claim that the instance-level problem is under-studied.
- NeSyAM's non-identifiability remark (an always-true literal cannot be told apart from a precondition using positive traces only) is the precondition-side mirror of D2's need for **failed actions (negative evidence)** to populate Π⁻. It is a citable justification for why D2 needs interaction.

### Gaps
- I could not verify whether OLAM and OffLAM have any explicit static-predicate step. The previous review recorded an OLAM remark that extra always-true preconditions are static predicates, but only from a snippet.
- I could not verify how LOP outputs the instance-level tuples of the static relation (for example, whether connected(r1, r2) facts are emitted per instance). The full text could not be opened.
- ARMS, AMAN/ADMN, LAMP, LOUGA and Latplan were assessed only through secondary descriptions.

---

## Q2. Does any work fix the schemas and learn only the instance-level static facts ("learning the problem, not the domain")?

### Takeaway
No AML work found does exactly this. The nearest mechanisms are:
- SIFT's learned instance P_T: given a learned domain, constraint propagation fixes the truth of the relevant atoms along traces, and static p_a facts are set true iff an action was observed.
- LSSS, which learns the initial state together with the domain.
- SYNTH+'s notion of *local observability* of static predicates such as adjacency.

All of these are offline. All use positive evidence only, or a closed-world default. None treats the schemas as given and fixed, and none gives identifiability conditions specifically for static facts.

### Cited Findings
- **SIFT, Theorem 12.** "The truth values of each ground atom p(o) in A(D,T) in each of the states s underlying a set of connected traces in T are fully determined by D and T". The learned instance's I_T is "A(D_T, T) with the truth values … as derived at the initial state of T". In other words, once the domain is fixed, the instance's initial facts follow by propagation; this is essentially what an observation-rule update over known templates does. — [arXiv 2411.14995](https://www.alphaxiv.org/abs/2411.14995)
- **SIFT, static facts.** p_a(o) is true iff a(o) appears in a trace. This is a closed-world rule from positive evidence only, so unobserved but legal groundings are set false. — [arXiv 2411.14995](https://www.alphaxiv.org/abs/2411.14995)
- **SYNTH+, local observability.** "an agent moving in a grid may observe the true atoms at(x) in the state, but only the atoms adjacent(x, x′) involving the single value of x for which at(x) is true". Static adjacency is therefore assumed to be *perceived locally*, not abduced. — [arXiv 2605.18627](https://www.alphaxiv.org/abs/2605.18627)
- **Agent interrogation (AIA).**
  - The interrogator *chooses* the initial states (from a set S collected by random walks) and plans.
  - When an action fails it searches for a state s ⊃ s_F where the action executes, then infers that "any predicate which is false in s will not appear in a_F's precondition in the positive mode". It then adds candidate predicates one at a time to test whether the action becomes inexecutable.
  - The target is the agent's schema model, not instance facts. — [Verma, Marpally & Srivastava, AAAI 2021, arXiv 1912.12613](https://www.alphaxiv.org/abs/1912.12613)
- **OGAMUS** (from the prior review): a known PDDL domain is instantiated online in an unknown environment, with facts obtained by perception, not abduction. — [arXiv 2112.10007](https://www.alphaxiv.org/abs/2112.10007)
- **Lamanna et al., AAAI 2023.** It extends "a base PDDL domain with new actions and predicates for learning object-type properties". It plans to collect training data so that a neural recogniser can map perceptions to PDDL properties. The domain is known, and *perceptual groundings* of predicates are learned. — [AAAI 37(10):12005–12013, arXiv 2301.06054](https://arxiv.org/abs/2301.06054) [abstract/snippet]
- **Bachor, Dekker & Behnke, ICAPS 2025.** Even with the state completely unobserved, "one can infer the existence of certain state features if the plans are assumed non-redundant". They give complexity results (P / NP-complete / coNP / Σ₂). The target is the domain, not instance facts. — [ICAPS 2025](https://ojs.aaai.org/index.php/ICAPS/article/view/36096) [abstract/snippet]

### Inferences
- The "learning the problem, given the domain" direction exists in the literature only **implicitly**: as SIFT's Theorem 12 inside a domain learner, as LSSS's joint search, and as perception-based grounding in OGAMUS and Lamanna 2023.
- D2 can claim novelty for its full combination:
  - (i) schemas given a priori and held fixed;
  - (ii) static facts as the *only* unknowns that select the transition system;
  - (iii) an online loop where both successes and failures update Π⁺ and Π⁻ (not a closed-world default);
  - (iv) identifiability conditions for static facts.
- The thesis should cite SIFT's Theorem 12 and its p_a construction as the closest formal neighbour. It should also point out that SIFT's closed-world rule is unsound for unseen groundings, whereas D2's Π⁻ requires an observed failure. This last point is my inference from SIFT's definition.

### Gaps
- No work was found that equates the static-fact assignment with "which transition system holds" and studies identifiability as a function of the template set. This absence is not proof that none exists: citation indexes were unavailable.

---

## Q3. Online / interactive learning vs offline learning from traces

### Takeaway
The interactive line (OBSERVER, EXPO, Walsh & Littman, OLAM, AIA, VSLAM's online update, the 2026 uncertainty-guided exploration work) learns **schemas** by acting or querying, and uses failures as negative evidence. The static-fact learners (LOP, Lindsay, SIFT, LSSS) are all **offline**. No interactive learner targets instance-level static facts. In the AML literature this is the empty cell D2 occupies; outside AML, the earlier note found OGAMUS, K-replanner and the CTP line, which use perception or sensing rather than abduction.

### Cited Findings
- **OBSERVER and EXPO.** These are early learners where "the models learn by successful and unsuccessful execution of the plans". OBSERVER keeps over- and under-approximations of preconditions that converge as it fails; EXPO "learns missing preconditions from unpredicted outcomes". — [cited by 2402.10726](https://www.alphaxiv.org/abs/2402.10726)
- **Walsh & Littman, AAAI 2008.**
  - "general STRIPS operators cannot be efficiently learned through raw experience, though restricting the size of action preconditions yields a positive result".
  - Efficient learning is possible with a teacher providing solution traces.
  - The KWIK framing appears in Walsh's thesis; its link to the 2008 paper is unverified. — [Sciweavers abstract](https://sciweavers.org/node/198657); [MLAnthology](https://mlanthology.org/aaai/2008/walsh2008aaai-efficient) [abstract/snippet]
- **OLAM.** It learns action models "online, incrementally during the execution of plans", interleaving learning with an exploration phase that selects plans. It starts from predicates and operator signatures with empty preconditions and effects. — [IJCAI 2021](https://www.ijcai.org/proceedings/2021/0566.pdf); [AMLGym docs](https://amlgym.readthedocs.io/en/latest/active_algorithms/olam.html) [abstract/snippet]
- **AIA.**
  - "This policy is computed and executed online".
  - It "requires O(|P*|×|A|) queries".
  - It uses plan-outcome queries that return the longest executable prefix and the final state. — [arXiv 1912.12613](https://www.alphaxiv.org/abs/1912.12613) [full text]
- **VSLAM.** It processes demonstrations "in an online fashion", including negative demonstrations ⟨s, a, ⊥⟩ (failures). It names "an active-learning approach" as future work. — [Aineto & Scala, KR 2024, arXiv 2404.09631](https://www.alphaxiv.org/abs/2404.09631) [full text]
- **Uncertainty-guided exploration (Oregon State, 2026).** It learns conditional and quantified effects through exploration. — [arXiv 2608.30955](https://www.alphaxiv.org/abs/2608.30955) [title/abstract only]
- **Self-directed AML using exploratory planning.** — [arXiv 2203.03485](https://www.alphaxiv.org/abs/2203.03485) [title only]
- **Offline learners:**
  - LOP (optimal plans) — [ICAPS 2015](https://ojs.aaai.org/index.php/ICAPS/article/view/13729)
  - Lindsay (sets of correct actions) — [KEPS 2021](https://ai-icaps.dmi.unibas.ch/workshops/KEPS/Papers/KEPS_2021_paper_12.pdf)
  - UDAM, which "follow[s] the standard offline approach in the field" — [arXiv 2001.11457](https://www.alphaxiv.org/abs/2001.11457)
  - SIFT, SYNTH and SYNTH+ (extended traces or graphs) — [2411.14995](https://www.alphaxiv.org/abs/2411.14995), [2508.21449](https://www.alphaxiv.org/abs/2508.21449), [2605.18627](https://www.alphaxiv.org/abs/2605.18627)

### Inferences
- The interactive AML mechanisms are the *mirror image* of D2:
  - AIA infers "predicate false in s ⇒ not a positive precondition" from a successful execution.
  - D2 infers "action a(o) succeeded ⇒ its static preconditions on o are true" (into Π⁺), and from a failure, with fluents known, "some static precondition is false" (into Π⁻, possibly as a disjunction).
  - The inference pattern is the same; the roles of the known and unknown sides are swapped. This is a good concrete statement of the "dual" claim.

### Gaps
- OLAM's and OffLAM's full texts were not read, so I cannot confirm whether OLAM ever treats an unknown static fact in the *state* (as opposed to in the model) when an action fails.

---

## Q4. Guarantees: safety/soundness, completeness, convergence, identifiability, sample complexity

### Takeaway
The guarantees offered in the literature are all about **schemas**:
- safety/soundness (SAM family, PI-SAM, VSLAM's lower bound);
- completeness (VSLAM's upper bound, SIFT, SYNTH, SYNTH+);
- convergence (VSLAM; OBSERVER informally);
- identifiability in the limit or finite identifiability (Bolander & Gierasimczuk; SIFT Theorem 20);
- PAC-style or sample complexity (SAM, Walsh & Littman);
- complexity of consistency (Balyo: graph-isomorphism-hard and in NP; Bachor & Behnke).

The only guarantee touching instance facts is SIFT's Theorem 12 ("truth values determined by D and T"). No work gives identifiability conditions for static facts as such.

### Cited Findings
- **VSLAM.**
  - The lower boundary gives an *optimal sound* model (Theorem 11).
  - The upper boundary plus the full effect version space gives an *optimal complete* non-deterministic model (Theorem 12).
  - "if L = U = {h} then h = pre(a)" (Corollary 10, convergence).
  - A version space "collapses" when the data are noisy or the hypothesis space is wrong. — [arXiv 2404.09631](https://www.alphaxiv.org/abs/2404.09631) [full text]
- **SIFT.**
  - Theorem 15 (soundness and completeness for complete traces).
  - Theorem 20: there is a finite set of extended traces such that consistency equals validity, so "The learned domain D_T is then equivalent to the hidden domain D". The authors add that "proving that a finite set of traces has this property … may even be undecidable in general".
  - Plain traces cannot learn Grid, because a lock is opened only once from one cell, while extended traces (with state equalities) can. — [arXiv 2411.14995](https://www.alphaxiv.org/abs/2411.14995) [full text]
- **SYNTH.** Theorem 1 (completeness of the precondition queries); soundness requires extra atoms; semantic equivalence of the learned and hidden domains. — [arXiv 2508.21449](https://www.alphaxiv.org/abs/2508.21449) [full text]
- **SYNTH+.** Theorem 8: for a well-formed, strongly connected hidden domain whose dependency graph is acyclic with rank ≤ d_max, SYNTH+ learns an equivalent domain. Theorem 12 extends this to locally observable predicates. — [arXiv 2605.18627](https://www.alphaxiv.org/abs/2605.18627) [full text]
- **AIA.** Theorem 3: it "will always terminate and return a set of models, each of which are functionally equivalent to the agent's model". — [arXiv 1912.12613](https://www.alphaxiv.org/abs/1912.12613) [full text]
- **SAM (first version).** Under injective action binding and bounded predicate arity, "it is sufficient to observe a linear number of state traces with respect to the size of the lifted action model". — [cited by 2402.10726](https://www.alphaxiv.org/abs/2402.10726)
- **PI-SAM.** Polynomial time and "the desired safety guarantee". — [AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/29995) [abstract/snippet]
- **Balyo et al.** Synthesising an action with k arguments "is at least as hard as the graph isomorphism problem and is contained in NP". — [arXiv 2402.10726](https://www.alphaxiv.org/abs/2402.10726) [full text]
- **SLAF.** It is exact in that "no model is misclassified given the observations". — [JAIR 2008](https://arxiv.org/pdf/1401.3437) [abstract/snippet]
- **Walsh & Littman.** Negative result for general STRIPS; positive result for bounded preconditions or with a teacher. — [Sciweavers](https://sciweavers.org/node/198657) [abstract/snippet]
- **Bolander & Gierasimczuk** (from the prior review): deterministic, fully observable actions are finitely identifiable. — [JLC 2018 PDF](https://people.compute.dtu.dk/tobo/bolander2018learning.pdf)
- **NeSyAM.** A precondition and an "always-true, not-involved" literal are not identifiable from positive traces. — [arXiv 2609.25766](https://www.alphaxiv.org/abs/2609.25766) [full text]

### Inferences
- VSLAM's boundaries (L, U) are the formal analogue of D2's (Π⁺, Π⁻), but over *hypotheses about schemas*, not over ground static literals. The thesis can position (Π⁺, Π⁻) as a version space over static-fact assignments with the schemas fixed. Soundness then means planning only with Π⁺-supported groundings, which mirrors SAM/VSLAM safety. Completeness or optimism means also allowing groundings not in Π⁻.
- An identifiability result for D2 would be the dual of SIFT Theorem 20 and Bolander & Gierasimczuk: under which template structures does a finite interaction sequence determine every static fact relevant to the goal? None of the guarantees above answers that question.

### Gaps
- No sample-complexity bound for identifying static facts was found anywhere.

---

## Q5. Included studies with verdicts against D2

### Takeaway
Of 32 included studies, **none is IDÊNTICO** and **none is SOBREPOSIÇÃO FORTE**. 9 are **PARCIAL**. The closest is SIFT: static facts are treated as instance properties and inferred from executed actions, and fluent truth values are propagated given the domain. It is offline, uses positive evidence only with a closed-world rule, and learns the schemas rather than taking them as given. The remaining 23 are **DISTANTE**: they learn schemas with static facts given.

### Cited Findings
Columns: Learns | Observability | Static facts | Guarantees | Mode | Verdict | Read.

| # | Study (bibliographic data, URL) | Learns | Observability | Static facts | Guarantees | Mode | Verdict | Read |
|---|---|---|---|---|---|---|---|---|
| 1 | Gösgens, Jansen & Geffner, "Learning Lifted STRIPS Models from Action Traces Alone: A Simple, General, and Scalable Solution", ICAPS 2025. [arXiv 2411.14995](https://www.alphaxiv.org/abs/2411.14995) | predicates + schemas + instance I_T | action traces only (states hidden) | **inferred per instance**: p_a(o) true iff a(o) observed (closed world); "treated separately" | sound/complete (Thm 15), finite identification (Thm 20), Thm 12 on instance truth values | offline | **PARCIAL** (closest) | full |
| 2 | Gösgens, Jansen & Geffner, "Learning Lifted Action Models from Traces with Minimal Information About Actions and States" (SIFT+/SYNTH+), [arXiv 2605.18627](https://www.alphaxiv.org/abs/2605.18627), 2026 | STRIPS+ domains | partial actions; full or **local** observability of chosen predicates | **given** (observed grid statics); "not a property of the domain but of the instances"; learning them is future work | completeness (Thms 8, 12) | offline | **PARCIAL** (framing; local observation of adjacency) | full |
| 3 | Jansen, Gösgens & Geffner, "Learning Lifted Action Models From Traces of Incomplete Actions and States" (SYNTH), KR 2025. [arXiv 2508.21449](https://www.alphaxiv.org/abs/2508.21449) | STRIPS+ schemas with implicit arguments | full states, partial action arguments | given (in states); static atoms "affect just the action groundings" | correctness/completeness | offline | DISTANTE | full |
| 4 | Gregory & Cresswell, "Domain Model Acquisition in the Presence of Static Relations in the LOP System", ICAPS 2015, pp. 97–105; IJCAI 2016. [ICAPS](https://ojs.aaai.org/index.php/ICAPS/article/view/13729) | static preconditions of operators (on top of LOCM2) | action traces (optimal plans) | **learned** (minimal static relations that preserve optimal plan length) | none formal | offline | **PARCIAL** | abstract |
| 5 | Lindsay, "Reuniting the LOCM family: An alternative method for identifying static relationships", ICAPS 2021 KEPS. [PDF](https://ai-icaps.dmi.unibas.ch/workshops/KEPS/Papers/KEPS_2021_paper_12.pdf) | static relationships | correct action sequences | **learned** via a Boolean validity test | none reported | offline | **PARCIAL** | abstract |
| 6 | Cresswell, McCluskey & West, "Acquiring planning domain models using LOCM", KER 28(2), 2013; Cresswell & Gregory, LOCM2, ICAPS 2011. [cited by 2402.10726](https://www.alphaxiv.org/abs/2402.10726) | object-centred FSM domain | action traces | conflicting: "including static predicates" (Balyo) vs lacking static restrictions (LOP) | heuristic, "scope not clear" | offline | DISTANTE | secondary |
| 7 | Bonet & Geffner, LSSS, ECAI 2020 (+ Rodriguez et al., KR 2021). [ECAI PDF](https://ecai2020.eu/papers/894_paper.pdf) | predicates, schemas, **initial state** | labelled state graphs, states unobserved | learned jointly with the domain (the number of statics is a hyperparameter) | SAT optimality; does not scale | offline | **PARCIAL** | secondary |
| 8 | Suárez-Hernández, Segovia-Aguas, Torras & Alenyà, "STRIPS Action Discovery", [arXiv 2001.11457](https://www.alphaxiv.org/abs/2001.11457) | action signatures + schemas | initial + goal states only | **given** in s0; learns static *preconditions* by removal | none (coverage) | offline | DISTANTE (inverse of D2) | full |
| 9 | Aineto, Jiménez & Onaindia, "Learning action models with minimal observability" (FAMA), AIJ 275:104–137, 2019. [PDF](https://riunet.upv.es/bitstream/10251/144560/7/Aineto%3bJim%c3%a9nez-Celorrio%3bOnaindia%20-%20Learning%20action%20models%20with%20minimal%20observability.pdf) | schemas | arbitrary missing states/actions; init/goal known | given | consistency via a planner | offline | DISTANTE | abstract |
| 10 | Amir & Chang, "Learning Partially Observable Deterministic Action Models" (SLAF), JAIR 33:349–402, 2008. [arXiv 1401.3437](https://arxiv.org/pdf/1401.3437) | transition relation + belief over states | actions observed, states partial | not distinguished; unknown state facts are filtered jointly | exact ("no model misclassified") | offline (filtering, incremental) | **PARCIAL** (joint filtering of unknown facts and model) | abstract |
| 11 | Yang, Wu & Jiang, "Learning action models from plan examples using weighted MAX-SAT" (ARMS), AIJ 171, 2007. [cited by 2402.10726](https://www.alphaxiv.org/abs/2402.10726) | schemas | init/final known, intermediate partial | given | optimisation, none | offline | DISTANTE | secondary |
| 12 | Zhuo & Kambhampati, AMAN, IJCAI 2013; Zhuo et al., ADMN, arXiv 1908.09800; Zhuo et al., LAMP, AIJ 174, 2010. [cited by 2402.10726](https://www.alphaxiv.org/abs/2402.10726) | schemas (noisy / ADL) | noisy, partial | given | none | offline | DISTANTE | secondary |
| 13 | Stern & Juba, IJCAI 2017; Juba, Le & Stern, "Safe learning of lifted action models", KR 2021 (SAM/ESAM). [arXiv 1705.08961](https://arxiv.org/pdf/1705.08961) | schemas | full | given (in states) | safety, linear sample bound | offline | DISTANTE | secondary |
| 14 | Le, Juba & Stern, "Learning Safe Action Models with Partial Observability" (PI-SAM/EPI-SAM), AAAI 38(18):20159–20167, 2024. [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/29995) | schemas | partial states | given or partially observed; no special treatment | safety, polynomial time | offline | DISTANTE | abstract |
| 15 | Lamanna, Serafini, Saetti, Gerevini & Traverso, "Lifted action models learning from partial traces" (OffLAM), AIJ 339:104256, 2025. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0004370224001929) | schemas | partial states **and** action arguments | not verified | "no guarantees" (per Jansen et al.) | offline | DISTANTE (provisional) | secondary |
| 16 | Lamanna, Saetti, Serafini, Gerevini & Traverso, "Online Learning of Action Models for PDDL Planning" (OLAM), IJCAI 2021, pp. 4112–4118. [IJCAI](https://www.ijcai.org/proceedings/2021/0566.pdf) | schemas | full states; acts and observes failures | given (in states) | consistency; not verified | **online** | DISTANTE (inverse; interactive) | snippet |
| 17 | Verma, Marpally & Srivastava, "Asking the Right Questions: Learning Interpretable Action Models Through Query Answering" (AIA), AAAI 35, 2021. [arXiv 1912.12613](https://www.alphaxiv.org/abs/1912.12613) | the agent's schemas | queries on chosen initial states | given (the interrogator sets s_I) | termination + functional equivalence (Thm 3), O(\|P*\|·\|A\|) queries | **online/active** | DISTANTE (mirror mechanism) | full |
| 18 | Aineto & Scala, "Action Model Learning with Guarantees" (VSLAM), KR 2024, pp. 801–811. [arXiv 2404.09631](https://www.alphaxiv.org/abs/2404.09631) | version space of schemas | full; positive and negative demonstrations | given | optimal sound and complete models, convergence | online update | **PARCIAL** (formal analogue of (Π⁺, Π⁻)) | full |
| 19 | Walsh & Littman, "Efficient Learning of Action Schemas and Web-Service Descriptions", AAAI 2008, pp. 714–719. [MLAnthology](https://mlanthology.org/aaai/2008/walsh2008aaai-efficient) | STRIPS / STRIPS+WS schemas | experience; teacher traces | given | learnability (negative/positive); KWIK link unverified | online | DISTANTE | abstract |
| 20 | Bolander & Gierasimczuk, "Learning Actions Models: Qualitative Approach", LORI 2015 / JLC 28(2), 2018. [PDF](https://people.compute.dtu.dk/tobo/bolander2018learning.pdf) | DEL action models | full | n/a (propositional) | finite identifiability / identification in the limit | online (streams) | **PARCIAL** (identifiability template) | prior review |
| 21 | Balyo, Suda, Chrpa, Šafránek, Gocht, Dvořák, Barták & Youngblood, "Planning Domain Model Acquisition from State Traces without Action Parameters", KR 2024, pp. 812–822. [arXiv 2402.10726](https://www.alphaxiv.org/abs/2402.10726) | schemas + parameters | full states, action names only | given (rigid predicates observed) | GI-hard, in NP | offline | DISTANTE | full |
| 22 | Bachor & Behnke, "Learning Planning Domains from Non-redundant Fully-Observed Traces", AAAI 38(18):20028–20035, 2024; Bachor, Dekker & Behnke, ICAPS 2025. [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/29980); [ICAPS](https://ojs.aaai.org/index.php/ICAPS/article/view/36096) | grounded domains | full / unobserved | n/a | complexity classes | offline | DISTANTE | abstract |
| 23 | Kikaj, De Smet, Marra & De Raedt, "Neurosymbolic Action Model Learning under Partial Observability" (NeSyAM), [arXiv 2609.25766](https://www.alphaxiv.org/abs/2609.25766), 2026 | probabilistic schemas from images | partial images + final state | given (in symbolic states); pre vs not-involved non-identifiable | variational analysis | offline | **PARCIAL** (identifiability argument only) | full |
| 24 | Xi, Gould & Thiébaux, "Neuro-symbolic learning of lifted action models from visual traces" (ROSAME-I), ICAPS 34:653–662, 2024; Xi et al., ICAPS 2026 ([arXiv 2604.19043](https://www.alphaxiv.org/abs/2604.19043)) | schemas from images | full images | given; bias towards preconditions | none formal | offline | DISTANTE | secondary |
| 25 | Asai & Fukunaga, AAAI 2018; Asai et al., "Classical Planning in Deep Latent Space", JAIR 74, 2022 (Latplan); Amado et al., IJCNN 2018 | propositional latent model | images | none (propositional, no lifted statics) [memory, unverified] | none | offline | DISTANTE | secondary |
| 26 | Mourão, Zettlemoyer, Petrick & Steedman, "Learning STRIPS operators from noisy and incomplete observations", UAI 2012 | schemas | noisy, partial | given | none | offline | DISTANTE | secondary |
| 27 | Wang, OBSERVER (CMU PhD 1996; ML 1995); Gil, EXPO (ICML 1994). [cited by 2402.10726](https://www.alphaxiv.org/abs/2402.10726) | schemas | full; acts, learns from failures | given | convergence of over/under approximations | online | DISTANTE | secondary |
| 28 | Lamanna et al., "Planning for Learning Object Properties", AAAI 37(10):12005–12013, 2023. [arXiv 2301.06054](https://arxiv.org/abs/2301.06054) | perceptual classifiers for PDDL properties | perception | n/a (properties, known domain) | none | online | DISTANTE | abstract |
| 29 | Lamanna et al., OGAMUS, [arXiv 2112.10007](https://www.alphaxiv.org/abs/2112.10007) (from the prior review) | instance objects and facts, known domain | egocentric perception | sensed, not abduced | none | online | **PARCIAL** | prior full text |
| 30 | "Learning Action Models with Conditional and Quantified Effects via Uncertainty-Guided Exploration", [arXiv 2608.30955](https://www.alphaxiv.org/abs/2608.30955), 2026 | schemas | exploration | not assessed | not assessed | online | DISTANTE (provisional) | title/abstract |
| 31 | Callanan et al., "MACQ: a unified library for action model acquisition", ICAPS Demos 2022. [arXiv 2206.06530](https://arxiv.org/abs/2206.06530) | toolkit | — | — | — | — | DISTANTE (infrastructure) | secondary |
| 32 | Arora, Fiorino, Pellier, Métivier & Pesty, "A review of learning planning action models", KER 33, 2018; Aineto, Jiménez & Onaindia, "A comprehensive framework for learning declarative action models", JAIR 74:1091–1123, 2022 (surveys) [cited by 2404.09631](https://www.alphaxiv.org/abs/2404.09631) | survey | — | — | — | — | context | secondary |

### Inferences
- **Verdict on D2 within the AML facet: the differential holds, with precise qualifiers.** The literature contains each of D2's ingredients in isolation:
  - static facts as instance properties (SIFT, SYNTH+);
  - static relations inferred from executions (LOP, Lindsay, SIFT);
  - joint filtering of unknown facts and model (SLAF);
  - interactive failure-driven inference (OBSERVER, OLAM, AIA);
  - (L, U) knowledge bounds with soundness and completeness (VSLAM);
  - identifiability theory (Bolander & Gierasimczuk, SIFT Thm 20).
- No work combines all of them with the roles reversed: templates given, static facts unknown, learned online from success and failure, with identifiability conditions.
- Recommended wording for the thesis: "dual of action-model learning: schemas fixed, static facts unknown; closest formal neighbours are SIFT's instance construction (Thm 12 and the p_a statics) and VSLAM's sound/complete boundaries, both of which are offline and schema-directed". Avoid claiming that "no one has learned static predicates": LOP, Lindsay and SIFT have.

### Gaps
- The verdicts on OffLAM, OLAM, LOP and Lindsay rely on abstracts or citing papers, and could move up to PARCIAL or SOBREPOSIÇÃO FORTE if their full texts show instance-level static inference.

---

## Q6. Limitations of this review

### Takeaway
This is a single-reviewer, tool-constrained review. Its negative conclusions ("no work does X") have moderate confidence.

### Cited Findings
- WebFetch to `ojs.aaai.org` and `ai.dmi.unibas.ch` failed with ENOTFOUND (session tool log, 2026-11-07). The OJS-hosted AAAI and ICAPS PDFs (LOP, PI-SAM, OLAM, Bachor & Behnke, Lamanna 2023) could therefore not be read in full.
- OffLAM is not indexed by alphaXiv (exact-title lookup failed), so its handling of static predicates is unverified.

### Inferences
- **Single screener**, so there is selection bias towards the planning-community vocabulary ("static predicate", "rigid", "invariant").
- **Search hit counts** are about 10 ranked results per query, so recall is unknown.
- **Forward snowballing** relied on alphaXiv semantic discovery rather than citation indexes.
- **Several items rely on secondary descriptions** (Balyo et al.'s and Jansen et al.'s related-work sections): ARMS, LOCM, AMAN/ADMN, LAMP, OBSERVER, EXPO, Latplan and ROSAME-I. Those papers' own wording on static predicates was not checked.
- **LOCM conflict.** The disagreement about whether LOCM learns static predicates is unresolved.

### Gaps
- Recommended follow-ups:
  1. Read OffLAM (AIJ 2025), LOP (ICAPS 2015) and Lindsay (KEPS 2021) in full once the hosts are reachable.
  2. Run Google Scholar "cited by" on SIFT and VSLAM, filtering for "static" and "instance".
  3. Search for "rigid fluents" and "initial state learning" in the KR/situation-calculus literature, which this facet did not cover.
