# RS2 — Identifiability and learnability of action models and hidden static structure (logic-based, abductive and ILP approaches), graded against thesis differential D2

Review date: 2026-11-07. Reviewer: one research agent working alone. This note extends `../Diferenciais da tese revisão sistemática/rs_fatos_estaticos.md` (2026-10-09). That review already logged Bolander & Gierasimczuk and SAM at snippet level, as well as OGAMUS, James et al. 2020, the K-replanner and SDR. Those entries are not repeated here except where this review read new full text or reached a different verdict.

**D2 claim under test.** "Identification of static facts given known templates, with identifiability guarantees, as the dual of Bolander & Gierasimczuk's learnability of action models in DEL (where the action model is unknown)." In the thesis, static facts (for example `connected(r1,r2)`) select the transition system. They are abduced by interaction: if `move(r1,r2)` produces an observable change, then `connected(r1,r2)` holds. Knowledge is tracked as (Π⁺, Π⁻) literal sets that observation rules update.

**Verdict scale.** IDÊNTICO; SOBREPOSIÇÃO FORTE; PARCIAL; DISTANTE.

**Read labels.**
- [full text]: I read the paper body or theorem statements in this session.
- [partial full text]: I read selected sections (abstract, intro, definitions and theorems) but not the whole paper.
- [AI report]: alphaXiv's structured report of the full paper, not the PDF itself.
- [abstract/snippet]: search abstract or snippet only.
- [memory, unverified]: background knowledge not confirmed with a source in this session.

---

## Q1. Protocol, search log, snowballing and PRISMA counts

### Takeaway
I fixed the protocol before searching and ran 17 search strings across 2 tools (WebSearch and alphaXiv), plus 4 full-text reads through alphaXiv. Snowballing started from 3 mandated seeds and 1 extra seed. In the end, 22 studies were included. people.compute.dtu.dk (the JLC 2018 PDF) and arxiv.org were unreachable from this session, so full texts came through alphaXiv.

### Cited Findings

**Protocol (written before the first search).**
- **RQ.** What identifiability and learnability results exist for symbolic action models, dynamics rules, or hidden static structure, learned from observation or interaction? This includes abductive and ILP methods. Does any work fix the lifted schemas and prove identifiability conditions for the unknown instance-level static facts?
- **Inclusion.**
  - IC1: 1995–2026.
  - IC2: Theory or method on identifiability or learnability (finite identification, identification in the limit, PAC, KWIK/mistake bound, safe/sound, or exact version-space or consistency characterisation) of one of the following: action models; dynamics rules; hidden static or instance structure.
  - IC3: Symbolic or logical representation.
  - IC4: English.
- **Exclusion.**
  - EC1: Purely statistical or continuous system identification with no symbolic structure.
  - EC2: Non-English.
  - EC3: Duplicates. A conference and journal version of the same work count as one study with two records.
  - EC4: Patents, aggregators or slides as the only source.
- **Tools.**
  - WebSearch (standard mode).
  - alphaXiv `discover_papers`, `get_paper_content` and `answer_pdf_queries` (full text).
  - WebFetch to people.compute.dtu.dk failed with ENOTFOUND (blocked by the session's network policy).
  - Consensus was not used: it failed in the previous review because of quota.

**Search log.** "Hits" is the number of records the tool returned. All hits were screened at title/snippet level.

| # | Exact string | Tool | Hits | Retained |
|---|---|---|---|---|
| S1 | `Bolander Gierasimczuk "Learning to act" qualitative learning deterministic action models Journal of Logic and Computation` | WebSearch | 10 | 2 (JLC 2018; LORI 2015) |
| S2 | `Gierasimczuk learnability dynamic epistemic logic action models identification in the limit follow-up` | WebSearch | 10 | 1 new (Gierasimczuk 2009 "Bridging learning theory and DEL") |
| S3 | `Walsh Littman "Efficient learning of action schemas and web-service descriptions" KWIK` | WebSearch | 10 | 1 |
| S4 | `Juba Stern safe action model learning partial observability sample complexity` | WebSearch | 10 | 1 (Le, Juba & Stern AAAI 2024) |
| S5 | `Evans "Making sense of sensory input" apperception engine static facts objects rules Artificial Intelligence 2021` | WebSearch | 10 | 1 |
| S6 | `learning from interpretation transitions Inoue Ribeiro Sakama 2014 Machine Learning LFIT state transitions` | WebSearch | 10 | 1 |
| S7 | `Mourão Zettlemoyer Petrick Steedman learning STRIPS operators noisy incomplete observations UAI 2012` | WebSearch | 10 | 1 |
| S8 | `Bolander Gierasimczuk Liberman "Learning to Act and Observe in Partially Observable Domains"` | WebSearch | 10 | 1 |
| S9 | `Amir Chang "Learning partially observable deterministic action models" SLAF JAIR 2008 initial state` | WebSearch | 10 | 1 |
| S10 | `Juba Le Stern "Safe learning of lifted action models" KR 2021 static predicates` | WebSearch | 10 | 1 |
| S11 | `Ray XHAIL abductive inductive learning event calculus Moyle Muggleton learning narratives` | WebSearch | 10 | 3 (XHAIL; Moyle & Muggleton 1997; Katzouris et al. 2015) |
| S12 | `Law Russo Broda ILASP learning from noisy answer sets inductive learning of answer set programs partial interpretations` | WebSearch | 10 | 2 (ILASP 2014; noisy 2018) |
| S13 | `Rodriguez Bonet Romero Geffner learning first-order representations planning black-box states static predicates` | WebSearch | 10 | 1 |
| S14 | question "Identifiability or learnability guarantees for learning action models, transition rules, or hidden static facts (unknown instance relations) of a planning domain from observed transitions or interaction, including abductive and ILP approaches"; keywords [action model learning, identifiability, static predicates, abduction]; prioritize=historical | alphaXiv discover_papers | 14 | 4 (2411.14995 SIFT; 2508.21449; 2605.18627; 2403.15251). 2410.17882 excluded by EC1 |
| S15 | `Stern Juba 2017 "Efficient, Safe, and Probably Approximately Complete Learning of Action Models" theorem sample bound` | WebSearch | 10 | 1 |
| S16 | `identifying unknown initial state facts known domain model observations diagnosis planning identifiability "initial state" learning action traces` | WebSearch | 10 | 2 (Krivic et al. initial-state prediction; Sharma et al. domain concretization). 1 patent excluded by EC4 |
| S17 | `Evans Bošnjak Buesing Ellis Pfau Kohli Sergot "Making sense of raw input" Artificial Intelligence 2021` | WebSearch | 10 | 1 |

**Snowballing (seeds).**
- **Seed A: Bolander & Gierasimczuk, LORI 2015** ([alphaXiv 1507.04285](https://www.alphaxiv.org/abs/1507.04285), full text).
  - Backward: its ref [18] is a qualitative STRIPS learner (I take it to be Walsh & Littman 2008; the reference number was seen but its list entry was not opened). Its ref [13] covers abstract properties of finite identifiability (I take it to be Gierasimczuk & de Jongh; memory, unverified).
  - Forward: Bolander, Gierasimczuk & Occhipinti Liberman 2021, via S8.
- **Seed B: Stern & Juba, IJCAI 2017.**
  - Forward: Juba, Le & Stern KR 2021; Le, Juba & Stern AAAI 2024.
  - Conditional-effect SAM ([2403.15251](https://www.alphaxiv.org/abs/2403.15251)) and numeric SAM ([2312.10705](https://www.alphaxiv.org/abs/2312.10705)), both via S14.
- **Seed C: Evans et al., Apperception Engine, AIJ 2021.**
  - Backward, through its related-work section: LFIT (Inoue et al.), XHAIL (Ray), ILASP (compared experimentally), Pasula et al.
  - Forward: "Making sense of raw input".
- **Extra seed D: Gösgens, Jansen & Geffner, SIFT, AAAI 2025** ([2411.14995](https://www.alphaxiv.org/abs/2411.14995)).
  - Backward: Bonet & Geffner ECAI 2020; Rodriguez et al. KR 2021; Lindsay 2021 (LOCM static relationships).

**PRISMA-style counts.**
- Identified: 174 database records (16 × 10 + 14), plus about 10 through snowballing, giving about 184.
- After de-duplication (the same paper across arXiv, publisher, aggregator and conference/journal copies): about 70 unique records.
- Screened on title/abstract: 70.
- Excluded at screening: 44. These were aggregator or duplicate copies, patents (EC4), statistical identification (EC1), and work off-topic for IC2.
- Assessed for eligibility: 26.
- Excluded after assessment: 4. These were 2410.17882 (statistical, EC1), 2601.18930 (POMDP parameter learning, no symbolic instance structure), the patent, and Sharma et al. (no learnability guarantee and no unknown-static-fact component beyond the snippet).
- Included: 22 studies (Q2–Q5 below).
- Full text or partial full text read: 6. These were B&G 2015, B&G&L 2021, Apperception (AI report), Rodriguez et al. 2021, SIFT 2025, and the prior review's James et al.

### Inferences
- The counts are approximate where search engines merged duplicates. They should be reported as "≈" in the thesis.

### Gaps
- DBLP, Semantic Scholar and OpenAlex citation APIs were not used (blocked in the previous review). Forward snowballing from B&G 2015 is therefore based on search, not citation indexes. Other DEL-learning follow-ups (2018–2026) may exist that I did not find.

---

## Q2. Bolander & Gierasimczuk line (DEL learnability): exact claims and follow-ups

### Takeaway
B&G prove finite identifiability for **fully observable, deterministic, propositional** action models from **sound and complete streams** of (s, s′) pairs, and only identifiability in the limit for non-deterministic models. Their learners work by **eliminating events from an initial maximal hypothesis model**, which mirrors (Π⁺, Π⁻) elimination. The 2021 follow-up with Occhipinti Liberman moves to partial observability and proves learnability **up to behavioural equivalence (bisimilarity)**. That notion is exactly the identifiability criterion D2 needs, but there the dynamics are unknown and the schemas are not lifted.

### Cited Findings

**LORI 2015** [full text] — [alphaXiv 1507.04285](https://www.alphaxiv.org/abs/1507.04285).
- *Observation model.* A stream is an infinite sequence of pairs (s, s′) of propositional states. It is assumed sound (each s′ ∈ s ⊗ A) and complete (every s′ ∈ s ⊗ A for every s ⊆ P eventually appears) (Defs. 5–6).
- *Theorem 1.* "For any finite set of propositions P the set of (fully observable) deterministic propositional actions over P is finitely identifiable." The proof builds a definite finite tell-tale set (DFTT) D(A) = {(s, s′) | s ⊗ A = s′}.
- *Theorem 2.* "Arbitrary (including non-deterministic) fully observable propositional actions … [are] not finitely identifiable." The example is a fake coin that always lands tails.
- *Theorem 3.* The same class "is identifiable in the limit", via finite tell-tale sets.
- *Theorems 4–7.* Update learners (A | S restricts the action model to the events consistent with all observations) finitely identify the following classes:
  - precondition-free atomic actions;
  - deterministic actions with maximal preconditions;
  - all deterministic actions (L₃^update, which waits until every s has been observed);
  - deterministic action libraries (Thm. 7).
- *The authors' own limits.* "We only considered fully observable actions applied in fully observable states". Partial observability was left for future work ("learning 'as much as possible'").
- *Comparison with STRIPS.* The paper presents its framework as richer than an earlier qualitative STRIPS learner (its ref [18]) in negative preconditions and effects and in conditional effects.

**JLC 2018 journal version** [abstract/snippet] — Bolander & Gierasimczuk, "Learning to act: qualitative learning of deterministic action models", *Journal of Logic and Computation* 28(2):337–365, 2018, [doi:10.1093/logcom/exx036](https://dblp.uni-trier.de/rec/journals/logcom/BolanderG18.html).
- The search summary says it studies "fully observable, universally applicable action models". Deterministic models are "conclusively learnable (finitely identifiable)", while arbitrary ones are not but are identifiable in the limit. It gives learning by update for unconditional and conditional deterministic actions.
- Source: [arXiv/DTU listing](https://www.arxiv.org/abs/1507.04285). The DTU PDF (people.compute.dtu.dk) was unreachable in this session.

**Follow-up: Bolander, Gierasimczuk & Occhipinti Liberman, "Learning to Act and Observe in Partially Observable Domains"** [partial full text: abstract, intro and theorem statements] — [arXiv 2109.06076](https://www.alphaxiv.org/abs/2109.06076). Also a chapter in *Dick de Jongh on Intuitionistic and Provability Logics*, Springer 2024, [doi:10.1007/978-3-031-47921-2_11](https://iit.comillas.edu/publicacion/libro/en/398/Dick%20de%20Jongh%20on%20Intuitionistic%20and%20Provability%20Logics).
- *Setting.* A domain is D = ((S, A, T, s₀), Ω, Obs), a transition system with an initial state and an observation function. The agent learns both "what it can observe and how its actions affect the environment".
- *Theorem 1.* LEARNER, given a sound and complete set of observed transitions, outputs event models that are "behaviourally correct with respect to explicit knowledge".
- *Implicit knowledge.* This is defined "as knowledge up to behavioural equivalence … as much knowledge as a learner can possibly acquire … in two behaviourally equivalent domains, any such experiment yields the same observations."
- *Propositions 1–3.* Every domain over P has a DFTT, and one exists whose observation traces contain 2^{2|P|} actions (Prop. 2). With a DFTT, a domain "behaviourally correct with respect to implicit knowledge" can be learned in finite time (Prop. 3).
- *Theorem 2.* The learner returns the synchronous composition of all domains bisimilar to D.
- *Exploration caveat.* The authors note it is "less clear" how the agent explores if it cannot recognise when it has returned to the initial state.

**Background: Gierasimczuk 2009, "Bridging learning theory and dynamic epistemic logic"** (Synthese) [snippet]. It models finite identification inside DEL — [search result](https://www.arxiv.org/abs/1507.04285).

### Inferences
- **The "dual" framing holds formally and should cite the 2021 paper, not only 2015.**
  - In B&G, the hypothesis space is a set of events (pre, post) per action. The learner removes events that contradict observations.
  - In D2, the templates are fixed and the hypothesis space is the set of valuations of ground static atoms. The learner removes valuations (equivalently, grows Π⁺ and Π⁻).
  - Both are finite hypothesis classes with sound data, so the same Angluin/Mukouchi-style DFTT characterisation of finite identifiability applies.
  - B&G&L 2021 already contains the right criterion for D2. Two valuations of the static facts are indistinguishable exactly when the induced domains are behaviourally equivalent under the observation model, so a static fact is identifiable iff flipping it changes the induced domain's behavioural-equivalence class. This is my inference, not a theorem in either paper.
- **Assumptions that do not carry over to D2.**
  - B&G 2015 assumes complete streams over all s ⊆ P. An interacting agent only sees reachable states, so an unreachable `connected(r7,r8)` is never identified.
  - B&G&L 2021 uses traces from s₀, which is closer to D2's setting.
  - Both papers are propositional. Neither gives lifted templates or per-object static atoms, and neither separates static atoms from fluents.
- **Verdicts.**
  - B&G 2015/2018: **PARCIAL**. It is the formal dual (same learning-theoretic criteria, same update/elimination mechanism), but the unknown object, the passive complete-stream observation model, and the propositional representation all differ.
  - B&G&L 2021: **PARCIAL**, and the strongest formal neighbour. Its identifiability up to behavioural equivalence, under partial observability and from s₀, covers in principle an unknown s₀ that encodes static facts. However, the dynamics are unknown too, there are no lifted templates, and there is no static/fluent split.

### Gaps
- I could not open the JLC 2018 full text (DTU host blocked), so I have not verified its theorem numbering or any extra classes beyond the abstract.
- I found no 2022–2026 DEL-learnability paper by Gierasimczuk or Bolander beyond the 2024 book chapter. This may be a coverage limitation (no citation index).

---

## Q3. PAC, KWIK and safe learnability of STRIPS/PDDL action schemas

### Takeaway
This line treats the **schemas as unknown** and the instance (states) as **observed**. Its guarantees are sample complexity or safety: KWIK/PAC-style for Walsh & Littman, safe and probably approximately complete for SAM. Static predicates matter only as precondition literals of the schemas. SAM's core inference ("an action was executed, so its preconditions held in the pre-state") is the same abductive direction as D2, but it is applied to schema literals, not to unknown ground static facts.

### Cited Findings
- **Walsh & Littman, AAAI 2008**, "Efficient Learning of Action Schemas and Web-Service Descriptions", pp. 714–719, doi:10.7282/t3q81hg7 [abstract/snippet] — [mlanthology](https://mlanthology.org/aaai/2008/walsh2008aaai-efficient); [Rutgers record](https://rucore.libraries.rutgers.edu/rutgers-lib/30476/record/).
  - General STRIPS operators "cannot be efficiently learned through raw experience". Restricting precondition size gives a positive result, and so does a teacher that supplies solution traces.
  - Walsh's dissertation frames the subproblems in KWIK.
  - Verdict on D2: **DISTANTE** (schemas unknown, states observed).
- **Stern & Juba, IJCAI 2017**, "Efficient, Safe, and Probably Approximately Complete Learning of Action Models" (SAM), doi:10.24963/ijcai.2017/615 [abstract/snippet] — [IJCAI](https://www.ijcai.org/proceedings/2017/615); [arXiv 1705.08961](https://arxiv.org/pdf/1705.08961v1.pdf).
  - Learns a conservative model in which actions are guaranteed applicable. The probability that it fails on a solvable problem "decreases quasi-linearly with the number of observed trajectories" under stated assumptions.
  - Verdict: **DISTANTE** on the problem, with a shared inference direction (see Inferences).
- **Juba, Le & Stern, KR 2021**, "Safe Learning of Lifted Action Models", pp. 379–389, doi:10.24963/kr.2021/36 [abstract/snippet] — [KR proceedings](https://proceedings.kr.org/2021/36/); [arXiv 2107.04169](https://arxiv.org/abs/2107.04169).
  - The first safe model-free planning algorithm for lifted domains. It learned the real action model in all 12 IPC domains with at most two trajectories.
  - Verdict: **DISTANTE**.
- **Le, Juba & Stern, AAAI 2024**, "Learning Safe Action Models with Partial Observability" (PI-SAM, EPI-SAM) [abstract/snippet] — [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/29995); [NSF PAR](https://par.nsf.gov/servlets/purl/10544585).
  - Safe under partial observability.
  - Sample complexity for probabilistic completeness under a *bounded concealment* assumption.
  - States that efficient learning of a safe model is not possible in general under partial observability.
  - Verdict: **DISTANTE**. The impossibility result is a useful contrast for D2's observation-model assumptions.
- **Mordoch et al., SAM with conditional effects** ([2403.15251](https://www.alphaxiv.org/abs/2403.15251)) and **numeric SAM** ([2312.10705](https://www.alphaxiv.org/abs/2312.10705)) [abstract/title]. Both extend safety guarantees to richer schemas. Verdict: **DISTANTE**.
- **Mourão, Zettlemoyer, Petrick & Steedman, UAI 2012**, "Learning STRIPS Operators from Noisy and Incomplete Observations" [abstract] — [PMLR](https://proceedings.mlr.press/r10/mourao12a.html); [arXiv 1210.4889](https://arxiv.org/abs/1210.4889).
  - Learns classifiers for the transition function, then extracts STRIPS rules. Empirical only, with no identifiability guarantee.
  - Verdict: **DISTANTE**.
- **Amir & Chang, JAIR 33:349–402, 2008**, "Learning Partially Observable Deterministic Action Models" (SLAF) [abstract] — [doi:10.1613/JAIR.2575](https://www.doi.org/10.1613/JAIR.2575); [arXiv 1401.3437](https://arxiv.org/pdf/1401.3437).
  - Gives exact learning. The output is a "transition belief formula" over the possible transition relations *and states*, computed in polynomial time for STRIPS-like classes.
  - Verdict: **PARCIAL**. With the model fixed, a joint belief over model and state reduces to filtering over unknown state facts, but it has no static/fluent split and no identifiability theorem (from memory; the abstract does not settle this).

### Inferences
- **SAM's rule matches D2's abduction rule.** SAM's rule is: "if a(o) is observed to execute from s, every literal of a's precondition, grounded by o, holds in s; keep only literals that held in every such s". This is the same monotone abductive rule as D2's: "move(r1,r2) executed ⇒ connected(r1,r2) ∈ Π⁺".
- **But the unknowns are on opposite sides.**
  - SAM: states are observed and schema literals are unknown, so it shrinks a precondition set.
  - D2: schemas are known and ground static atoms are unknown, so it grows Π⁺ and Π⁻.
  - This is a genuine duality and can be stated as such. The safety notion transfers directly: plan only with actions whose static preconditions are in Π⁺.
- **Partial observability is the critical observation-model assumption.** The negative result under partial observability (Le et al. 2024) suggests that D2's identifiability guarantees must state their observation model explicitly: which fluents are observable, and whether failure or no-op is observable.

### Gaps
- I did not read the exact SAM sample bounds (theorem statements not retrieved).
- I did not verify Walsh & Littman's precise hypothesis class and bound.

---

## Q4. Abduction and ILP for dynamics (LFIT, XHAIL, Event Calculus, ILASP, Apperception Engine)

### Takeaway
Of the ILP systems, only the Apperception Engine and XHAIL **abduce ground facts**, and they do it jointly with induced rules. The Apperception Engine abduces initial conditions, including latent objects and relations, from partial sequences. None gives identifiability guarantees: they use parsimony (minimum cost) instead. LFIT and ILASP learn rules from fully specified or partial interpretations without latent ground facts. ILASP's partial-interpretation pair ⟨inclusions, exclusions⟩ is a notational cousin of (Π⁺, Π⁻).

### Cited Findings
- **Evans, Hernández-Orallo, Welbl, Kohli & Sergot, "Making sense of sensory input"**, *Artificial Intelligence* 293:103438, 2021 [AI report of full text] — [alphaXiv 1910.02227](https://www.alphaxiv.org/abs/1910.02227); [arXiv](https://arxiv.org/abs/1910.02227).
  - *Theory.* θ = (φ, I, R, C): type signature, initial conditions (ground atoms, "including latent information"), static and causal rules in Datalog⊃−, and constraints.
  - *Search.* A joint process of "abduction (for initial conditions I) and induction (for rules R and constraints C)" in ASP, picking the lowest-cost theory that covers the sequence and satisfies the unity conditions. Missing readings are allowed.
  - *Complexity.* Finding a unified interpretation for a template is in NP; finding the lowest cost is in Σ₂ᴾ.
  - *Results.* Learns spatial relations in the ECA tasks "without explicit prior encoding". Infers persistent latent objects in occlusion tasks.
  - *Guarantee.* None regarding identifiability. Selection is by minimum cost.
  - Verdict on D2: **PARCIAL** (the closest ILP analogue: it abduces persistent relational facts that select the dynamics). But the rules are also unknown, the data is passive, there is no interaction or identifiability, and it does not separate static facts from fluents in the D2 sense.
- **Evans, Bošnjak, Buesing, Ellis, Pfau, Kohli & Sergot, "Making sense of raw input"**, *Artificial Intelligence* 299:103521, 2021, doi:10.1016/j.artint.2021.103521 [abstract] — [IJCAI 2022 extended abstract](https://ijcai.org/proceedings/2022/799); [PhilArchive](https://philarchive.org/archive/EVAMSO-3).
  - A neuro-symbolic extension to raw input. The network and the theory are solved jointly as one SAT problem.
  - Verdict: **DISTANTE**.
- **Inoue, Ribeiro & Sakama, "Learning from interpretation transition" (LFIT)**, *Machine Learning* 94(1):51–79, 2014 [abstract] — [Springer PDF](https://link.springer.com/content/pdf/10.1007/s10994-013-5353-8.pdf); [Sakama abstract](https://web.wakayama-u.ac.jp/~sakama/abst/mlj2013.html).
  - Learns normal logic programs as state-transition functions from pairs of interpretations (Boolean networks, cellular automata).
  - Per Evans et al., LFIT does not posit latent objects or predicates and does not generate initial conditions.
  - Verdict: **DISTANTE**.
- **Ray, "Nonmonotonic abductive inductive learning" (XHAIL)**, *Journal of Applied Logic* 7(3), 2009 [abstract/snippet] — [Bristol record](https://research-information.bris.ac.uk/en/publications/nonmonotonic-abductive-inductive-learning/).
  - Abduces ground hypotheses, then generalises them inductively, with the Event Calculus as background.
  - Verdict: **PARCIAL** (mechanism only). With a fixed theory, its abductive phase is "abduce ground facts that explain the narrative", but there is no identifiability analysis and no interaction.
- **Moyle & Muggleton, "Learning programs in the event calculus"** (ILP 1997) [snippet] — [record](https://katalog.lib.cas.cz/KNAV/EdsRecord/edsair,edsair.doi...........69bbf0fbd1d5953efb1deaf8a761f8ea).
  - Learns EC axioms from temporal observations by theory completion.
  - Verdict: **DISTANTE**.
- **Katzouris, Artikis & Paliouras, "Incremental learning of event definitions with ILP"**, *Machine Learning* 2015 [snippet] — [Springer](https://link.springer.com/10.1007/s10994-015-5512-1); [arXiv 1402.5988](https://arxiv.org/pdf/1402.5988).
  - Abductive–inductive clause refinement built on XHAIL.
  - Verdict: **DISTANTE**.
- **Law, Russo & Broda, "Inductive Learning of Answer Set Programs"**, JELIA 2014; "… from Noisy Examples", 2018 [abstract] — [ILP2014 PDF](https://www.doc.ic.ac.uk/~ml1909/ILASP_ILP2014.pdf); [arXiv 1808.08441](https://arxiv.org/pdf/1808.08441v1.pdf).
  - Examples are partial interpretations ⟨inclusions, exclusions⟩. The 2018 version tolerates noise.
  - Evans et al. report that ILASP's grounding blew up on ECA tasks.
  - Verdict: **DISTANTE**. The pair structure is analogous to (Π⁺, Π⁻), but here it describes examples, not beliefs.

### Inferences
- **The ILP route gives D2 a mechanism precedent, not a guarantee precedent.**
  - "Abduce ground facts that, together with known rules, explain the observations" is standard in XHAIL and the Apperception Engine. That is the D2 inference when R is fixed.
  - The ILP systems choose among explanations by cost, not by eliminating them through interaction, and they prove no identifiability.
  - D2 can therefore claim the combination "abduction + identifiability conditions + active interaction", not abduction itself.

### Gaps
- I did not verify in the PDF whether the Apperception Engine's ECA neighbour relation is placed in I (abduced) or in the given background. The AI report says the spatial relationships are learned "without explicit prior encoding". Mark this as "per report".
- I found no identifiability theorem for LFIT with latent variables. This may be a gap in coverage of Ribeiro's later work (for example GULA and delayed systems), which I did not search.

---

## Q5. Does any work fix the schemas, treat instance static facts as unknown, and give identifiability conditions?

### Takeaway
**Yes, partially, and closer than the previous review found.**
- Bonet & Geffner / Rodriguez et al. explicitly define a **"verification problem"**: the domain D is fixed and only the instance information (objects and initial atoms, including static atoms) is inferred so that the induced state graph matches the observed one. The guarantee is exact encoding correctness (answer sets ↔ instances accounting for G), i.e. identification up to graph isomorphism. It is not a per-fact identifiability analysis, and the input is a complete black-box state graph, not interaction.
- SIFT (Gösgens, Jansen & Geffner 2025) handles static predicates by the rule "p_a(o) true iff a(o) appears in a trace", which is the D2 abduction stated as a definition. It explicitly leaves the compact form of static relations as an open extension.
- Krivic et al. predict unknown initial-state relations under a known domain statistically.

**No study states per-static-fact identifiability conditions under an interaction observation model.** That remains D2's defensible core.

### Cited Findings
- **Rodriguez, Bonet, Romero & Geffner, "Learning First-Order Representations for Planning from Black-Box States: New Results"**, KR 2021, pp. 539–548, doi:10.24963/kr.2021/51 [partial full text, pp. 1–7] — [alphaXiv 2105.10830](https://www.alphaxiv.org/abs/2105.10830); [KR](https://proceedings.kr.org/2021/51/).
  - *Def. 1.* Learn the simplest P_i = ⟨D, I_i⟩ such that the state graphs G(P_i) are isomorphic to the input labelled graphs G_i.
  - *Def. 2.* "The verification problem is a subproblem of the learning problem where the domain D is not learned but fixed, and just the instance information needs to be inferred."
  - *Encoding of static atoms.* Static atoms are encoded as `val((P,OO),V)` with no state index: "static predicates … appear in the precondition of action schemas but do not appear in the effects".
  - *Theorem 3.* M is an answer set iff a planning instance that accounts for G exists within the bounds. This is exact characterisation, not uniqueness.
  - Learned gripper and Hanoi schemas carry separate "Static:" preconditions, for example `-BIGGER(d,to)`.
  - Builds on Bonet & Geffner, ECAI 2020.
  - Verdict on D2: **SOBREPOSIÇÃO FORTE on the problem formulation** (Def. 2 is "templates known, instance static facts unknown"). **PARCIAL on guarantees and observation model**: a complete noise-free graph is assumed, the input states are black boxes rather than observed fluents, identification is only up to isomorphism plus Occam ranking, and there is no interaction or (Π⁺, Π⁻) epistemic tracking.
- **Gösgens, Jansen & Geffner, "Learning Lifted STRIPS Models from Action Traces Alone: A Simple, General, and Scalable Solution" (SIFT)**, AAAI 2025 [full text of the relevant sections] — [alphaXiv 2411.14995](https://www.alphaxiv.org/abs/2411.14995).
  - *Static predicates.* "It suffices to introduce a static predicate p_a for each lifted action a … atoms p_a(o) are set to true in I_T iff the ground action a(o) appears in a trace in T."
  - *Lower-arity statics.* "Obtaining a compact representation of the static p_a relations is not necessary, we will not address the problem." The discussion lists "the derivation of static predicates of lower arity" as an unaddressed extension.
  - *Guarantees.* Theorem 15 gives soundness and completeness for domains without static predicates, given complete traces. Theorem 20: "There is a finite set of extended traces T such that f is consistent with T iff f is valid. The learned domain D_T is then equivalent to the hidden domain D." The authors add: "Proving that a finite set of traces has this property … may even be undecidable in general."
  - Features "do not see" static predicates. Grid cannot be learned from plain traces alone (it needs extended traces with state equalities), which is an explicit non-identifiability example.
  - Verdict: **PARCIAL**. It has the same positive abduction rule and a finite-identifiability-style theorem (Thm. 20), but for schemas. Static facts get only positive, action-indexed evidence, with no negative evidence (Π⁻) and no conditions.
- **Krivic, Cashmore, Magazzeni, Szedmak & Piater, "Initial State Prediction in Planning"** (AAAI 2017 workshop / strathprints) and the JAIR 69:765–806 (2020) follow-up with active learning [abstract/snippet] — [Strathprints](https://strathprints.strath.ac.uk/69980/); [JAIR 2020 PDF](https://jair.org/index.php/jair/article/download/11567/26621).
  - Under a known deterministic domain, it predicts the unknown edges of a partially known initial-state multigraph (relations) with Maximum Margin Multi-Valued Regression. The JAIR version actively selects what to sense.
  - Verdict: **PARCIAL** (known domain, unknown relational initial facts, active selection), but statistical and sensing-based, with no abduction from action outcomes and no identifiability.
- **Lindsay, "Reuniting the LOCM family: An alternative method for identifying static relationships"**, ICAPS 2021 KEPS Workshop [reference only, cited in SIFT's related work] — [alphaXiv 2411.14995](https://www.alphaxiv.org/abs/2411.14995). Verdict: **PARCIAL** (provisional, not read).
- **2508.21449 "Learning Lifted Action Models From Traces of Incomplete Actions and States"** and **2605.18627 "… with Minimal Information About Actions and States"** [title/snippet only] — [2508.21449](https://www.alphaxiv.org/abs/2508.21449); [2605.18627](https://www.alphaxiv.org/abs/2605.18627). Geffner-group follow-ups to SIFT. Verdict: **DISTANTE/PARCIAL, unassessed**.
- **System-identification analogue (brief).** alphaXiv returned "Identifiable Representation and Model Learning for Latent Dynamic Systems" ([2410.17882](https://www.alphaxiv.org/abs/2410.17882)), which covers statistical identifiability of latent dynamics. It was excluded by EC1.
  - The analogy holds. In classical structural identifiability, a parameter θ is identifiable iff θ ≠ θ′ implies different input–output behaviour. That is the continuous counterpart of "static valuation identifiable iff the induced domains are not behaviourally equivalent" [memory, unverified as a citation].
  - The prior review's Kwon et al. (latent MDPs, [2102.04939](https://arxiv.org/pdf/2102.04939)) is the RL counterpart (latent context drawn once per episode).

### Inferences
- **Recommended restatement of D2** (my synthesis): "Given known lifted templates and an explicit interaction observation model, static facts are identified by monotone abduction into (Π⁺, Π⁻). A static atom is finitely identifiable iff [condition], i.e. iff instances differing on it are not behaviourally equivalent from s₀ under the observation model (cf. Bolander, Gierasimczuk & Occhipinti Liberman 2021). This is the dual of Bolander & Gierasimczuk (2015, 2018) and complementary to Rodriguez et al.'s (2021) verification problem, which infers instance information under a fixed domain from complete state graphs but without per-fact identifiability or interaction."
- **Concrete identifiability conditions D2 should prove.** These are my inferences and are not in the literature found.
  1. *Positive.* σ = s(c̄) enters Π⁺ once an action whose precondition grounds to σ succeeds.
  2. *Negative.* σ enters Π⁻ only when an attempted action fails observably and all its other preconditions are in the known state or in Π⁺. Otherwise only a disjunction is learned. This is where (Π⁺, Π⁻) literal sets become incomplete and a clause store would be needed.
  3. *Unidentifiable cases.*
     - σ is relevant only to actions never enabled from reachable states.
     - Two static valuations induce bisimilar domains (symmetry; for example, unordered rooms with no distinguishing fluents).
     - Failures are silent or unobservable.
  4. *Finite identification.* It is finite when the reachable part induces a DFTT. The bound is analogous to Prop. 2 of B&G&L (length 2^{2|P|}) but should shrink because only static atoms are unknown.
- **Overall grading.** No study is IDÊNTICO.
  - SOBREPOSIÇÃO FORTE on the problem statement: Rodriguez et al. 2021 (verification problem).
  - PARCIAL, closest formal frame: B&G&L 2021 (identifiability up to behavioural equivalence).
  - PARCIAL, closest abductive rule: SIFT 2025 (static p_a from action occurrence).
  - PARCIAL, closest ILP precedent: the Apperception Engine (abduces initial latent relations with rules).
  - So the "dual of B&G" phrase is accurate but should not be the only positioning. The thesis must cite Rodriguez et al.'s verification problem and SIFT's static-predicate treatment, or a referee will.

### Gaps
- **Limitations of this review.**
  1. Full text was read for 6 studies. Walsh & Littman, SAM 2017/2021/2024, SLAF, LFIT, XHAIL and ILASP were graded on abstracts or snippets.
  2. There was no citation-index forward snowballing.
  3. JLC 2018 and DTU hosts were unreachable.
  4. The Apperception Engine was read through an AI-generated report, not the PDF.
  5. I did not open Bonet & Geffner ECAI 2020, Lindsay 2021, 2508.21449 or 2605.18627.
  6. The search strings were English-only and focused on planning and KR. Model-checking or automata-learning work on "learning parameters of a known template" (for example, L*-style active learning of parameterised systems, or conformance testing) was not searched. It may contain identifiability results closer to D2, and is a recommended next facet.
  7. Counts are approximate (≈) because search engines merged duplicates.
