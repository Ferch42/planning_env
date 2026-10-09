# Learning unknown static facts ("meta-structure") of a planning instance under known action schemas: prior work and overlap with thesis claims (a), (b), (c)

Scope note: about 17 tool calls (web search, alphaXiv discovery). Items marked **[UNVERIFIED]** come from background knowledge. I did not confirm them by fetching a source in this session, so check them before citing.

Thesis claims under assessment:
- (a) Separate uncertainty over static facts (the meta-structure, which is monotonic once learned) from uncertainty over fluents.
- (b) Derive update rules for static facts automatically from action templates, by abducing preconditions from observed abstract transitions.
- (c) Treat static facts as the symbolic image of environment hyperparameters θ (as in procedurally generated environments): relearn them per environment while the templates transfer.

---

## Q1. Planning and acting with unknown static facts or unknown maps when the schemas are known

### Takeaway
This is a well-established area. Three lines of work cover it: navigation in unknown terrain with optimistic or free-space assumptions; the Canadian Traveller Problem (CTP); and contingent or partially observable classical replanning (K-replanner, SDR). All of them treat the map or connectivity as unknown, static state that the agent discovers while it executes, with the action model fixed. None of them names "static facts" as a separate epistemic layer, and none abduces those facts from transitions. They learn the facts by **direct sensing** (an observed blocked edge, an observed obstacle, a sensing action), not by explaining which action instance must have fired.

### Cited Findings
- **Koenig & Smirnov (ICRA 1997), "Sensor-Based Planning with the Freespace Assumption."** The planner treats unexplored terrain as passable and replans as sensors reveal obstacles. It has good guarantees on grids but is not worst-case optimal in general, and the paper proposes Basic-VECA as an alternative. Proc. ICRA vol. 4, pp. 3540–3545. — [CMU RI record](https://www.ri.cmu.edu/publications/sensor-based-planning-with-the-freespace-assumption)
- Koenig's later analysis of the freespace assumption, Greedy Mapping and Greedy Localization was done with C. Tovey and Y. Smirnov. — [Technion seminar abstract](https://www.cs.technion.ac.il/events/2008/382/); [Koenig greedy-online tutorial](https://idm-lab.org/greedyonline-tutorial.html)
- **Nourbakhsh & Genesereth (1996), "Assumptive Planning and Execution: a Simple, Working Robot Architecture,"** *Autonomous Robots* 3(1):49–67. The robot interleaves planning and execution under simplifying assumptions about incompletely known environments. The paper gives conditions for soundness and completeness and was used in the robot Dervish, winner of the 1994 AAAI robot competition. — [CMU RI record](https://ri.cmu.edu/publications/assumptive-planning-and-execution-a-simple-working-robot-architecture)
- Nourbakhsh & Genesereth also wrote "Time-Saving Tips for Problem-Solving with Incomplete Information" (AAAI 1993). — [CMU RI](https://www.ri.cmu.edu/publications/time-saving-tips-for-problem-solving-with-incomplete-information)
- **Papadimitriou & Yannakakis (1991), "Shortest paths without a map,"** *TCS* 84:127–150. The traveller learns that an edge is blocked only on reaching one of its endpoints. Guaranteeing a given competitive ratio is PSPACE-complete, and the stochastic variants are #P-hard. — [PDF (UCSB course mirror)](https://sites.cs.ucsb.edu/~suri/cs235/Rlist/noMapNavigation.pdf); [Wikipedia: CTP](https://en.wikipedia.org/wiki/Canadian_traveller_problem). Wikipedia gives 1989 for the problem's introduction while the journal version is 1991. Cite 1991.
- Stochastic CTP: Nikolova & Karger (AAAI 2008), "Route Planning under Uncertainty: The Canadian Traveller Problem." — [PDF](https://users.ece.utexas.edu/~nikolova/papers/enikolova-aaai08-6pages.pdf)
- **Bonet & Geffner (IJCAI 2011), "Planning under Partial Observability by Classical Replanning: Theory and Experiments,"** pp. 1936–1941. The approach is tractable because of two restrictions. First, the non-unary clauses that encode uncertainty about the initial situation are **invariant**. Second, variables hidden in the initial situation **do not appear in the body of conditional effects**. Under these restrictions the problem translates in linear time to a fully observable non-deterministic (FOND) problem, solvable with classical replanning (sound and complete when the state space is connected). — [IJCAI abstract](https://ijcai.org/Abstract/11/324); [PDF](https://www-i6.informatik.rwth-aachen.de/~hector.geffner/www.dtic.upf.edu/~hgeffner/blai-ijcai11.pdf)
- K-replanner and LW1 were extended in Bonet & Geffner, "Flexible and Scalable Partially Observable Planning with Linear Translations" (AAAI 2014). — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/9047)
- **Brafman & Shani (JAIR 45, 2012, 565–600), "Replanning in Domains with Partial Information and Sensing Actions" (SDR: Sample, Determinize, Replan).** SDR samples a hidden initial state, plans classically, executes while the plan is safe, and replans. It uses lazy regression to query the belief state. DOI 10.1613/jair.3711. — [JAIR](https://jair.org/index.php/jair/article/view/10790); [arXiv 1401.6048](https://arxiv.org/pdf/1401.6048)
- **Talamadupula et al. (AAAI 2010), "Integrating a Closed World Planner with an Open World Robot."** Introduces Open World Quantified Goals for objects not known in the initial state, used in urban search and rescue. — [PDF](https://hrilab.tufts.edu/publications/talamadupulaetal10aaai.pdf). Its formal ancestor is Babaian & Schmolze's PSIPLAN, open-world reasoning with incomplete initial knowledge. — [arXiv cs/0601032](https://arxiv.org/pdf/cs/0601032v1). A later critique (arXiv 2112.11199) notes that the approach replans when new objects appear, not on plan failure, and does not model uncertainty over object properties. — [arXiv 2112.11199](https://www.arxiv.org/pdf/2112.11199)
- 2026 work continues this line in robotics. Examples: "Hypothesis-driven Model Expansion under Uncertainty for Open-World Robot Planning" (NUS, arXiv 2607.06501), "Effective Task Planning with Missing Objects using Learning-Informed Object Search" (arXiv 2602.11468), and "Concurrent Semantic Search and Mission Execution for LTL Missions in Unknown Environments" (arXiv 2609.39153). — [alphaXiv 2607.06501](https://www.alphaxiv.org/abs/2607.06501); [alphaXiv 2602.11468](https://www.alphaxiv.org/abs/2602.11468); [alphaXiv 2609.39153](https://www.alphaxiv.org/abs/2609.39153). I saw only titles and abstract snippets; fetching the full text failed (DNS).

### Inferences
- In contingent and partially observable classical planning, the unknown map is usually encoded as **initial-state uncertainty over atoms that no action changes**. In practice these are static, hidden atoms. The K-replanner's restriction (hidden variables do not appear in conditional-effect bodies; invariant clauses) is the closest formal analogue to "static meta-structure." It identifies a tractable class precisely because the hidden atoms do not interact with the dynamics in complex ways. In this literature, belief over static atoms is also **monotonic**: once an atom is observed it stays known. That is the same property claim (a) states.
- What these papers usually lack is an explicit *conceptual* separation: one layer of beliefs over "which transition system", another over "which state within it", each with its own update rule. They keep a single belief over all hidden atoms, or sample states as SDR does. Information about map atoms arrives through **sensing actions or observation models**, not by abducing which grounded action explains an observed transition.

### Gaps
- I did not verify whether the K-replanner or LW1 papers explicitly use the term "static" or treat static hidden atoms specially beyond the stated restrictions. Read §2–3 of the IJCAI 2011 PDF to confirm.
- I did not verify MiniGrid/BabyAI/Crafter work with a symbolic layer that infers topology. Candidates such as PDSketch (arXiv 2303.05501) and the neurosymbolic world-model papers need checking.

---

## Q2. Abducing preconditions and inferring instance facts from observed transitions under a known model

### Takeaway
The action-model learning literature **does** recognise static predicates as a special case: LOCM is blind to them and LOP (Gregory & Cresswell, ICAPS 2015) adds them. However, these methods learn static *relations in the schema* (which static precondition each operator has) from offline optimal plans. They do **not** keep the schemas fixed while learning the instance's static facts online. Diagnosis-as-planning (Sohrabi, Baier & McIlraith) and plan recognition abduce explanations of observations, but the explanations are hidden *events* or *goals*, not static instance atoms. I found **no paper that does exactly (b)**: fixed lifted schemas, online abduction of static instance atoms from observed abstract transitions, with update rules compiled from preconditions. The idea is simple enough that it may appear implicitly somewhere, for example in model-based diagnosis or in SAM's reasoning. That is a residual risk.

### Cited Findings
- **Gregory & Cresswell (ICAPS 2015), "Domain Model Acquisition in the Presence of Static Relations in the LOP System."**
  - LOCM's assumption that every action parameter undergoes a transition fails for static predicates, so LOP treats static predicates as **restrictions on valid groundings of actions**.
  - Given LOCM2 output and optimal plans, LOP finds a minimal set of static predicates per operator that preserves optimal plan length. Without them, LOCM2 domains yield plans shorter than the true optimum.
  - — [ICAPS](https://ojs.aaai.org/index.php/ICAPS/article/view/13729); [PDF](https://ojs.aaai.org/index.php/ICAPS/article/download/13729/13578)
- A KEPS 2021 paper says LOP's need for optimal plans is "among the strictest requirements" in the LOCM family. — [KEPS 2021 paper 12](https://ai-icaps.dmi.unibas.ch/workshops/KEPS/Papers/KEPS_2021_paper_12.pdf)
- Aineto, Jiménez & Onaindia ("Learning STRIPS Action Models with Classical Planning," arXiv 1903.01153) note that many preconditions missed by LOCM-style learning are static predicates, because they always hold and so are never learned. — [arXiv 1903.01153](https://arxiv.org/pdf/1903.01153)
- Learning lifted STRIPS models (domain predicates included) from action traces alone. These learn the schemas, not the instance facts with fixed schemas.
  - Gösgens, Jansen & Geffner, arXiv 2411.14995 (2024). — [arXiv 2411.14995](https://www.arxiv.org/pdf/2411.14995)
  - Follow-ups (RWTH): arXiv 2605.18627 and arXiv 2605.13282 (differentiable lifted schemas). — [alphaXiv 2605.18627](https://www.alphaxiv.org/abs/2605.18627); [alphaXiv 2605.13282](https://www.alphaxiv.org/abs/2605.13282)
- **Juba, Le & Stern, SAM learning ("Safe Learning of Lifted Action Models," KR 2021 [UNVERIFIED venue]; Stern & Juba IJCAI 2017).** Learns conservative STRIPS models from successful trajectories, so that plans built on the learned model are guaranteed to execute. Predicates and objects are given. — [ar5iv 1705.08961](https://ar5iv.arxiv.org/html/1705.08961); [Simons slides](https://simons.berkeley.edu/sites/default/files/2023-02/sam-theory-final.pdf). SAM's core inference rule is: "if action a(o) was applied in state s, every precondition literal must hold in s, so remove from the candidate preconditions any literal false in s." This is the **dual** of the thesis's rule. SAM fixes the facts and learns the schema; the thesis fixes the schema and learns the facts. I did not find how SAM treats static predicates.
- **Lamanna, Saetti, Serafini, Gerevini & Traverso (IJCAI 2021), OLAM: "Online Learning of Action Models for PDDL Planning."** Learns STRIPS schemas *online* under full observability and chooses goals that make the traces informative. pp. 4112–4118, DOI 10.24963/ijcai.2021/566. — [IJCAI](https://www.ijcai.org/proceedings/2021/566). It is close to the thesis in being online and exploration-driven, but it learns the schemas, not the instance's static facts.
- **Bonet & Geffner (ECAI 2020), "Learning First-Order Symbolic Representations for Planning from the Structure of the State Space."** From the labelled state graph of an instance, learns a general first-order domain (schemas plus predicates, "some of which are possibly static") **and** instance information (objects and initial situation). It uses SAT inside a hyperparameter search. — [arXiv 1909.05546](https://arxiv.org/pdf/1909.05546); [code](https://github.com/bonetblai/learner-strips). Follow-up: Rodriguez, Bonet, Romero & Geffner, "Learning First-Order Representations for Planning from Black-Box States" (KR 2021). — [ar5iv 2105.10830](https://ar5iv.labs.arxiv.org/html/2105.10830). These papers make the domain/instance split explicit, including static atoms that belong to the instance. But they learn both jointly and offline from the full state graph, not online by abduction with fixed schemas.
- **Sohrabi, Baier & McIlraith (KR 2010), "Diagnosis as Planning Revisited."** Explains observations as posited event sequences under incomplete information, by compilation to planning. — [PUC record](https://ing.uc.cl/publicaciones/diagnosis-as-planning-revisited)
- Roos & Witteveen use model-based diagnosis of plans, explaining deviations by abnormal plan steps. — [PDF](https://dke.maastrichtuniversity.nl/nico.roos/wp-content/uploads/2015/04/RW06MBS.pdf)
- Plan recognition as planning: Ramírez & Geffner (IJCAI 2009; AAAI 2010) **[UNVERIFIED in this session]**.

### Inferences
- Overlap with (b) is **partial**.
  - The *logical rule* is folklore in action-model learning: an observed applicable action implies its preconditions held. LOP and SAM use it to constrain schemas.
  - Contingent planners do the same implicitly when they progress beliefs with a known model, because executing an action whose precondition mentions a hidden atom reveals that atom. SDR and K-replanner treat preconditions over hidden atoms this way, either through the K-translation or through sampling and regression.
  - The specific combination appears new in what I found:
    - abduction from *abstract transitions* where the triggering action is not observed (only the event or the before/after states);
    - an **automatic compilation** of the action templates into update rules for the static layer;
    - the static layer kept separate from the fluent belief.
- In a framing for a PhD examiner, the novelty of (b) is better stated as "a compilation from lifted schemas to explanation-based update rules on a separated static layer". The claim "abduction of static facts" is too weak, since that step is a direct consequence of standard belief progression with a known model.

### Gaps
- I could not confirm whether SAM or its extensions (N-SAM, conditional-effects SAM) handle static predicates or instance facts.
- I could not confirm whether any paper explicitly does "fixed schema, learn init-state static atoms from trajectories." Possible places to check:
  - "learning initial state" / "state-space recovery" papers;
  - Aineto et al.'s FAMA, which can learn with unknown intermediate states [UNVERIFIED that it covers static atoms];
  - Bonet & Geffner's "Learning general planning policies... unknown instances."
- I found no concrete paper on inverse planning with an unknown initial state.

---

## Q3. Hidden-parameter, contextual and Bayes-adaptive MDPs, compared with symbolic static facts

### Takeaway
HiP-MDPs and contextual MDPs formalise exactly the structure claim (c) describes. A *static latent parameter* θ, drawn per task, selects the dynamics. The agent infers θ online, as VariBAD does, while the shared structure transfers. The thesis's static facts are a **symbolic, discrete, relational, partially identifiable instantiation of θ**. Mapping static facts to θ is therefore a re-interpretation of a known framework, not a new framework. What remains distinctive is that θ is a set of ground atoms over objects, structured by lifted schemas, so that inference is logical (abductive) rather than GP- or VAE-based.

### Cited Findings
- **Doshi-Velez & Konidaris (IJCAI 2016), "Hidden Parameter Markov Decision Processes: A Semiparametric Regression Approach for Discovering Latent Task Parametrizations,"** pp. 1432–1440. A family of related dynamical systems is parametrised by low-dimensional latent factors θ; fixing θ gives one instance. — [IJCAI](https://www.ijcai.org/Abstract/16/206); [arXiv 1308.3513](https://ar5iv.arxiv.org/html/1308.3513)
- Killian, Daulton, Konidaris & Doshi-Velez (NeurIPS 2017), "Robust and Efficient Transfer Learning with HiP-MDPs," replace the GP with a BNN. — [NeurIPS](https://papers.neurips.cc/paper_files/paper/2017/hash/2227d753dc18505031869d44673728e2-Abstract.html)
- **Hallak, Di Castro & Mannor (2015), "Contextual Markov Decision Processes,"** arXiv 1502.02259. Dynamics and rewards depend on a "hidden static parameter referred to as the context." — [alphaXiv 1502.02259](https://www.alphaxiv.org/abs/1502.02259)
- **Zintgraf et al. (ICLR 2020; JMLR 22, 2021), VariBAD.** Meta-learns approximately Bayes-optimal policies conditioned on a VAE posterior over a task embedding. In gridworlds it explores cells not yet visited until it locates the goal. — [arXiv 1910.08348](https://arxiv.org/pdf/1910.08348); [JMLR](https://www.jmlr.org/papers/v22/21-0657.html)
- RL² (Duan et al. 2016) and the Procgen benchmark (Cobbe et al. 2020) — **[UNVERIFIED in this session]**.

### Inferences
- For claim (c), the closest existing frame is the HiP-MDP or CMDP, where θ is static per episode or environment and the shared structure transfers. The thesis's version is symbolic, which brings several differences:
  - θ ↦ a set of ground static atoms;
  - the shared structure ↦ the lifted schemas;
  - posterior inference ↦ logical abduction, which is exact and monotonic.
- There is also a practical difference. A HiP-MDP θ is low-dimensional and continuous. Symbolic static facts scale with the number of objects (O(n²) for `connected`) and are identifiable only along visited edges. That makes the problem closer to CTP and map learning than to classical HiP-MDP regression.

### Gaps
- I found no paper that explicitly equates PDDL static predicates with HiP-MDP or CMDP hidden parameters. This framing seems to be a genuine bridging contribution, but my search was not exhaustive.

---

## Q4. Generalized planning and domain-level transfer (templates transfer, instances vary)

### Takeaway
Generalized planning and generalized-policy learning already assume that the domain (schemas) is shared and the instances vary. Static facts such as maps and connectivity are part of the instance and are given to the solver in full. So "templates transfer, static facts are per-instance" is the **standard** domain/instance split of PDDL and generalized planning. What the thesis adds is that the per-instance static facts are *unknown and learned online*.

### Cited Findings
- **Jiménez, Segovia-Aguas & Jonsson (2019), "A review of generalized planning,"** *Knowledge Engineering Review* 34, e5, 1–28, DOI 10.1017/S0269888918000231. Surveys formalisms and algorithms for solutions that generalise across instances, and their links to planning under uncertainty. — [Cambridge](https://resolve.cambridge.org/core/journals/knowledge-engineering-review/article/abs/review-of-generalized-planning/61056E7879134E057677CE0CEDD1339E); [OA PDF](https://www.maxapress.com/data/article/ker/preview/pdf/S0269888918000231.pdf)
- Bonet & Geffner (ECAI 2020) separate a general first-order domain from per-instance information (objects and initial situation). — [arXiv 1909.05546](https://arxiv.org/pdf/1909.05546)
- Other works in this line, all **[UNVERIFIED in this session]**:
  - Srivastava, Immerman & Zilberstein (AAAI 2008; AIJ 2011 "A new representation and associated algorithms for generalized planning");
  - Toyer et al., ASNets (AAAI 2018; JAIR 2020);
  - Ståhlberg, Bonet & Geffner, GNN generalized policies (ICAPS 2022; KR 2022);
  - Rivlin, Hazan & Karpas, "Generalized planning with deep RL" (2020).

### Inferences
- The "templates transfer" half of (c) is standard: it is the PDDL domain/problem split. Novelty can only come from the "static facts unknown, relearned per environment θ" half, together with the HiP-MDP correspondence.

### Gaps
- I did not find generalized-planning work in which instance static facts are hidden and learned during execution. The nearest are generalized contingent or POMDP policies, which I did not search for specifically.

---

## Overlap verdicts per claim

**(a) Separating static-fact (meta-structure) uncertainty from fluent uncertainty, with static beliefs monotonic: PARTIAL overlap.**
- The *mechanics* exist. Contingent and partially observable replanners (Bonet & Geffner 2011; Brafman & Shani 2012), CTP and free-space navigation all keep belief over static hidden atoms such as map edges or blocked roads, and those beliefs are monotonic once observed. The K-replanner's tractability conditions are close to formalising "static hidden atoms" as a special class.
- HiP-MDP and CMDP separate a static latent parameter from the state.
- What I did not find in symbolic planning is an explicit two-layer epistemic architecture: belief over transition systems induced by static facts, as distinct from belief over fluents, with different update semantics. The contribution is defensible as an explicit, named separation, but it must cite these works as implicit precedents.

**(b) Automatic derivation of static-fact update rules from action templates by precondition abduction: PARTIAL, bordering on superficial for the core logic.**
- The inference "the observed transition was caused by a(ō), so pre(a(ō)) held, including its static atoms" is standard. It underlies SAM (used in the dual direction), LOP (static predicates as restrictions on groundings) and belief progression in contingent planning.
- Novel elements not found elsewhere:
  - compiling update rules automatically from lifted schemas for the static layer only;
  - abduction from abstract transitions where the action instance is unobserved and must be identified, with uniqueness ("the only template instance explaining this");
  - treating ambiguity when several instances explain the transition.
- Recommendation: frame (b) as "schema-to-update-rule compilation with identifiability conditions," not as abduction per se.

**(c) Static facts as the symbolic image of environment hyperparameters θ (relearned per environment while templates transfer): PARTIAL overlap.**
- Conceptually this is the HiP-MDP / contextual-MDP / Bayes-adaptive meta-RL structure (Doshi-Velez & Konidaris 2016; Hallak et al. 2015; VariBAD 2020), combined with the standard PDDL domain/instance split used in generalized planning.
- The bridge is apparently not stated in the literature I found. That bridge is that PDDL static predicates *are* the HiP-MDP hidden parameter, and that procedural-generation hyperparameters project onto static atoms. Its novelty lies in the bridging and the symbolic, relational and logical inference, not in either side.

No claim is **identical** to an existing work. The closest single papers are:
- for (a): Bonet & Geffner IJCAI 2011 and Brafman & Shani JAIR 2012;
- for (b): Gregory & Cresswell ICAPS 2015 (LOP) and SAM (Stern & Juba 2017; Juba et al. 2021);
- for (c): Doshi-Velez & Konidaris IJCAI 2016 and Bonet & Geffner ECAI 2020 (domain/instance split, including static predicates).
