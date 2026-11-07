# Systematic review facet RS2: RL and decision-theoretic formalisms where a hidden static parameter or relational structure selects the dynamics (check of thesis differential D2)

D2, as tested here: *symbolic static facts (predicates that no action changes, for example `connected(r1,r2)`) are the hidden context that selects the transition system. The lifted action templates are known. The agent identifies the facts online by interaction, using logical (not probabilistic) inference: abduction of preconditions, plus knowledge tracking (Π⁺, Π⁻) with observation rules. The work is compared with memory-based RL baselines (PPO-GRU, DRQN).*

To grade each study, I split D2 into six features:

- **F1.** The hidden object is static per episode or task and selects the dynamics.
- **F2.** It is discrete and relational (ground atoms of static predicates).
- **F3.** The dynamics templates are known; only the hidden parameter is unknown.
- **F4.** Identification is online, by interaction, within the episode or task.
- **F5.** Inference is logical or set-based (consistency/elimination), not a posterior or a learned embedding.
- **F6.** Positive and negative knowledge is tracked explicitly, with observation rules.

Verdict scale: IDÊNTICO, SOBREPOSIÇÃO FORTE, PARCIAL, DISTANTE.

This note extends `Diferenciais da tese revisão sistemática/rs_fatos_estaticos.md`, which already covers James, Rosman & Konidaris 2020, Occhipinti et al. 2022 and the relocatable-model line, and `rs_modelo_de_interacao.md`, which covers PaRL and PO-PaRL. I do not repeat those analyses here; they are only referenced.

---

## Q1. Protocol, search log and PRISMA-style counts

### Takeaway
I ran 17 distinct search strings over two discovery tools (2 on alphaXiv discover, 15 on WebSearch). I then read 7 papers in full or in large part through alphaXiv PDF queries, and snowballed from 4 seeds. 22 studies were included. None is IDÊNTICO to D2.

### Protocol (fixed before searching)
- **Facet question:** Which RL or decision-theoretic formalisms model a hidden, static (per episode or per task) parameter or relational structure that selects the dynamics, and that the agent infers by interaction? For each one: what is hidden, how is it represented, how is it inferred, are the templates known, and what guarantees exist?
- **Inclusion:**
  - published 2000–2026;
  - a formalism or method where a hidden static, task-level parameter or relational structure determines the dynamics (or the transition rules) and is inferred or learned from interaction.
  - I also included relational or object-oriented model-learning RL where the *rules* are hidden. This is the "inverse" of D2 and is needed for contrast.
- **Exclusion:**
  - E1: pure continuous control with no hidden structure;
  - E2: non-English;
  - E3: duplicates;
  - E4: no interaction-based inference (for example, the context is fully observed and nothing is inferred). E4 studies are kept only as boundary cases, and are labelled as such.
- **Seeds:**
  - S-a: HiP-MDP (Doshi-Velez & Konidaris, IJCAI 2016);
  - S-b: James, Rosman & Konidaris, ICML 2020;
  - S-c: DOORMAX / OO-MDP (Diuk, Cohen & Littman, ICML 2008);
  - S-d: Lang, Toussaint & Kersting, JMLR 2012.

### Search log
| # | String | Tool | Hits screened | Retained (new) |
|---|---|---|---|---|
| A1 | keywords [hidden parameter MDP, contextual MDP, latent MDP, Bayes-adaptive MDP, VariBAD, epistemic POMDP]; question "RL formalisms where a hidden static per-task parameter selects the dynamics…" | alphaXiv discover (historical) | 15 | 3 (VariBAD 1910.08348; Kwon et al. 2024 LMDP 2406.01389; Meta-RL Bayesian linear task models 2512.20974, screened by title, not retained beyond the list) |
| A2 | keywords [object-oriented MDP, DOORMAX, relational RL, schema networks, KWIK, relational exploration]; question "relational/OO model-based RL… known templates, unknown per-task relational structure" | alphaXiv discover (historical) | 12 | 3 (Feng et al. 2502.06146; Modi et al. 1711.05726; Epistemic exploration 2402.08145, title only) |
| S1 | `Hidden Parameter Markov Decision Processes Doshi-Velez Konidaris IJCAI 2016` | WebSearch | 10 | 2 (HiP-MDP; Killian et al. AAAI-17 / arXiv 1612.00475) |
| S2 | `Contextual Markov Decision Processes Hallak Di Castro Mannor 2015` | WebSearch | 10 | 1 (+ Modi ALT-18 pointer) |
| S3 | `Kwon latent MDP NeurIPS 2021 "RL for Latent MDPs" regret guarantees` | WebSearch | 10 | 1 |
| S4 | `mixed observability MDP MOMDP Ong Png Hsu Lee …` | WebSearch | 10 | 1 (RSS-09 / IJRR-10) |
| S5 | `"An Object-Oriented Representation for Efficient Reinforcement Learning" Diuk DOORMAX` | WebSearch | 10 | 1 |
| S6 | `Lang Toussaint Kersting "Exploration in relational domains…" JMLR` | WebSearch | 10 | 1 |
| S7 | `Ghosh "Why Generalization in RL is Difficult: Epistemic POMDPs…" NeurIPS 2021` | WebSearch | 10 | 1 |
| S8 | `Kansky Schema Networks zero-shot transfer … ICML 2017` | WebSearch | 10 | 1 |
| S9 | `Walsh Littman KWIK learning STRIPS action schemas preconditions …` | WebSearch | 10 | 1 (Walsh & Littman AAAI-08) |
| S10 | `Bayes-adaptive Monte-Carlo planning Guez Silver Dayan BAMCP Duff …` | WebSearch | 10 | 1 (BAMCP NeurIPS-12 / JAIR-13) |
| S11 | `hidden-parameter relational MDP unknown static map topology learned logical inference exploration reinforcement learning` | WebSearch (extended) | 9 | 3 (HiP polynomial MDPs AAAI-23; POMDP-lite 1602.04875; Fu et al. ICLR-23 HiP-MDP bounds) |
| S12 | `Killian Daulton Konidaris Doshi-Velez "Robust and Efficient Transfer Learning with HiP-MDPs" NeurIPS 2017` | WebSearch | 10 | 1 |
| S13 | `Marom Rosman "Zero-Shot Transfer with Deictic Object-Oriented Representation…" NeurIPS 2018` (forward from seed S-c) | WebSearch | 10 | 1 |
| S14 | `Džeroski De Raedt Driessens relational reinforcement learning Machine Learning 2001` | WebSearch | 10 | 1 |
| S15 | `Modi Jiang Tewari Singh "MDPs with continuous side information" ALT 2018 …` | WebSearch | 10 | 1 |

There are 17 strings in total. A1 and A2 are multi-facet alphaXiv queries; each counts as one string.

### Snowballing (seeds)
- **S-a, HiP-MDP.**
  - Forward: Killian et al. AAAI-17 and NeurIPS-17, Fu et al. ICLR-23 (performance bounds), Costen et al. AAAI-23 (HiP polynomial MDPs).
  - Lateral, via the VariBAD related work: Zintgraf et al. place HiP-MDPs, CMDPs (Hallak et al.; Modi & Tewari) and BAMDPs in one family ([VariBAD §4](https://www.alphaxiv.org/abs/1910.08348)).
  - Backward, from Kwon et al. 2021: LMDPs "have previously been introduced under many different names, e.g., hidden-model MDP, Multitask RL, Contextual MDP, Multi-modal MDP and Concurrent MDP" ([Kwon et al. 2021](https://www.alphaxiv.org/abs/2102.04939)).
- **S-b, James et al. 2020.** Covered in full in `rs_fatos_estaticos.md`: portable lifted rules plus per-task linking functions learned by counting. No new forward citations were obtainable here, because citation APIs are blocked.
- **S-c, DOORMAX.**
  - Forward: Marom & Rosman NeurIPS-18 (Deictic OO-MDPs, DOORMAX_D), read in full.
  - Its reference list adds Guestrin et al. IJCAI-03 (relational MDPs, dynamics known), Kansky et al. 2017, Scholz et al. ICML-14, and Li, Littman & Walsh ICML-08 (KWIK) ([Marom & Rosman 2018](https://papers.NeurIPS.cc/paper_files/paper/2018/file/df0aab058ce179e4f7ab135ed4e641a9-Paper.pdf)).
- **S-d, Lang et al. 2012.**
  - Backward: Walsh 2010 (PhD thesis, KWIK learning of relational models), Pasula, Zettlemoyer & Kaelbling 2007 (NID rules), and Li et al. 2011 (KWIK) ([Lang et al. JMLR 2012](https://jmlr.org/papers/v13/lang12a.html)).
  - Forward/lateral, via A2: GLIB (Chitnis et al. AAAI-21) and Feng et al. 2025.

### PRISMA-style counts (approximate)
- **Identified:** 177 records (15 WebSearch strings × about 10 links, plus 27 from alphaXiv). Snowballing added 6.
- **Duplicates removed:** about 95. Most WebSearch hits were mirrors of the same paper (proceedings, mlanthology, aitopics, reviewer sites).
- **Distinct records screened (title and abstract):** about 88.
- **Excluded at screening:** about 66. Reasons:
  - E1: continuous-control meta-RL and robotics adaptation papers such as PrivilegedDreamer, BEACON and "Scouting the dynamics gap";
  - E4 or off-topic: object-centric slot models (SOLD), GNN relational policy learners, in-context RL surveys.
- **Included:** 22. Seven were read in full or in large part:
  - VariBAD;
  - Fu et al. ICLR-23;
  - Lang et al. JMLR-12;
  - Feng et al. 2025;
  - Marom & Rosman NeurIPS-18;
  - Kwon et al. NeurIPS-21;
  - Hallak et al. 2015.

  The rest were assessed from abstracts or snippets. Three more (James et al. 2020, PaRL, PO-PaRL) are rated from the earlier full-text notes.

### Gaps
- The DOORMAX PDF (icml.cc) could not be fetched: the host is outside the session's allowed network. DOORMAX is therefore characterised from the abstract and from the detailed restatement in Marom & Rosman 2018, which was read in full.
- No citation index was available, so forward snowballing is approximate.

---

## Q2. Latent-context formalisms: HiP-MDP, CMDP, LMDP, BAMDP, VariBAD/meta-RL, MOMDP, epistemic POMDP

### Takeaway
Every one of these formalisms already has **F1** (a static hidden parameter that selects the dynamics), and most have **F4** (online identification). So "a static hidden context selecting the transition system" is **not** new in itself. The defensible differences are:

- **F2:** the hidden object is a set of ground relational atoms, not a continuous vector or an index;
- **F3:** the templates are known, so the per-context models are not learned;
- **F5/F6:** identification is by logical consistency, not by a posterior or an embedding.

Kwon et al.'s LMDP is the formal closest match: a finite set of M MDPs, one drawn per episode and hidden. Its deterministic-LMDP appendix even identifies the context by distinguishing observations rather than by statistics.

### Cited Findings
**HiP-MDP family**
- In HiP-MDPs, a "family of related dynamical systems [is described] with a small set of latent factors". Each task instance, conditioned on θ, is an ordinary MDP, and θ has no dynamics of its own during the task. θ is learned with Indian Buffet Processes and Gaussian processes (semiparametric regression) — [IJCAI-16 abstract](https://www.ijcai.org/Abstract/16/206); [arXiv 1308.3513](https://ar5iv.arxiv.org/html/1308.3513)
- Killian et al. (AAAI-17) identified a flaw: in the original formulation, embedding uncertainty was "modelled independently from the agent's state uncertainty". The NeurIPS-17 version models "the joint uncertainty in the latent parameters and the state space" and replaces the GP with a Bayesian neural network — [NeurIPS-17](https://proceedings.neurips.cc/paper/2017/hash/2227d753dc18505031869d44673728e2-Abstract.html)
- Fu, Yao, Gottesman, Doshi-Velez & Konidaris (ICLR 2023) give guarantees for HiP-MDPs:
  - θ is drawn from P_Ω; the agent "neither observes θ nor has access to the distribution";
  - the bounds assume that T, R and the policy are **Lipschitz in θ**, and they bound regret linearly in a Wasserstein estimation error ε of the θ-posterior;
  - the experiments use continuous angles (goal or wind direction) as θ.

  — [Fu et al. ICLR-23](https://cs.brown.edu/people/gdk/pubs/hipmdp_bounds.pdf)

**Contextual MDP**
- Hallak, Di Castro & Mannor (2015) define a CMDP as a tuple (C, S, A, M(c)): "dynamics and rewards … depend on a hidden static parameter referred to as the context". The paper focuses on "finite horizon with a small known number of possible contexts". It notes that static features put into the state form "distinct unconnected dynamic chains" — [arXiv 1502.02259](https://www.alphaxiv.org/abs/1502.02259)
- Their CECE algorithm (Cluster, Explore, Classify, Exploit) works as follows:
  - it clusters trajectories into K models, then explores with a uniform policy for T_EC steps;
  - it classifies by the smallest L1 distance between empirical and model transitions, so it is statistical;
  - regret bounds require a separation D between contexts and visits to every (s,a) pair;
  - the authors also view CMDPs as robust MDPs with uncertainty that is "singular — determining one transition sets all of them".

  — [arXiv 1502.02259](https://www.alphaxiv.org/abs/1502.02259)
- Modi, Jiang, Tewari & Singh (ALT 2018) treat a context that is *observed* at the start of each episode, with MDP parameters varying smoothly with it. They prove PAC bounds, including a KWIK-based algorithm for the linear-mixture case. Because the context is observed, this is a boundary case under E4 — [PMLR v83](https://proceedings.mlr.press/v83/modi18a.html)

**Latent MDP**
- Kwon, Efroni, Caramanis & Mannor (NeurIPS 2021) define an LMDP as M MDPs over a joint S and A. "The static latent (hidden) variable that selects one of M MDPs is randomly chosen at the beginning of each episode, yet is not revealed". The belief over m is updated by Bayes' rule using T_m and R_m — [arXiv 2102.04939](https://www.alphaxiv.org/abs/2102.04939)
- Main results of Kwon et al. 2021:
  - a lower bound of Ω((SA)^M) episodes in general, "even … with deterministic MDPs";
  - polynomial regret (L-UCRL) under δ-strong separation and with a good initialisation;
  - initialisation can be removed with PSR sufficiency and reachability assumptions.

  — [arXiv 2102.04939](https://www.alphaxiv.org/abs/2102.04939)
- **Planning with known parameters is a named sub-problem.** Prior work on "hidden-model MDP" and "multi-model MDP" studies planning when the true parameters are given. It is PSPACE-hard to plan exactly and NP-hard to find an optimal memoryless policy — [Kwon et al. 2021 §1.1](https://www.alphaxiv.org/abs/2102.04939)
- **Appendix A.2 (deterministic LMDPs)** works without statistics:
  - "Exploring Deterministic MDPs with Latent Contexts" (Algorithm 5) records *distinguishing observations* C;
  - it branches only when (s,a) produces more than one observed outcome;
  - it finds the optimal policy within O(H(SA)^M · poly(M) log M) episodes.

  — [arXiv 2102.04939](https://www.alphaxiv.org/abs/2102.04939)
- Kwon et al. 2024 ("RL in Latent MDPs is Tractable: Online Guarantees via Off-Policy Evaluation") describe problems with "hidden or latent information that remains fixed throughout an interaction" (abstract only) — [arXiv 2406.01389](https://www.alphaxiv.org/abs/2406.01389)

**Bayes-adaptive MDP and meta-RL**
- **BAMDP.** The state is augmented with a belief over (R, T); "the belief is over the transition and reward functions, which are constant for a given task" (Martin 1967; Duff & Barto 2002). Solving it is "hopelessly intractable for most interesting problems" — [VariBAD §2.2](https://www.alphaxiv.org/abs/1910.08348)
- **BAMCP** (Guez, Silver & Dayan, NeurIPS 2012; JAIR 2013) runs MCTS with root sampling from the posterior and converges to the Bayes-optimal policy — [NeurIPS-12 PDF](https://proceedings.neurips.cc/paper_files/paper/2012/file/35051070e572e47d2c26c241ab88307f-Paper.pdf); [JAIR-13 PDF](https://www.gatsby.ucl.ac.uk/~aguez/files/guez_jair2013.pdf)
- **VariBAD** (Zintgraf et al., ICLR 2020):
  - represents the task by a "learned, low-dimensional stochastic latent variable m";
  - uses an RNN/GRU encoder q_φ(m|τ) trained by an ELBO;
  - conditions the policy on the posterior.

  In its 5×5 gridworld with a hidden goal, the learned posterior "predicts no reward for cells it has visited", which is in effect elimination. However, "variBAD lacks the formal guarantees" because it uses deep networks — [arXiv 1910.08348](https://www.alphaxiv.org/abs/1910.08348)
- VariBAD's related work states that HiP-MDPs and CMDPs are "related to BAMDPs", and that "in a BAMDP the underlying task, and therefore the hidden state, is fixed per task. We exploit this property by learning an embedding that is fixed over time" — [arXiv 1910.08348 §4](https://www.alphaxiv.org/abs/1910.08348)

**MOMDP**
- Ong, Png, Hsu & Lee (RSS 2009; IJRR 2010) factor the state into fully observable and partially observable components. Beliefs are kept only over the hidden part, which gives a lower-dimensional belief space and much faster point-based solving — [RSS-09 PDF](https://adacomp.comp.nus.edu.sg/wp-content/uploads/2016/01/rss09.pdf); [NUS page](https://adacomp.comp.nus.edu.sg/offline-pomdp-planning)

**Epistemic POMDP**
- Ghosh et al. (NeurIPS 2021) show that training on a limited set of environments induces "implicit partial observability". Generalisation is then a POMDP over which environment is true (the epistemic POMDP); "standard RL methods that ignore this can be arbitrarily suboptimal". They approximate it with an ensemble (LEEP) on Procgen — [arXiv 2107.06277](https://arxiv.org/pdf/2107.06277); [NeurIPS PDF](https://papers.neurips.cc/paper_files/paper/2021/file/d5ff135377d39f1de7372c95c74dd962-Paper.pdf)

**Related planning-side variants (snippet only)**
- POMDP-lite (Chen et al., IJCAI 2016 [venue from memory - unverified]) "maintains a belief … over the hidden parameter" and uses an internal reward bonus. Because "the hidden state is fixed, the agent does not have to reason over hidden state transitions" — [arXiv 1602.04875](https://arxiv.org/pdf/1602.04875)
- Planning with HiP *polynomial* MDPs (Costen et al., AAAI 2023): snippet only — [PDF](https://www.robots.ox.ac.uk/~nickh/papers/costen23aaai.pdf)

### Inferences
- **The thesis model is, formally, a hidden-model / latent MDP with known per-context models.** Each assignment of the static facts consistent with the type constraints is one context m. The known lifted templates give T_m exactly. So the thesis sits in the "planning when parameters are given" branch that Kwon et al. cite (hidden-model MDP, multi-model MDP), not in the "learn T_m" branch. The thesis should say this explicitly, because it pre-empts the reviewer objection "this is just an LMDP or CMDP".
- **In the deterministic case, logical inference is equivalent to a uniform-prior Bayesian belief restricted to its support.** With deterministic templates and truthful observations, every likelihood is 0 or 1. A Bayesian belief then collapses to a uniform distribution over the set of consistent contexts, which is exactly what (Π⁺, Π⁻) represent in factored form. This is also how Kwon's deterministic-LMDP algorithm and the VariBAD gridworld posterior behave. "Logical, not probabilistic" is therefore **not** a difference in *what is inferred*. The real difference is computational and representational:
  - (i) the consistent set is represented *factored and lifted* (per-atom known-true/known-false) instead of enumerating up to 2^|static atoms| contexts;
  - (ii) the inference is abductive over precondition explanations, not a belief update over flat contexts;
  - (iii) no prior or mixing weights w_m are needed.

  The thesis should phrase D2 this way rather than as "non-probabilistic".
- **The factored knowledge state does not always represent the exact consistent set.** Per-atom (Π⁺, Π⁻) can represent it exactly only when the set is a product (a conjunction of literals). Disjunctive explanations (for example, "the action failed, so ¬p ∨ ¬q") make it non-product, and per-atom tracking then loses information. MOMDP's factoring of the belief over the hidden variables is the probabilistic analogue. The thesis should state when (Π⁺, Π⁻) is exact and when it is a sound under-approximation.
- **The HiP-MDP regret theory does not carry over.** Fu et al.'s bounds rest on Lipschitz continuity in θ, which is meaningless for Boolean atoms: flipping one `connected` atom can change reachability discontinuously. This supports F2 as a real differentiator. The Kwon et al. separation assumption, by contrast, has a discrete analogue: two fact assignments are distinguishable iff some reachable (s,a) has different outcomes. This is the natural identifiability notion for D2.
- **The epistemic POMDP justifies the choice of baselines.** Ghosh et al. formally explain why memoryless policies fail when the training-consistent set of environments is not a singleton. This justifies recurrent baselines (PPO-GRU, DRQN) as the right model-free comparison. The thesis can frame them as approximate epistemic-POMDP / meta-RL solvers in the RL² and VariBAD style.

### Gaps
- Chadès et al., "MOMDPs: a solution for modelling adaptive management problems" (AAAI 2012), and Steimle et al. ("multi-model MDP") are the planning-with-known-models references cited by Kwon. I did not open them [from memory - unverified authorship and venue]. Both should be checked, because they are the closest formal match to "known per-context models, hidden context".
- I did not verify whether any LMDP or CMDP paper uses a *factored or relational* context space; none was found in the screened records.
- Duff's 2002 thesis and Martin 1967 are cited only through VariBAD.

---

## Q3. Relational, object-oriented and symbolic variants: what is hidden and how it is inferred

### Takeaway
In the relational RL line (OO-MDP/DOORMAX, Deictic OO-MDP, KWIK-STRIPS, REX, schema networks, GLIB, RRL), what is hidden is the **rules**, that is, the lifted transition model. The **instance structure** (walls, map, object relations) is *observed* in the state. D2 is the inverse: rules known, instance static structure hidden.

Some of these methods do use logical, elimination-style inference with KWIK or PAC-MDP bounds:

- DOORMAX intersects condition sets;
- KWIK-STRIPS learns preconditions by elimination.

Their inference machinery is therefore the closest technical precedent for D2's abduction. James et al. 2020 (from the earlier note) remains the only included study where lifted rules transfer and *problem-specific* structure is relearned per task.

### Cited Findings
**OO-MDPs and DOORMAX**
- Diuk, Cohen & Littman (ICML 2008) introduce OO-MDPs. DOORMAX learns *deterministic* OO-MDPs with a "polynomial bound on its sample complexity", on Taxi and a video game — [mlanthology](https://mlanthology.org/icml/2008/diuk2008icml-object); [ICML PDF](https://icml.cc/imls/conferences/2008/papers/571.pdf) (PDF not fetchable here; abstract-level)
- As restated by Marom & Rosman (read in full):
  - in Propositional OO-MDPs, "the transition dynamics are represented in terms of propositional preconditions over object classes that map to effects";
  - DOORMAX learns them by intersecting term sets that produced the same effect. Their example: two observations yield "T_E(Taxi, Wall) = 0" as the relevant condition;
  - the assumptions are a full binary tree of conditions and each effect appearing at most once;
  - the result is a KWIK bound.

  — [Marom & Rosman NeurIPS-18](https://papers.NeurIPS.cc/paper_files/paper/2018/file/df0aab058ce179e4f7ab135ed4e641a9-Paper.pdf)
- In OO-MDP Taxi, walls are **objects with x, y and pos attributes in the state**, so the map is observed. Relations such as T_E(taxi, s) are computed from the state. Marom & Rosman use "one of four wall configurations" per task and learn schema dynamics that transfer zero-shot across wall configurations and grid sizes, and from a toy Sokoban level to a Micro-Cosmos level with about 10^6 states — [Marom & Rosman 2018](https://papers.NeurIPS.cc/paper_files/paper/2018/file/df0aab058ce179e4f7ab135ed4e641a9-Paper.pdf)
- DOORMAX_D's KWIK bound is N(K₀M + K₁(D+1)+1)+N−1 for D hypothesised deictic preconditions and N effect types, "if exactly one h* ∈ H is true". Relational MDPs in the style of Guestrin et al. 2003, by contrast, "assume that the transition dynamics are known" — [Marom & Rosman 2018](https://papers.NeurIPS.cc/paper_files/paper/2018/file/df0aab058ce179e4f7ab135ed4e641a9-Paper.pdf)

**KWIK learning of STRIPS schemas**
- Walsh & Littman (AAAI 2008): "general STRIPS operators cannot be efficiently learned through raw experience", but restricting precondition size gives a positive result; a teacher providing traces removes the restriction — [mlanthology](https://mlanthology.org/aaai/2008/walsh2008aaai-efficient); [Rutgers record](https://scholarship.libraries.rutgers.edu/esploro/outputs/technicalDocumentation/Efficient-Learning-of-Action-Schemas-and/991031550148804646)
- Walsh's (2010) thesis "proved the existence of a KWIK learning algorithm in a relational RL setting by nesting several KWIK algorithms". Walsh himself called it "clearly approaching the edge of tractability" — cited in [Lang et al. JMLR 2012](https://jmlr2020.csail.mit.edu/papers/volume13/lang12a/lang12a.pdf)

**REX (Lang, Toussaint & Kersting, JMLR 2012)**
- REX defines relational count functions κ_Q(s) = Σ_q c_E(q)·I(∃σ: s ⊨ σ(q)) over relational queries. The dynamics are NID rules that are learned (Pasula et al.); the agent is given "a relational vocabulary" and the reward. REX is PAC-MDP *if* it has a relational KWIK learner and exact planning (Corollary 4, which requires mutually exclusive contexts, no deictic references and no noise outcomes). The practical instance has no guarantees — [JMLR 13:3725](https://jmlr2020.csail.mit.edu/papers/volume13/lang12a/lang12a.pdf)
- REX's IPPC experiments run "subsequent rounds (worlds do not change)", and rules transfer to new tasks with different objects. Static world structure is part of the observed state — [JMLR 2012](https://jmlr2020.csail.mit.edu/papers/volume13/lang12a/lang12a.pdf)

**GLIB and guided exploration**
- Feng, Kumar, Lozano-Pérez & Kaelbling (2025) build on GLIB:
  - the predicates P and the objects are given, and the state is a fully observed set of ground atoms;
  - lifted operators ⟨v, P, E⁺, E⁻, a⟩ are learned online;
  - informative data are "transitions that provide counterexamples to an existing operator's lifted preconditions";
  - goals are chosen to separate learned preconditions from true ones that are stronger, weaker or mixed. The guidance is an oracle that uses the true model.

  — [arXiv 2502.06146](https://www.alphaxiv.org/abs/2502.06146)

**Schema networks and relational RL**
- Kansky et al. (ICML 2017) present schema networks as "a generative model for object-oriented reinforcement learning and planning". Schemas are causal fragments over entities and attributes, *learned*. They are evaluated on zero-shot transfer across Breakout variants against A3C — [ICML PDF](https://www.vicarious.com/wp-content/uploads/2020/03/Paper-ICML2017.pdf); [mlanthology](https://mlanthology.org/icml/2017/kansky2017icml-schema)
- Džeroski, De Raedt & Driessens (MLJ 2001) represent states as "a set of ground facts" and learn relational Q-functions by regression trees. In blocks-world, "the effects of the actions are unknown to the agent" — [mlanthology](https://mlanthology.org/mlj/2001/dzeroski2001mlj-relational); [dblp](https://dblp.uni-trier.de/rec/journals/ml/DzeroskiRD01.html)

**From earlier notes (full text there)**
- James, Rosman & Konidaris (ICML 2020): lifted rules in an egocentric agent-space transfer across tasks; per-task linking functions are learned "by simply executing options and recording the start and end partition labels", with a count-based approach — [arXiv 1905.12006](https://www.alphaxiv.org/abs/1905.12006); see `rs_fatos_estaticos.md`
- PO-PaRL (Rozek et al. 2024): FOND discovery predicates over L(h), PPO options — see `rs_modelo_de_interacao.md`

### Inferences
- **The roles are inverted.** OO-MDP, DOORMAX, KWIK-STRIPS, REX, GLIB and Feng et al. assume *observed* instance structure (walls, objects and their relations are in s) and *unknown* rules. D2 assumes *known* rules and *unobserved* static relations. In OO-MDP terms, D2 is "the touch_E(taxi, wall) attribute values are not observed, but the DOORMAX tree is given". None of the included relational RL works addresses that case.
- **The inference machinery transfers, but in reverse.** DOORMAX's condition-set intersection and KWIK-STRIPS's precondition elimination are logical, set-based procedures with KWIK bounds, much like D2's abduction. The thesis could borrow their analysis style: a KWIK-like bound on the number of "⊥" (unknown) predictions before the static facts relevant to the plan are determined. Such a bound plausibly depends on the number of static atoms in reachable precondition sets rather than on |S|. This bound is an inference of mine, not a published result.
- **The knowledge representations are related.** DOORMAX's "⊥" and REX's known/unknown state counts are knowledge states about *rules*. D2's (Π⁺, Π⁻) is a knowledge state about *facts*. The analogy is useful for positioning, but the objects are different.

### Gaps
- I could not open the DOORMAX original or Diuk's thesis (the KOOL stochastic extension).
- The Walsh 2010 thesis was cited only through Lang et al.
- No paper was found that assumes known OO-MDP or STRIPS rules and unobserved static object relations, under RL framing, with an exact-identification guarantee. This absence holds only within the screened records.

---

## Q4. Included studies and verdicts against D2

### Takeaway
No study is IDÊNTICO. One is SOBREPOSIÇÃO FORTE: James et al. 2020, carried over from the earlier review. Kwon et al. 2021 (LMDP) is the strongest *formal* overlap, at PARCIAL-alto, because the context there is a flat index with models to be learned. The other formalisms are PARCIAL on F1 and F4 only.

### Cited Findings (table)
Abbreviations: FT = full text read (in large part); Abs = abstract or snippet; EN = earlier note (full text read there).

| # | Study (bibliographic) + URL | Read | What is hidden | Representation | Inference | Templates known? | Guarantees | D2 verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | James, Rosman & Konidaris, "Learning Portable Representations for High-Level Planning", ICML 2020 — [arXiv 1905.12006](https://www.alphaxiv.org/abs/1905.12006) | EN | problem-specific partition links (layout) | discrete learned symbols | count-based | lifted agent-space rules learned once, transferred | none on identification | **SOBREPOSIÇÃO FORTE** (F1, F3-ish, F4; not F5/F6; symbols learned) |
| 2 | Kwon, Efroni, Caramanis & Mannor, "RL for Latent MDPs: Regret Guarantees and a Lower Bound", NeurIPS 2021 — [arXiv 2102.04939](https://www.alphaxiv.org/abs/2102.04939) | FT | index m of 1 of M MDPs, fixed per episode | discrete flat index | Bayes belief / EM / PSR clustering; deterministic case by distinguishing observations | no (T_m learned); known-model planning cited as a sub-problem | Ω((SA)^M) lower bound; regret O(HS√(MAN)) under separation | **PARCIAL** (strong on F1, F4; F5 only in the deterministic appendix; not F2, F3, F6) |
| 3 | Hallak, Di Castro & Mannor, "Contextual Markov Decision Processes", arXiv 2015 — [1502.02259](https://www.alphaxiv.org/abs/1502.02259) | FT | context c ∈ C, \|C\|=K small, static | discrete flat | clustering + L1 classification | no | regret under separation D | **PARCIAL** |
| 4 | Doshi-Velez & Konidaris, "Hidden Parameter MDPs…", IJCAI 2016 — [IJCAI](https://www.ijcai.org/Abstract/16/206) | Abs + EN | θ, fixed per task | continuous latent vector | Bayesian (IBP + GP) | no (dynamics learned) | none | **PARCIAL** |
| 5 | Killian, Daulton, Konidaris & Doshi-Velez, NeurIPS 2017 — [NeurIPS](https://proceedings.neurips.cc/paper/2017/hash/2227d753dc18505031869d44673728e2-Abstract.html) | Abs | θ (joint with state uncertainty) | continuous | Bayesian NN | no | none | **PARCIAL** |
| 6 | Fu, Yao, Gottesman, Doshi-Velez & Konidaris, "Performance Bounds for Model and Policy Transfer in HiP-MDPs", ICLR 2023 — [PDF](https://cs.brown.edu/people/gdk/pubs/hipmdp_bounds.pdf) | FT | θ | continuous, Lipschitz | context encoder (abstracted) | assumed accurately learned | regret linear in ε, via Lipschitz | **DISTANTE** (the theory is inapplicable to Boolean facts) |
| 7 | Modi, Jiang, Tewari & Singh, "MDPs with Continuous Side Information", ALT 2018 — [PMLR](https://proceedings.mlr.press/v83/modi18a.html) | Abs | MDP parameters as a smooth function of an *observed* context | continuous | KWIK (linear case) | no | PAC bounds | **DISTANTE** (E4 boundary) |
| 8 | Zintgraf et al., "VariBAD", ICLR 2020 — [arXiv 1910.08348](https://www.alphaxiv.org/abs/1910.08348) | FT | task (goal, dynamics) | learned continuous latent m | amortised variational posterior (GRU) | no | none ("lacks formal guarantees") | **PARCIAL** (F1, F4; also the canonical RL baseline family) |
| 9 | Guez, Silver & Dayan, BAMCP, NeurIPS 2012 / JAIR 2013 — [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2012/file/35051070e572e47d2c26c241ab88307f-Paper.pdf) | Abs | T (and R), constant per task | any parametric prior | exact Bayes plus MCTS | prior over model class | converges to the Bayes-optimal policy | **PARCIAL** |
| 10 | Duff & Barto 2002 / Martin 1967, BAMDP (via VariBAD) | Abs (secondary) | T, R | belief hyper-state | Bayes | prior | Bayes-optimality definition | **PARCIAL** |
| 11 | Ong, Png, Hsu & Lee, MOMDP, RSS 2009 / IJRR 2010 — [RSS PDF](https://adacomp.comp.nus.edu.sg/wp-content/uploads/2016/01/rss09.pdf) | Abs | partially observable state factors (may be static) | factored discrete | belief over the hidden factors | yes (model known) | point-based solver | **PARCIAL** (F1, F3; factored like Π⁺/Π⁻ but probabilistic) |
| 12 | Ghosh et al., "Epistemic POMDPs and Implicit Partial Observability", NeurIPS 2021 — [arXiv 2107.06277](https://arxiv.org/pdf/2107.06277) | Abs | which training-consistent MDP is true | implicit | ensemble posterior (LEEP) | no | suboptimality results for Markov policies | **PARCIAL** (conceptual; justifies memory baselines) |
| 13 | Kwon et al., "RL in Latent MDPs is Tractable…", 2024 — [arXiv 2406.01389](https://www.alphaxiv.org/abs/2406.01389) | Abs | latent context fixed in an interaction | discrete | off-policy-evaluation-based | no | online guarantees | **PARCIAL** |
| 14 | Chen et al., POMDP-lite (2016) — [arXiv 1602.04875](https://arxiv.org/pdf/1602.04875) | Abs | hidden parameter, static or deterministic | discrete/parametric | belief + reward bonus | yes | near-Bayes-optimal (snippet) | **PARCIAL** |
| 15 | Costen et al., "Planning with HiP Polynomial MDPs", AAAI 2023 — [PDF](https://www.robots.ox.ac.uk/~nickh/papers/costen23aaai.pdf) | Abs | θ | continuous polynomial | belief | yes (polynomial form) | n/a | **DISTANTE** |
| 16 | Diuk, Cohen & Littman, "An Object-Oriented Representation for Efficient RL" (DOORMAX), ICML 2008 — [mlanthology](https://mlanthology.org/icml/2008/diuk2008icml-object) | Abs + restatement | condition→effect rules | relational/propositional over classes | logical intersection (elimination) | no (rules learned); map observed | KWIK / polynomial sample complexity | **PARCIAL** (F5-like inference, inverted target) |
| 17 | Marom & Rosman, "Zero-Shot Transfer with Deictic OO Representation", NeurIPS 2018 — [PDF](https://papers.NeurIPS.cc/paper_files/paper/2018/file/df0aab058ce179e4f7ab135ed4e641a9-Paper.pdf) | FT | schema rules | deictic predicates | logical elimination (DOORMAX_D) | no; walls observed and varied per task | KWIK bound (Thm 1) | **PARCIAL** |
| 18 | Walsh & Littman, "Efficient Learning of Action Schemas and Web-Service Descriptions", AAAI 2008 — [mlanthology](https://mlanthology.org/aaai/2008/walsh2008aaai-efficient) | Abs | STRIPS preconditions/effects | lifted STRIPS | elimination, KWIK-style | no | efficient if precondition size is bounded | **PARCIAL** |
| 19 | Lang, Toussaint & Kersting, "Exploration in Relational Domains for Model-based RL" (REX), JMLR 2012 — [JMLR](https://jmlr.org/papers/v13/lang12a.html) | FT | NID rules | relational probabilistic rules | heuristic ILP learner + relational counts | vocabulary given, rules not | PAC-MDP conditional (Cor. 4) | **PARCIAL** (lower end) |
| 20 | Feng, Kumar, Lozano-Pérez & Kaelbling, "Guided Exploration for Efficient Relational Model Learning", 2025 — [arXiv 2502.06146](https://www.alphaxiv.org/abs/2502.06146) | FT | lifted operator preconditions/effects | PDDL-like | counterexample-driven refinement (oracle-guided) | predicates given, operators not | none | **PARCIAL** (lower end) |
| 21 | Kansky et al., "Schema Networks", ICML 2017 — [PDF](https://www.vicarious.com/wp-content/uploads/2020/03/Paper-ICML2017.pdf) | Abs | entity-attribute schemas | factor-graph fragments | learned (structure learning) | no | none | **DISTANTE** |
| 22 | Džeroski, De Raedt & Driessens, "Relational Reinforcement Learning", MLJ 2001 — [mlanthology](https://mlanthology.org/mlj/2001/dzeroski2001mlj-relational) | Abs | Q-function / policy (model unknown) | relational | regression trees (TILDE) | no | none | **DISTANTE** |

Also rated in `rs_modelo_de_interacao.md`, and not re-rated here: PaRL (Lee et al. 2022/23) at PARCIAL, and PO-PaRL (Rozek et al. 2024) at PARCIAL.

### Inferences
- **Which studies treat the hidden parameter as discrete relational facts, identified exactly (not as a posterior), with templates known?** None of the 22. The nearest partial matches are:
  - (a) Kwon et al.'s deterministic-LMDP algorithm: exact, but over a flat context index with unknown models;
  - (b) MOMDP and POMDP-lite: known model and factored hidden variables, but probabilistic belief;
  - (c) DOORMAX and KWIK-STRIPS: exact elimination and relational, but over rules rather than facts, with the facts observed;
  - (d) James et al. 2020: per-task structure relearned with lifted rules transferred, but count-based, with learned symbols.

  D2's novelty is therefore the **conjunction** F2 + F3 + F5 + F6 in an RL-comparable setting. No single feature is new on its own.
- **Recommended wording for D2.** "We instantiate a latent/contextual MDP (Hallak et al. 2015; Kwon et al. 2021) whose context space is the set of PDDL static-fact assignments and whose per-context models are given by known lifted templates. Under determinism and truthful observation, the Bayesian belief over contexts collapses to the consistent set, which we represent in factored form (Π⁺, Π⁻) and update by abduction. HiP-MDP/VariBAD-style learned latent inference (and its model-free proxies PPO-GRU/DRQN) is the RL baseline."
  This is accurate, and it turns the strongest potential objection into the positioning.

### Gaps
- Studies 4, 5, 9, 11–16, 18, 21 and 22 were rated from abstracts or secondary restatements.
- I could not check whether VariBAD-style methods have been evaluated on PDDLGym-like relational task distributions; none was found.

---

## Q5. Limitations

### Takeaway
The review is broad over the canonical formalisms, but shallow on forward citations and on planning-side hidden-model MDP work.

### Cited Findings
- The icml.cc host was blocked (DNS ENOTFOUND), so the DOORMAX primary PDF could not be read. Its characterisation relies on the full-text restatement in [Marom & Rosman 2018](https://papers.NeurIPS.cc/paper_files/paper/2018/file/df0aab058ce179e4f7ab135ed4e641a9-Paper.pdf).
- The two Killian et al. papers (AAAI 2017 without Daulton; NeurIPS 2017 with Daulton) are easy to conflate. The author order also varies between sources — [NeurIPS-17](https://proceedings.neurips.cc/paper/2017/hash/2227d753dc18505031869d44673728e2-Abstract.html)

### Inferences
- **Selection bias.** Searches were seeded by named formalisms, which favours canonical work. Work on hidden graph or topology exploration (Canadian Traveller-style RL, unknown-map navigation as an LMDP) may be under-sampled; parts of it are covered in `rs_fatos_estaticos.md`.
- **Single screener.** All screening was done by one reviewer, with no second-rater agreement check.
- **Hit counts.** For WebSearch, hit counts are the number of links returned (about 10 per query), not database totals.
- **Verdict subjectivity.** The verdicts rest on the F1–F6 decomposition I defined. A different decomposition, for example weighting F1 heavily, would raise LMDP and CMDP to SOBREPOSIÇÃO FORTE.

### Gaps
- **To do next:**
  - Google Scholar "cited by" for Kwon et al. 2021, Marom & Rosman 2018 and Hallak et al. 2015, filtered by "relational", "symbolic", "PDDL", "static";
  - read Chadès et al. AAAI 2012 and Steimle et al. on multi-model MDPs, the known-model LMDP planning branch;
  - read Diuk's 2010 thesis (KOOL) and Walsh's 2010 thesis, on KWIK for relational models.
