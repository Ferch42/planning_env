# Systematic review: learning an instance's static facts online when the lifted action templates are known

Review date: 2026-10-09. Reviewer: one research agent working alone. This is a systematic re-run of an earlier search that had no protocol (`../Diferenciais da tese revisão bibliográfica/fatos_estaticos.md`).

Labels used in this document:
- **[read: full text]**: I read the full text in this session.
- **[read: abstract/snippet]**: I read only search snippets or an abstract.
- **[prior review]**: The URL comes from the earlier non-systematic review and I did not re-fetch it here.
- **[from memory - unverified]**: Background knowledge that I did not confirm with a source in this session.

---

## Q1. Protocol, databases and the search log

### Takeaway
The protocol was fixed before searching. I ran 15 search strings (13 web, 2 alphaXiv) and 2 Consensus queries; both Consensus queries failed. The Semantic Scholar, OpenAlex and DBLP APIs were blocked by the network proxy. As a result, forward-citation snowballing was partial, done through web search instead of citation indexes.

### Protocol (written before searching)
- **Research question (RQ).** Is there prior work in which an agent (i) knows the lifted action templates (PDDL/STRIPS-like schemas, or portable operators), (ii) does not know the static facts of the current instance (relations no operator changes, such as `connected` or `adjacent`), and (iii) infers those facts online from observed transitions? If so, does it also:
  - (a) separate uncertainty over static facts (monotonic) from uncertainty over fluents?
  - (b) compile the templates into update or abduction rules, with identifiability conditions?
  - (c) treat static facts as the symbolic image of environment hyperparameters θ (as in HiP-MDP, contextual MDPs or procedural generation), with templates transferring and facts relearned per environment?
- **Sub-questions.**
  - SQ1: Is there any work that fixes the schemas and learns instance facts online?
  - SQ2: Is there any work that equates PDDL static predicates with latent context or hidden parameters?
  - SQ3: Are there symbolic agents in MiniGrid, Crafter, XLand or Minecraft that infer an unknown topology?
- **Time window.** 1990–2026.
- **Inclusion criteria.**
  - IC1: Peer-reviewed paper or arXiv preprint in AI planning, RL or KR.
  - IC2: The agent or learner reasons about (part of) an instance or task structure that is unknown, or about action models where static relations are explicitly treated.
  - IC3: At least one of (a), (b), (c) is addressed, even partially.
  - IC4: Text in English.
- **Exclusion criteria.**
  - EC1: Pure continuous motion planning or SLAM with no symbolic layer.
  - EC2: LLM-only planners with no explicit model of instance facts.
  - EC3: Patents, course slides, or secondary aggregators used as the only source. These may be logged but are not included.
  - EC4: No identifiable bibliographic record.
- **Tools actually used.**
  - WebSearch (general web; `extended` mode for most queries).
  - alphaXiv `discover_papers` and `get_paper_content`, used for full text.
  - Consensus `search`: attempted, failed (monthly quota exhausted, then rate-limited).
  - Semantic Scholar Graph API, OpenAlex API and DBLP API: attempted through curl/WebFetch, blocked (proxy 403 / ENOTFOUND).
  - arxiv.org through WebFetch: blocked (ENOTFOUND). Full text was obtained through alphaXiv instead.
- **Venues targeted through queries.** ICAPS, IJCAI, AAAI, JAIR, AIJ, KR, NeurIPS, ICML, ICLR. No venue-specific database was queried directly, because the DBLP API was blocked.

### Search log
"Results" is the number of records the tool returned. "Screened" is the number of titles and snippets I read. "Retained" is the number passed to eligibility.

| # | Query string (exactly as typed) | Tool | Results | Screened | Retained |
|---|---|---|---|---|---|
| S1 | `learning unknown static facts (e.g., connectivity/topology) of a planning instance online when the lifted action schemas are known; planning with incomplete initial state; abduction from observed transitions` (keywords: static predicates, action schemas, incomplete initial state, online learning, planning; prioritize=historical) | alphaXiv discover_papers | 12 | 12 | 6 (2607.27287, 2605.13282, 2508.21449, 2404.09631, 2411.14995, 2607.06501) |
| S2 | `symbolic planner agent in MiniGrid, Crafter, or Minecraft that infers unknown map topology / object facts online while action models are given; procedurally generated environments as hidden parameters` (keywords: MiniGrid, Crafter, Minecraft, PDDL, symbolic planning, exploration) | alphaXiv discover_papers | 10 | 10 | 4 (2602.11468, 2607.06501, 2510.12088, 2603.11351) |
| S3 | `learning static predicates planning unknown instance online` | WebSearch | 10 | 10 | 5 (OLAM IJCAI-21, Bonet&Geffner ECAI-20, Lamanna AAAI-23, OGAMUS 2112.10007, Occhipinti 2204.11902) |
| S4 | `planning with incomplete initial state static facts sensing replanning` | WebSearch | 9 | 9 | 1 (Brafman & Shani SDR, 1401.6048) |
| S5 | `action model learning static relations LOCM static predicates` | WebSearch | 9 | 9 | 4 (LOP ICAPS-15 / IJCAI-16, Aineto AIJ-19, STRIPS Action Discovery 2001.11457, 2402.10726) |
| S6 | `"hidden parameter" MDP symbolic planning relational latent context PDDL` | WebSearch | 9 | 9 | 4 (HiP-MDP 1308.3513, GHP-MDP 2002.03072, HiP polynomial MDP AAAI-23, Latent MDPs 2102.04939) |
| S7 | `abduction preconditions observed transitions infer unknown initial state facts known action model` | WebSearch | 10 | 10 | 1 (PIE-APT 2607.27287; the rest were patents, textbook or irrelevant, EC3) |
| S8 | `identifiability action model learning unique explanation observations planning` | WebSearch | 9 | 9 | 2 (Bolander et al. via secondary source, then verified in S15; 2108.09586 screened out) |
| S9 | `learning unknown static facts of planning instance online with known action schemas` | Consensus search | 0 (error: monthly quota exhausted) | 0 | 0 |
| S10 | `online learning of planning instance topology unknown map symbolic planner` | Consensus search | 0 (error: rate limit) | 0 | 0 |
| S11 | `Learning Portable Representations for High-Level Planning James Rosman Konidaris problem-specific agent-space` | WebSearch (targeted lookup of a candidate recalled while screening S6) | 9 | 9 | 1 (James et al. ICML 2020) |
| S12 | `symbolic planner MiniGrid PDDL unknown map exploration replanning` | WebSearch | 9 | 9 | 4 (Sreedharan & Katz NeurIPS-23, SPOTTER 2012.13037, DANLI 2210.12485, SPCA MAKE 2025) |
| S13 | `"static predicates" generalized planning unknown instance transfer procedurally generated environments` | WebSearch | 9 | 9 | 2 (Jiménez et al. KER review; 2201.03199 virtual actions, used for definitions only) |
| S14 | `Minecraft Crafter PDDL symbolic planning agent infer unknown map facts online exploration` | WebSearch | 9 | 9 | 2 (OneLife 2510.12088, LLM-DP Dagan et al.) |
| S15 | `papers citing "Domain Model Acquisition in the Presence of Static Relations" OR "Efficient, Safe, and Probably Approximately Complete Learning of Action Models" static facts` | WebSearch (forward-citation substitute) | 18 (2 result sets) | 18 | 4 (noisy traces ICAPS-24, conditional effects ICAPS-24, numeric SAM 2312.10705, 2402.10726) |
| S16 | `Bolander Gierasimczuk "Learning Actions Models: Qualitative Approach" finite identifiability` | WebSearch | 9 | 9 | 2 (LORI 2015 arXiv 1507.04285; JLC 2018) |
| S17 | `Bayes-adaptive relational planning unknown relations objects belief over static structure PDDL POMDP` | WebSearch | 18 (2 result sets) | 18 | 4 (BAPOMDP Ross et al., HYPE MLJ-17, "Abstract planning with unknown object quantities and properties", Seeing-is-Believing 2504.03245) |
| — | Semantic Scholar `paper/search/match` for 5 seeds; OpenAlex `works?search=`; DBLP `publ/api?q=static predicates learning` | curl / WebFetch | blocked (403 / ENOTFOUND) | — | — |

Distinct strings typed: 17. Of these, 15 executed successfully and 2 (Consensus) failed.

### Selection flow
These are approximate counts. Deduplication was done by hand by title.
- **Identified** through database search: 159 records (sum of the "Results" column). Snowballing added about 12 more, so the total is about 171.
- **Duplicates removed:** about 27. For example, OGAMUS appeared 4 times, LOP 4 times (two venues), 2411.14995 3 times, and HiP-MDP twice.
- **Screened** (title and snippet): about 144.
- **Excluded at screening:** about 94. Reasons: EC1 (robot navigation, motion planning); EC2 (LLM planners); EC3 (patents, AIMA chapter, Wikipedia, emergentmind); off-topic.
- **Assessed for eligibility:** 50 (abstract or snippet level).
- **Full text read in this session:**
  - Lamanna et al. 2021 (OGAMUS): full extracted text.
  - James, Rosman & Konidaris 2020: full text.
  - Occhipinti, Bonet & Geffner 2022: targeted reading of the "static" passages and references only.
- **Included in synthesis:** 34 (table in Q3). 15 of these are seeds or items carried over from the prior review.

### Gaps
- Consensus was unavailable, and the Semantic Scholar, OpenAlex and DBLP APIs were blocked. Proper forward-citation counts and lists could not be obtained, so snowballing is weaker than a full systematic review requires.
- "Results" counts are whatever the web search engine returned, about 10 per page. They are not database hit counts, so recall is not measurable.

---

## Q2. Snowballing from the seeds

### Takeaway
Backward snowballing from James et al. 2020 produced the most important missed lineage: portable or relocatable action models with per-task grounding, which is the closest prior art for claim (c). Forward snowballing from LOP and SAM found only offline action-model learners. None of them learns instance facts online.

### Cited Findings
- **Seed 1: Bonet & Geffner, IJCAI 2011, K-replanner** [prior review]. [PDF](https://www-i6.informatik.rwth-aachen.de/~hector.geffner/www.dtic.upf.edu/~hgeffner/blai-ijcai11.pdf).
  - Backward: CLG and SDR lineage. SDR was retained via S4: [Brafman & Shani, "Replanning in Domains with Partial Information and Sensing Actions", JAIR 2012](https://arxiv.org/pdf/1401.6048).
  - Forward: the citation index was blocked. No new online static-fact learner was found through the web.
  - **Retained:** SDR.
- **Seed 2: Stern & Juba, IJCAI 2017, SAM.** [arXiv 1705.08961](https://arxiv.org/pdf/1705.08961); [BibTeX](https://www.ijcai.org/proceedings/2017/bibtex/615).
  - Forward, via S15: ["Action model learning from noisy traces", ICAPS 2024](https://dl.acm.org/doi/10.1609/icaps.v34i1.31493); ["Safe learning of PDDL domains with conditional effects", ICAPS 2024](https://dl.acm.org/doi/10.1609/icaps.v34i1.31498); ["Learning Safe Numeric Planning Action Models"](https://arxiv.org/pdf/2312.10705).
  - Semantic Scholar shows 34 citations ([S2 page](https://www.semanticscholar.org/paper/Efficient-%2C-Safe-%2C-and-Probably-Approximately-of-Gurion/7d01801aae7f107501d7f15fcdf4698b4feee016), seen only as a snippet). WashU/Scopus reports 22 ([profile](https://profiles.wustl.edu/en/publications/efficient-safe-and-probably-approximately-complete-learning-of-ac)). The two counts disagree.
  - **Retained:** these 3, as context. All learn schemas, not instance facts.
- **Seed 3: Gregory & Cresswell, ICAPS 2015, LOP.** [ICAPS](https://ojs.aaai.org/index.php/ICAPS/article/view/13729); IJCAI 2016 version: [PDF](https://www.ijcai.org/Proceedings/16/Papers/622.pdf).
  - Forward: Semantic Scholar lists 40 citations ([S2](https://www.semanticscholar.org/paper/Domain-Model-Acquisition-in-the-Presence-of-Static-Gregory-Cresswell/cdf50f42fb4cc4fb1afd41813a7192babd005d88), snippet only); [2402.10726](https://arxiv.org/html/2402.10726v2); [Springer chapter on domain-learning tools](https://link.springer.com/chapter/10.1007/978-3-030-38561-3_2).
  - Backward via the Occhipinti references: *Lindsay, "Reuniting the LOCM family: An alternative method for identifying static relationships", ICAPS 2021 KEPS Workshop* (found in the reference list of [2204.11902](https://www.alphaxiv.org/abs/2204.11902); paper not opened).
  - **Retained:** Lindsay 2021 and 2402.10726.
- **Seed 4: Doshi-Velez & Konidaris, HiP-MDP.** [arXiv 1308.3513](https://arxiv.org/pdf/1308.3513), IJCAI 2016.
  - Forward via S6: [Perez et al., "Generalized Hidden Parameter MDPs", AAAI 2020](https://arxiv.org/pdf/2002.03072); ["Planning with Hidden Parameter Polynomial MDPs", AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/26411); [Kwon et al., "RL for Latent MDPs"](https://arxiv.org/pdf/2102.04939).
  - Lateral, a Konidaris-group paper: [James, Rosman & Konidaris, "Learning Portable Representations for High-Level Planning", ICML 2020](https://proceedings.mlr.press/v119/james20a.html) [read: full text, arXiv 1905.12006 via alphaXiv].
  - **Retained:** all 4.
- **Seed 5 (added): Lamanna et al., OGAMUS**, [arXiv 2112.10007](https://www.alphaxiv.org/abs/2112.10007) [read: full text].
  - Backward: the references are mostly embodied-navigation RL (Chaplot et al. Active Neural SLAM; RoboTHOR; DD-PPO) and anchoring (Coradeschi & Saffiotti 2003). None learns static facts by abduction.
  - **Retained:** none new. Same-group work retained via S3: [OLAM, IJCAI 2021](https://www.ijcai.org/proceedings/2021/0566.pdf) and ["Planning for Learning Object Properties", AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/download/26416/26188).
- **Backward from James et al. 2020** (reference list read in full):
  - Relocatable action models: *Sherstov & Stone, AAAI 2005*; *Leffler, Littman & Edmunds, "Efficient reinforcement learning with relocatable action models", AAAI 2007*.
  - *Zhang et al., "Composable planning with attributes", ICML 2018*.
  - *Konidaris, Kaelbling & Lozano-Pérez, "From skills to symbols", JAIR 61, 2018*.
  - *Pasula, Zettlemoyer & Kaelbling, ICAPS 2004*.
  - *Andersen & Konidaris, NeurIPS 2017*.
  - Source for all of these: [James et al. full text](https://arxiv.org/abs/1905.12006).
  - **Retained:** Leffler et al. 2007 and Konidaris et al. 2018, at reference level only.

### Inferences
- The relocatable / portable-model line (Sherstov & Stone 2005 → Leffler et al. 2007 → Konidaris et al. 2012/2018 → James et al. 2020) did not appear in the earlier search. It is the strongest prior art for claim (c): lifted rules transfer and task-specific parameters are relearned. It needs to be discussed explicitly in the thesis.

### Gaps
- Forward citations of K-replanner, HiP-MDP and James et al. 2020 could not be enumerated, because citation APIs were blocked. Recommendation: rerun on Google Scholar or Semantic Scholar "cited by", filtering for "static", "portable", "topology" and "unknown map".

---

## Q3. Included studies and overlap verdicts per claim

### Takeaway
I found no work that does all of the following: fixes lifted PDDL templates; treats static predicates as a separate, monotonic layer of epistemic uncertainty; compiles the templates into abductive update rules for those facts with identifiability conditions; and frames those facts as the symbolic image of θ.
- **Claim (c)** has the strongest prior art: James et al. 2020 (portable lifted rules plus per-task "linking functions" learned from observed start and end partitions), with HiP-MDP and GHP-MDP on the RL side.
- **Claim (b)** has only partial analogues:
  - LOP infers static preconditions offline, from plan optimality.
  - Bolander & Gierasimczuk give finite identifiability, but for action models, not instance facts.
  - James et al. learn task links by counting, not by abduction.
- **Claim (a)** is implicit in the unknown-map, CTP and contingent-planning literature, but nowhere is it made explicit as "static versus fluent".

### Included studies
Overlap ratings: **strong**, **partial**, **weak**, **none**.

| # | Study | Read | (a) static vs fluent | (b) templates → update/abduction rules, identifiability | (c) static facts = θ, templates transfer | Justification |
|---|---|---|---|---|---|---|
| 1 | James, Rosman & Konidaris, "Learning Portable Representations for High-Level Planning", ICML 2020. [PMLR](https://proceedings.mlr.press/v119/james20a.html), [arXiv 1905.12006](https://www.alphaxiv.org/abs/1905.12006) | full text | partial | partial | **strong** | Lifted rules over an egocentric space transfer across tasks; "the only time problem-specific information is required is to determine the values of X", the partition labels. Per-task linking functions are learned "by simply executing options and recording the start and end partition labels of each transition", using a count-based method. Tasks differ in block placement or level layout (procedural variation). Not abduction over named static predicates, no identifiability analysis, and the symbols are learned rather than given as PDDL. |
| 2 | Lamanna, Serafini, Saetti, Gerevini & Traverso, "Online Grounding of Symbolic Planning Domains in Unknown Environments" (OGAMUS), [arXiv 2112.10007](https://www.alphaxiv.org/abs/2112.10007) | full text | partial | weak | none | A known PDDL domain is incrementally instantiated "by planning, acting, and sensing, in an unknown environment". The agent discovers objects, properties and relations through perception, with a belief made of known objects plus an occupancy map. Facts come from direct sensing, not abduction from transitions. No static/fluent distinction. |
| 3 | Lamanna et al., "Online Learning of Action Models for PDDL Planning" (OLAM), IJCAI 2021. [PDF](https://www.ijcai.org/proceedings/2021/0566.pdf) | snippet | weak | weak (inverse) | none | Learns schemas online, the inverse of the thesis setting. It notes that extra learned preconditions that are always true are static predicates. |
| 4 | Lamanna et al., "Planning for Learning Object Properties", AAAI 2023. [PDF](https://ojs.aaai.org/index.php/AAAI/article/download/26416/26188) | title/snippet only | partial? | ? | none | Plans in order to learn object properties. Details not verified, so it needs a full read. |
| 5 | Occhipinti, Bonet & Geffner, "Learning First-Order Symbolic Planning Representations That Are Grounded", [arXiv 2204.11902](https://www.alphaxiv.org/abs/2204.11902) (venue not verified; KR 2022 [from memory - unverified]) | targeted full-text passages | partial | weak | partial | "for each instance I, we compute the predicates that are static over that instance … and encode their truth value V in all states of that instance". So static facts are explicitly per-instance while the domain is shared. The learning is offline and batch, from labelled state graphs. |
| 6 | Bonet & Geffner, "Learning First-Order Symbolic Representations for Planning from the Structure of the State Space", ECAI 2020. [PDF](https://ecai2020.eu/papers/894_paper.pdf) | snippet | weak | weak | partial | Infers instances over a common unknown first-order domain. The number of static predicates is a search hyperparameter. Offline. |
| 7 | Gregory & Cresswell, "Domain Model Acquisition in the Presence of Static Relations in the LOP System", ICAPS 2015. [ICAPS](https://ojs.aaai.org/index.php/ICAPS/article/view/13729); IJCAI 2016 version [PDF](https://www.ijcai.org/Proceedings/16/Papers/622.pdf) | snippet + prior review | weak | partial | none | Static predicates are treated as restrictions on valid groundings. LOP finds a minimal static precondition per operator that preserves optimal plan length. Learning is offline and domain-level, from optimal plans, not online and instance-level. |
| 8 | Lindsay, "Reuniting the LOCM family: An alternative method for identifying static relationships", ICAPS 2021 KEPS Workshop | reference only | weak | partial | none | Identified via the references of 2204.11902; not opened. |
| 9 | Gösgens, Jansen & Geffner (authorship [from memory - unverified]), "Learning Lifted STRIPS Models from Action Traces Alone", [arXiv 2411.14995](https://arxiv.org/pdf/2411.14995); ICAPS version [PDF](https://ojs.aaai.org/index.php/ICAPS/article/download/36117/38271/40190) | snippet | none | weak | none | Completes learned domains with preconditions and static predicates; notes that learning them is "not strictly necessary". |
| 10 | Stern & Juba, SAM, IJCAI 2017. [arXiv 1705.08961](https://arxiv.org/pdf/1705.08961) | prior review + snippet | none | partial (safety/PAC, not identifiability of facts) | none | Conservative, safe learning of schemas from successful plans. |
| 11 | Safe learning with conditional effects, ICAPS 2024 [ACM](https://dl.acm.org/doi/10.1609/icaps.v34i1.31498); noisy traces, ICAPS 2024 [ACM](https://dl.acm.org/doi/10.1609/icaps.v34i1.31493); numeric SAM [arXiv 2312.10705](https://arxiv.org/pdf/2312.10705) | title/snippet | none | weak | none | Forward citations of SAM; all learn schemas. |
| 12 | Aineto, Jiménez & Onaindia, "Learning action models with minimal observability", AIJ 2019. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0004370218304259) | snippet | none | weak | none | Compiles learning into planning, a schema-level analogue of "compiling into rules". |
| 13 | Bolander & Gierasimczuk, "Learning Actions Models: Qualitative Approach", LORI 2015 [arXiv 1507.04285](https://www.arxiv.org/abs/1507.04285); JLC 28(2) 2018 [PDF](https://people.compute.dtu.dk/tobo/bolander2018learning.pdf) | snippet | none | **partial** | none | Deterministic, fully observable propositional actions are *finitely identifiable*; non-deterministic actions are only identifiable in the limit. This is the closest formal precedent for the identifiability part of (b), but it concerns the action model, not instance facts. |
| 14 | "Action Model Learning with Guarantees" (version spaces), [arXiv 2404.09631](https://www.alphaxiv.org/abs/2404.09631) | snippet | none | partial | none | Version-space theory for action-model learning under full observability. Relevant to the unique-explanation conditions in (b). |
| 15 | Bonet & Geffner, "Planning under Partial Observability by Classical Replanning" (K-replanner), IJCAI 2011. [PDF](https://www-i6.informatik.rwth-aachen.de/~hector.geffner/www.dtic.upf.edu/~hgeffner/blai-ijcai11.pdf) | prior review | partial | weak | none | Unknown initial facts are handled through K-literals and optimistic replanning. Learned by sensing, not abduction. No static/fluent split. |
| 16 | Brafman & Shani, "Replanning in Domains with Partial Information and Sensing Actions" (SDR), JAIR 2012. [arXiv 1401.6048](https://arxiv.org/pdf/1401.6048) | snippet | partial | partial | none | Belief over initial states, refined when sensing contradicts the sampled state. The regression of observations to the initial state is a mechanism adjacent to abduction [regression detail from memory - unverified]. |
| 17 | Koenig & Smirnov, "Sensor-Based Planning with the Freespace Assumption", ICRA 1997. [CMU](https://www.ri.cmu.edu/publications/sensor-based-planning-with-the-freespace-assumption) | prior review | partial | none | none | The unknown map is static and discovered by sensors under an optimistic default. |
| 18 | Nourbakhsh & Genesereth, "Assumptive Planning and Execution", Autonomous Robots 1996. [CMU](https://ri.cmu.edu/publications/assumptive-planning-and-execution-a-simple-working-robot-architecture) | prior review | partial | none | none | Assumptions about an unknown static environment, revised on execution. |
| 19 | Canadian Traveller Problem (Papadimitriou & Yannakakis 1991 [from memory - unverified]). [Overview](https://en.wikipedia.org/wiki/Canadian_traveller_problem) | prior review | **partial** | none | none | Edge status is static and unknown, revealed on arrival. This is the cleanest analogue of monotonic static-fact uncertainty, but it is graph-specific and not template-based. |
| 20 | Doshi-Velez & Konidaris, HiP-MDP. [arXiv 1308.3513](https://arxiv.org/pdf/1308.3513) | prior review + snippet | partial | none | **partial** | θ is fixed per task ("no dynamics for the duration of the task"), which matches the monotonic static layer, but θ is a continuous vector, not relational facts. |
| 21 | Perez, Such & Karaletsos, "Generalized Hidden Parameter MDPs", AAAI 2020. [arXiv 2002.03072](https://arxiv.org/pdf/2002.03072) | snippet | partial | none | partial | Structured latent factors (agent, environment, goal), but not logical or relational. |
| 22 | "Planning with Hidden Parameter Polynomial MDPs", AAAI 2023. [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/26411) | snippet | partial | weak | partial | Closed-form belief over hidden parameters, an analogue of compiled belief updates, but not symbolic. |
| 23 | Kwon et al., "RL for Latent MDPs: Regret Guarantees and a Lower Bound". [arXiv 2102.04939](https://arxiv.org/pdf/2102.04939) | snippet | partial | weak (identifiability of latent context) | partial | The latent context is drawn once per episode. |
| 24 | Ross, Chaib-draa & Pineau, Bayes-Adaptive POMDPs (NeurIPS 2007; JMLR 2011). [NeurIPS](https://papers.nips.cc/paper/3333-bayes-adaptive-pomdps), [JMLR](https://jmlr.org/papers/volume12/ross11a/ross11a.pdf) | snippet | partial | none | partial | Model parameters are placed in the state and tracked by belief. The state space must be finite and known. |
| 25 | Nitti, Belle & De Raedt (authors [from memory - unverified]), "Planning in hybrid relational MDPs" (HYPE), MLJ 2017. [Springer](https://link.springer.com/article/10.1007/s10994-017-5669-x) | snippet | weak | none | weak | Relational MDPs with unknown objects, using probabilistic programming. |
| 26 | "Abstract Planning with Unknown Object Quantities and Properties" (Srivastava et al., SARA 2009 [from memory - unverified]). [ResearchGate](https://www.researchgate.net/publication/220970625_Abstract_Planning_with_Unknown_Object_Quantities_and_Properties) | snippet | partial | weak | weak | Belief over relational structure in 3-valued logic. |
| 27 | Jiménez, Segovia-Aguas & Jonsson, "A review of generalized planning", KER 2019. [PDF](https://serjice.webs.upv.es/publications/sergio-ker18/sergio-ker18.pdf) | prior review + snippet | none | none | partial | Shared predicates and schemas across instances, with per-instance unification. No online learning of facts. |
| 28 | Sreedharan & Katz, "Optimistic Exploration in RL Using Symbolic Model Estimates", NeurIPS 2023. [PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/6cbd0a1251f41b41aa68e728bcc1ee40-Paper-Conference.pdf) | snippet | weak | weak | weak | PDDL models for MiniGrid with optimistic symbolic estimates. The model, not the instance topology, is estimated [scope from snippet only]. |
| 29 | Sarathy et al., SPOTTER, [arXiv 2012.13037](https://arxiv.org/pdf/2012.13037) | snippet | none | none | none | MiniGrid plus PDDL; learns new operators by RL. |
| 30 | Zhang et al., DANLI, [arXiv 2210.12485](https://arxiv.org/pdf/2210.12485) | snippet | weak | none | none | The planner assumes unseen objects exist and searches for them. |
| 31 | Khan et al. (authors [unverified]), OneLife, [arXiv 2510.12088](https://arxiv.org/html/2510.12088v1) | snippet | none | none | weak | Crafter-OO; the world model is a Python program learned from exploration. PDDL is explicitly rejected for stochastic dynamics. |
| 32 | "Hypothesis-driven Model Expansion under Uncertainty for Open-World Robot Planning", [arXiv 2607.06501](https://www.alphaxiv.org/abs/2607.06501); "Effective Task Planning with Missing Objects…", [arXiv 2602.11468](https://www.alphaxiv.org/abs/2602.11468) | abstract | partial | weak | none | Open-world PDDL with unknown objects or actions, handled by hypotheses or learned search. |
| 33 | PIE-APT, "Abductive Planning over Temporal Dynamic Knowledge Graphs via Incremental Reasoning", [arXiv 2607.27287](https://arxiv.org/pdf/2607.27287) | abstract/snippet | weak | partial | none | Distinguishes pure abductive explanation from actions combined with assumptions. Not instance static facts. |
| 34 | Dagan et al., LLM-DP. [PDF](https://homepages.inf.ed.ac.uk/alex/papers/langame.pdf) | snippet | weak | none | none | An LLM samples plausible beliefs about unknown objects for a PDDL planner (ALFWorld). |

### Inferences
- **(a) static vs fluent.** No included study names this separation as a contribution. The CTP, free-space and HiP-MDP lines all assume a per-task fixed unknown, which is the monotonic case. The novelty is therefore in *making it explicit at the PDDL level*, where the static versus fluent distinction is syntactically decidable from the templates. The novelty is not in the idea of fixed unknowns itself.
- **(b) compilation and identifiability.** This is the most defensible differential.
  - Existing identifiability results (Bolander & Gierasimczuk; version spaces) concern schemas.
  - LOP's inference is offline and based on optimality.
  - James et al. learn task links by counting, without logical explanation or uniqueness conditions.
  - Recommendation: position (b) as the dual of action-model identifiability (schemas known, ground static facts unknown). Cite Bolander & Gierasimczuk and LOP as the closest formal neighbours.
- **(c) static facts = θ.** This claim must be weakened or reframed. James et al. 2020 already give "lifted rules preserved across tasks, with parameters instantiated per task". Occhipinti, Bonet & Geffner already encode static predicates per instance over a shared domain. Possible framing: the explicit equation "PDDL static predicates = HiP-MDP θ", with known (not learned) templates and an abductive update.

### Gaps
- Lamanna et al. AAAI 2023, Sreedharan & Katz NeurIPS 2023, and Lindsay KEPS 2021 were not read beyond the title or snippet. They should be read in full, because they are the most likely to contain partial (b)-type mechanisms.
- I did not verify whether Leffler et al. 2007 (relocatable action models) infers state "types" online, which would overlap with (b) and (c).

---

## Q4. Specific checks

### Takeaway
1. **Schemas fixed, instance facts learned online.** Yes, but only through direct perception or sensing (OGAMUS, K-replanner, SDR, CTP, free-space), or through count-based linking (James et al. 2020). I found no abduction over static predicates from abstract transitions.
2. **PDDL static predicates explicitly equated with latent context or hidden parameters.** I found no such work. S6 and S17 explicitly returned none, and the search engine's own synthesis also reported none.
3. **Symbolic agents in MiniGrid, Crafter, XLand or Minecraft inferring topology.** I found only weak cases: optimistic PDDL estimates in MiniGrid, SPOTTER, an LLM-PDDL MiniGrid pipeline with incremental map building, and OneLife in Crafter without PDDL. Nothing for XLand.

### Cited Findings
- OGAMUS: the belief at each time point is "the set of objects currently known by the agent and their properties expressed with the predicates of the PDDL domain", plus the map. — [arXiv 2112.10007](https://www.alphaxiv.org/abs/2112.10007)
- James et al.: the linking functions "are learned by simply executing options and recording the start and end partition labels of each transition. We use a simple count-based approach". — [arXiv 1905.12006](https://www.alphaxiv.org/abs/1905.12006)
- The SPCA framework (MDPI MAKE 2025) builds a global map incrementally for replanning in a MiniGrid variant, with LLM-generated PDDL. — [MAKE 8(1):22](https://doi.org/10.3390/make8010022) [snippet only]
- OneLife states that PDDL cannot easily capture Crafter-OO's stochastic dynamics. — [arXiv 2510.12088](https://arxiv.org/html/2510.12088v1)
- I found no symbolic-planning agent for DeepMind XLand in any query (S2, S12, S14).

### Inferences
- The procedurally generated benchmark niche, with symbolic templates known and topology abduced per seed, appears open. The closest empirical precedent is James et al. 2020 (Treasure Game levels, Rod-and-Block layouts), which is not built on standard PCG benchmarks.

### Gaps
- No Minecraft-specific symbolic work (for example Polycraft or NovelGym novelty agents) was retrieved. The Tufts hybrid LLM-symbolic novelty paper ([arXiv 2603.11351](https://www.alphaxiv.org/abs/2603.11351)) was only seen as a title. A dedicated query is needed.

---

## Q5. Limitations

### Takeaway
This review is protocol-driven, but it was run under constrained tooling. Its negative conclusions ("no work found") have moderate confidence, not high.

### Cited Findings
- Consensus returned "You've used all 30 searches this month" and then a rate-limit error. The Semantic Scholar, OpenAlex and DBLP APIs returned proxy 403 or ENOTFOUND. This is from the tool logs of this session; there is no external source.

### Inferences
- **Single reviewer**, with no second independent screener. Inclusion decisions may be biased toward the planning community.
- **Web search returns about 10 results per query** and is ranking-dependent. Recall is unknown, and the "Results" counts are not database hit counts.
- **Forward snowballing was approximated** by web search instead of citation indexes. Forward citations of K-replanner, HiP-MDP and James et al. are therefore incomplete.
- **Full text was read for only 3 papers.** Most verdicts are based on abstracts or snippets.
- **Terminology gap.** The same concept goes by "static predicates", "rigid relations", "invariants", "relocatable models", "portable symbols", "task parameters" and "context". Older work, such as rigid fluents in the situation calculus or KR "rigid designators", was not searched explicitly.
- **Some metadata is marked [from memory - unverified]:** venue or authors for Occhipinti 2022, 2411.14995, HYPE, the SARA 2009 paper, and the CTP origin.

### Gaps
- Suggested follow-ups:
  1. Run Google Scholar "cited by" for James et al. 2020, Leffler et al. 2007, and LOP.
  2. Run queries on "rigid predicates online", "relocatable action models", "portable symbols transfer", and "Polycraft PDDL novelty".
  3. Read Lamanna et al. AAAI 2023 and Sreedharan & Katz NeurIPS 2023 in full.
