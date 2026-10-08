# Contingent planning (offline and online) from the 2000s to 2026: planners, techniques, benchmarks, empirical comparisons and implementations

Scope note: these notes cover deterministic or non-deterministic, set-based (non-probabilistic) planning under partial observability with sensing, also called PPOS or contingent planning. POMDP solvers are only mentioned where contingent-planning papers compare against them. Several primary PDFs (ijcai.org, aaai.org, Geffner's site) could not be fetched from this environment. Claims that rest only on abstracts or catalogue pages are marked "(abstract only)". Claims from memory that I could not verify are marked **[UNVERIFIED]**.

---

## 1. Offline planners: which ones exist, what techniques they use, and how they compare

### Takeaway
Offline contingent planning went through three stages. It began with explicit AND/OR search in belief space using symbolic belief representations (MBP with BDDs, Contingent-FF with implicit CNF, POND with BDDs and planning-graph heuristics, and To/Son/Pontelli's DNF, CNF and prime-implicate planners). The next stage translated the problem into a fully observable non-deterministic (FOND) problem or a classical one (CLG, PO-PRP). The most recent stage builds plan trees or graphs by calling an online replanner many times (CPOR, Maliah/Komarnitsky/Shani). Offline plan trees can be exponential in size. Offline planners have handled Wumpus only up to about 7x7, while online planners have reached 20x20, the K-replanner 40x40, and beam tracking more than 100x100.

### Cited Findings

**MBP (Bertoli, Cimatti, Roveri, Traverso; FBK/IRST Trento)**
- IJCAI 2001, "Planning in Nondeterministic Domains under Partial Observability via Symbolic Model Checking", pp. 473–478. The planner searches a possibly cyclic AND-OR graph and returns conditional plans that are guaranteed to reach the goal despite uncertain initial conditions, non-deterministic effects and partial observability. It is implemented with BDD-based symbolic model checking. — [IRIS UniTN](https://iris.unitn.it/handle/11572/258842); [mlanthology](https://mlanthology.org/ijcai/2001/bertoli2001ijcai-planning)
- Journal version: "Strong planning under partial observability", *Artificial Intelligence* 170 (2006) 337–384. It defines strong planning under partial observability and gives an AND-OR search over belief states that always terminates and is correct and complete. The authors combine heuristic distance measures with mechanisms that reduce runtime uncertainty, and report that MBP "often outperforms competing systems by orders of magnitude" (abstract only). — [IRIS UniTN](https://iris.unitn.it/handle/11572/258694); [CERIST](https://biblio.cerist.dz/show/ar/00000000000000597499000000)
- MBP also covers conformant planning and temporally extended goals. For partial observability it produces tree-shaped plans conditioned on the execution history. — [MBP system description, ICAPS 2003](https://icaps03.icaps-conference.org/satellite_events/documents/sd/04/bertoli.pdf)
- A 2003 ICAPS workshop paper extended MBP to interleave planning and execution under partial observability, an early online variant. — [Bertoli, Cimatti, Traverso, ICAPS-03 WS](https://icaps03.icaps-conference.org/satellite_events/documents/WS/WS2/13/Bertoli.pdf)

**Contingent-FF (CFF), Hoffmann & Brafman, ICAPS 2005**
- "Contingent planning via heuristic forward search with implicit belief states", ICAPS 2005, pp. 71–80. It extends Conformant-FF (AIJ 2006). The belief state is represented implicitly by the action and observation history plus the initial-state CNF. SAT queries, cached through unit propagation, decide whether a literal, a precondition or the goal holds. — reference list and §6 of [Brafman & Shani, JAIR 2012](https://arxiv.org/pdf/1401.6048)
- Brafman & Shani compared CFF's belief-maintenance method with SDR's lazy regression on the largest instances CFF could handle. Times in seconds, CFF vs SDR: unix3 427.7 vs 9.6; Wumpus10 688.8 vs 102.1; ebtcs-70 481.3 vs 17.4; doors9 283.9 vs 72.5; localize15 909.1 vs 667.2. — [Brafman & Shani 2012, Table 5](https://arxiv.org/pdf/1401.6048)

**POND, Bryce, Kambhampati & Smith, JAIR 26 (2006)**
- "Planning graph heuristics for belief space search", JAIR 26:35–99. POND does AND/OR (AO*-style) search in belief space, represents beliefs with BDDs, and uses heuristics from planning graphs (labelled uncertainty graph, LUG). — cited in [Brafman & Shani 2012](https://arxiv.org/pdf/1401.6048) as a BDD-based offline planner that produces complete plan trees. The LUG and AO* details come from memory **[UNVERIFIED in a fetched source]**.

**CLG (Closed-Loop Greedy planner), Albore, Palacios & Geffner, IJCAI 2009**
- "A Translation-Based Approach to Contingent Planning", IJCAI 2009, pp. 1623–1628. A contingent problem P is mapped to a non-deterministic problem X(P) in state space, with a K-literal style "knowledge" encoding inherited from Palacios & Geffner's conformant translations. X(P) is solved through a classical relaxation X⁺(P). The authors introduce a contingent width parameter: for bounded width the translation is sound, polynomial and complete. — [mlanthology](https://mlanthology.org/ijcai/2009/albore2009ijcai-translation); [IJCAI abstract](https://www.ijcai.org/Abstract/09/271)
- CLG has an offline mode that builds the full plan and an "execution" (online) mode. Brafman & Shani used the execution mode as the online state-of-the-art baseline. — [Brafman & Shani 2012](https://arxiv.org/pdf/1401.6048)
- CLG cannot translate problems of width greater than 1, for example MasterMind ("TF"). Conditional effects over unknown variables, as in localize, are "the key bottleneck for the CLG translation". — [Brafman & Shani 2012, §7 and Table 9](https://arxiv.org/pdf/1401.6048)

**DNFct / CNFct / PIct, To, Son & Pontelli (NMSU), 2009–2011**
- These planners combine an AND/OR forward search algorithm (PrAO) with different belief-state representations: minimal DNF (DNFct), minimal CNF (CNFct) and prime implicates (PIct). Belief representations compared at AAAI 2011: "all planners outperform other state-of-the-art planners on most benchmarks" (abstract only). — [AAAI 2011, "On the Effectiveness of Belief State Representation in Contingent Planning"](https://mlanthology.org/aaai/2011/to2011aaai-effectiveness)
- IJCAI 2011: CNFct and DNFct are "very competitive … but neither of the two representations is a clear winner". — [mlanthology IJCAI 2011](https://mlanthology.org/ijcai/2011/to2011ijcai-effectiveness)
- AAAI 2011: PIct uses the same AND/OR search and heuristic as CNFct and also performs well. — [AAAI 2011 "Conjunctive Representations in Contingent Planning: Prime Implicates Versus Minimal CNF Formula"](https://mlanthology.org/aaai/2011/to2011aaai-conjunctive)
- Journal synthesis: "A generic approach to planning in the presence of incomplete information: Theory and implementation", appearing as an IJCAI 2017 journal-track extended abstract. The underlying journal is almost certainly AIJ 2015 **[venue/year UNVERIFIED]**. — [IJCAI 2017 abstract](https://ijcai.org/proceedings/2017/725)
- Brafman & Shani summarize To's thesis: "different domains require different representations". — [Brafman & Shani 2012, §6](https://arxiv.org/pdf/1401.6048)
- Brafman & Shani 2012 and Bonet & Geffner 2013/2019 report that offline planners (CLG offline, To et al.) scale on Wumpus only up to n = 7. — [Bonet & Geffner, arXiv 1909.13778 §9](https://arxiv.org/pdf/1909.13778)

**PO-PRP, Muise, Belle & McIlraith, AAAI 2014**
- "Computing Contingent Plans via Fully Observable Non-Deterministic Planning", AAAI 2014, DOI 10.1609/aaai.v28i1.9049. It applies Bonet & Geffner's 2011 K-replanner compilation (PPOS to FOND), where a sensing action becomes a non-deterministic action. The FOND planner PRP (Muise, McIlraith & Beck, ICAPS 2012) is modified to compute strong-cyclic policies, which are then rolled out into DAG-shaped plans. Plans are "orders of magnitude smaller than previously possible in some domains". CLG is used as the offline baseline. — [AAAI page](https://ojs.aaai.org/index.php/AAAI/article/view/9049)
- The PO-PRP wiki states the translation is "always sound, and complete for simple contingent problems". Code is on the `contingent` branch and runs via `src/poprp`. Non-deterministic effects and sensing are supported only in isolation, not together. Dead-end detection is off by default. Later changes in the K-replanner compiler broke the compact regression on some domains, and the binaries used in the paper were not frozen. — [PO-PRP wiki](https://github.com/qumulab/planner-for-relevant-policies/wiki/PO-PRP)

**Komarnitsky & Shani (AAAI 2016), followed by CPOR (Maliah, Komarnitsky & Shani, ACM TAAS 2021/22)**
- "Computing Contingent Plans Using Online Replanning", AAAI 2016, pp. 3159–3165. An online solver (SDR-style) is called repeatedly. Its plan is executed up to the next observation action, the tree branches on each observed value, and the planner replans per branch. This avoids the very large PPOS-to-FOND translations, and the authors report better scaling than offline state-of-the-art planners (abstract only). — [AAAI page](https://ojs.aaai.org/index.php/AAAI/article/view/10406)
- The journal version, "Computing Contingent Plan Graphs using Online Planning", appeared in ACM TAAS 16(1), January 2022 (online 2021). Exponential trees are compressed into directed graphs by merging equivalent nodes, and cycles allow non-deterministic domains (abstract only). — [CRIS BGU](https://cris.bgu.ac.il/en/publications/computing-contingent-plan-graphs-using-online-planning-2/)

**Other offline or related work found**
- BCP, a "Partially observable non-deterministic (POND) planner that finds contingent plans with bounded branching" (Java, K. McAreavey). — [GitHub kevinmcareavey/bcp](https://github.com/kevinmcareavey/bcp)
- cp2fsc (Bonet, Palacios & Geffner) compiles partially observable problems into finite-state controllers (`--fsc-states n`). It is a different kind of offline solution: a controller instead of a plan tree. — [GitHub bonetblai/cp2fsc-and-replanner](https://github.com/bonetblai/cp2fsc-and-replanner)

### Inferences
- In offline planning the bottleneck has shifted from belief representation (BDD, CNF, DNF, PI) to translation size and plan size. PO-PRP compresses through strong-cyclic policies. CPOR compresses through plan graphs and avoids the large translations entirely.
- The warehouse gridworld has many independent hidden facts (item at table k, door states, and so on), so a complete offline plan tree is likely to be exponential in the number of unknown items, as in colorballs, unix or doors. Offline planners are therefore realistic only for small instances or for sub-tasks.

### Gaps
- I could not fetch the full PO-PRP, CLG, CFF, POND or To et al. papers, so their per-domain coverage tables are not reproduced here. The comparison numbers in these notes come mostly from Brafman & Shani 2012 and Bonet & Geffner 2013/2019.
- I found no confirmed entry for "HCP" as an offline planner. HCP is the online Heuristic Contingent Planner (§2).
- I did not find any dedicated offline contingent planner from 2023–2026 beyond the dead-end extensions to CPOR (§2).

---

## 2. Online and replanning approaches, and their completeness guarantees

### Takeaway
Online contingent planners interleave belief tracking with classical planning on an optimistic or sampled determinization. They include CLG in execution mode, SDR, MPSR, the K-replanner, LW1, HCP and beam-tracking agents. Completeness holds only under restrictive conditions. The K-replanner is complete for "simple" problems (static hidden variables, no dead-ends). LW1 is complete for width-1 problems. SDR's guarantee assumes a full belief state and connectivity, meaning no dead-ends. All of them fail or degrade in domains with dead-ends, for example SDR fails completely on Wumpus with lethal cells. This is essentially the same assumption-based ("assumptive") planning as the free-space assumption in robot navigation, whose soundness and completeness also depend on reversibility.

### Cited Findings

**CLG execution mode (Albore, Palacios & Geffner 2009)**
- CLG is the online baseline in Brafman & Shani 2012. It runs the translation and then executes with closed-loop replanning. For domains with conditional effects, such as localize, its simulator could not be used ("CSU"), and it fails (PF) on doors ≥15, localize ≥11 and colorballs 9-5. Translation timeout was 20 min and execution timeout 30 min. — [Brafman & Shani 2012, Table 1](https://arxiv.org/pdf/1401.6048)
- On domains CLG does solve, it often produces shorter plans because it reasons over the complete belief, for example unix4 with 90.8 actions vs 195.8 for SDR. It is slower on larger instances, for example unix4 189 s vs 53 s for SDR. — [Brafman & Shani 2012, Table 2](https://arxiv.org/pdf/1401.6048)

**SDR: Sample, Determinize, Replan (Brafman & Shani, IJCAI 2011; JAIR 45:565–600, 2012)**
- At each step SDR samples a small set of possible initial states (as few as 2 worked on the benchmarks). It builds a classical problem with a Palacios–Geffner-style translation whose state captures the belief, solves it with FF, and executes while the plan is "safe". It replans when an observation contradicts the plan. A lazy regression-based belief query regresses a literal through the action and observation history and checks consistency with the initial belief using a SAT solver (Minisat), with "partially-specified states" as a cache. — [JAIR 2012](https://arxiv.org/pdf/1401.6048); [JAIR page](https://jair.org/index.php/jair/article/view/10790)
- Theory: the translation is sound and complete when the sampled initial state is the true one. Under "certain assumptions on the connectivity of the domain", idealized SDR with the full initial belief reaches the goal whenever the goal is reachable. — [Brafman & Shani 2012, §1](https://arxiv.org/pdf/1401.6048)
- Variants: SDR-obs, with an observation bias, was fastest in most domains. SDR-SR adds state refutation to the goal. Over 25 runs, the best counts were SDR-obs 25 for time and CLG 12 for plan length. — [Table 3](https://arxiv.org/pdf/1401.6048)
- Dead-ends: on Wumpus where entering an unsafe cell kills the agent, all SDR variants fail, while CLG solves 4x4 (0.17 s), 8x8 (2.8 s) and 16x16 (182 s). SDR assumes a single initial state and does not sense for dead-ends. On the "restart" variant, CLG fails and SDR succeeds, but SDR does not weigh the cost of sensing against the risk of a restart. — [Tables 6–7](https://arxiv.org/pdf/1401.6048)
- New domains introduced: RockSample (8x8, 4–14 rocks; SDR solves them "not smartly"), MasterMind (only SDR-obs solves 4 pegs and 6 colours), Wumpus with dead-ends, and Wumpus with restart. — [§7.5](https://arxiv.org/pdf/1401.6048)
- Known benchmark weaknesses: no dead-ends, little conditional uncertainty propagation, and in colorballs, doors and unix "there isn't any smart exploration method" (the agent must visit each location). — [§7.5](https://arxiv.org/pdf/1401.6048)

**MPSR (Brafman & Shani, AAAI 2012)**
- "A Multi-Path Compilation Approach to Contingent Planning", AAAI 2012, pp. 1868–1874, DOI 10.1609/aaai.v26i1.8392. It is a sound and complete compilation of contingent problems into classical planning, where a single linear plan encodes several branches. Because the full compilation is huge, an incomplete sampled variant is used inside an online replanner. MPSR reasons about several sensing outcomes, which makes it less prone to dead-ends. It finds plans faster on most domains, though often longer ones. On a harder Wumpus variant with dead-ends it finds smaller plans faster and scales better (abstract only). — [mlanthology](https://mlanthology.org/aaai/2012/brafman2012aaai-multi); [PDF](https://tzin.bgu.ac.il/~shanigu/Publications/aaai12-28.pdf)

**K-replanner (Bonet & Geffner, IJCAI 2011)**
- "Planning under partial observability by classical replanning: theory and experiments", IJCAI 2011, pp. 1936–1941. — [repositori UPF](https://repositori.upf.edu/items/3cb5516d-a66f-40e7-8c82-af88db57376b/full)
- As described by Brafman & Shani: the K-replanner handles PPOS problems with only static hidden variables. It has no explicit sensing actions and assumes every observation is available immediately on entering a state. It uses an optimistic heuristic: in Wumpus it assumes the top-right cell is safe, goes there, and backtracks if not. It is "by far the best approach for domains with static hidden variables", but it is hard to extend, and localize, where uncertainty propagates, is "unsuitable for the K-planner". — [Brafman & Shani 2012, §7.2](https://arxiv.org/pdf/1401.6048)
- Bonet & Geffner say it relies on "a very effective form of belief representation based on literals and invariants" and scales to Wumpus n = 40, compared with n = 7 for offline planners and n = 20 for CLG and SDR. — [Bonet & Geffner, arXiv 1909.13778 (IJCAI 2013 version) §9](https://arxiv.org/pdf/1909.13778)
- PO-PRP describes the K-replanner translation as "always sound, and complete for simple contingent problems". — [PO-PRP wiki](https://github.com/qumulab/planner-for-relevant-policies/wiki/PO-PRP)
- From memory **[UNVERIFIED in fetched text]**: in "simple" problems the hidden fluents are static and every literal in a precondition or goal is either known or eventually sensed. In the completeness theorem, the K-replanner reaches the goal in a number of replanning episodes bounded by the number of possible observations, provided the problem is connected (no dead-ends). The planner treats sensing outcomes as action choices of the planner ("optimistic assumption"), and these assumptions are falsified at execution time.

**LW1 (Bonet & Geffner, AAAI 2014)**
- "Flexible and Scalable Partially Observable Planning with Linear Translations", AAAI 2014, DOI 10.1609/aaai.v28i1.9047. LW1 combines the broad scope of CLG with the speed of the K-replanner. Its translation is linear in size and complete for width-1 problems. It was evaluated on existing benchmarks and on new problems (abstract only). — [mlanthology](https://mlanthology.org/aaai/2014/bonet2014aaai-flexible); [AAAI page](https://ojs.aaai.org/index.php/AAAI/article/view/9047)
- From memory **[UNVERIFIED]**: the new problems included large Wumpus and Minesweeper instances and "Battleship"-like domains. LW1 handles non-static hidden fluents and explicit sensing, which the K-replanner does not.

**Belief-tracking theory behind LW1 (width, causal width, beam tracking)**
- Bonet & Geffner, "Belief Tracking for Planning with Sensing: Width, Complexity and Approximations", JAIR 50 (2014) 923–970, DOI 10.1613/JAIR.4475. Factored belief tracking is exponential in the problem width. Beam tracking is exponential in the smaller causal width. It achieves state-of-the-art real-time performance on large Battleship, Minesweeper and Wumpus instances. — [mlanthology](https://mlanthology.org/jair/2014/bonet2014jair-belief)
- IJCAI 2013 (arXiv 1909.13778): belief tracking for planning is NP-hard and coNP-hard. Causal belief tracking is sound, and it is complete for "causally decomposable" problems, where shared variables between beams are memory variables, for example static variables. Beam tracking is sound but incomplete. Results: Wumpus 30x30 with 32 pits and 32 wumpuses had 89% wins at 4.7 ms per decision. Minesweeper 16x16 with 40 mines had 79.8% wins, comparable to UCT+CSP. Battleship 10x10 needed 40 torpedoes per game in 0.0096 s per game, against about 2 s per game for POMCP (Silver & Veness). — [arXiv 1909.13778](https://arxiv.org/pdf/1909.13778)

**HCP: Heuristic Contingent Planner (Maliah, Brafman, Karpas & Shani, ICAPS 2014; JAAMAS 2018)**
- "Partially Observable Online Contingent Planning Using Landmark Heuristics", ICAPS 2014, pp. 163–171, DOI 10.1609/icaps.v24i1.13632. A contingent plan is viewed as alternating sensing actions and conformant segments. A landmark-based heuristic picks the next useful sensing action, and classical planning on a projection solves the conformant sub-problems, so no explicit belief-space model is built. It "solves many more problems than state-of-the-art translation-based online contingent planners, and in most cases much faster" (abstract only). — [PDF](https://tzin.bgu.ac.il/~shanigu/Publications/ICAPS2014.pdf); [ICAPS listing](https://icaps-conference.org/?p=1297)
- Journal version: "Landmark-based heuristic online contingent planning", *Autonomous Agents and Multi-Agent Systems* 32(5):602–634, 2018. It reports up to 3x speedups on simple problems and 200x on non-simple domains (abstract only). — [CRIS IUCC](https://cris.iucc.ac.il/en/publications/landmark-based-heuristic-online-contingent-planning/)

**Dead-ends and plan quality (2019–2023)**
- Shtutland, Shmaryahu, Brafman & Shani, "Unavoidable deadends in deterministic partially observable contingent planning", JAAMAS 37, art. 3 (2023), DOI 10.1007/s10458-022-09570-w. When no plan reaches the goal from every initial state, the planner maximizes the set of solved states. The paper distinguishes two types of unavoidable dead-end and compares two methods: an active one, which first separates solvable states from dead-end states, and a lazy one, which detects dead-ends while planning. Both are applied to offline and online planners (abstract only). — [CRIS IUCC](https://cris.iucc.ac.il/en/publications/unavoidable-deadends-in-deterministic-partially-observable-contin/)
- "Comparative criteria for partially observable contingent planning", JAAMAS (2019), covers how to rank valid contingent plans when no probabilities are available (abstract only). — [Springer](https://link.springer.com/article/10.1007/s10458-019-09406-0)
- "Heuristics for Partially Observable Stochastic Contingent Planning" (arXiv 2410.05870, 2024) takes an offline, probabilistic direction, in which policies run on weak devices. — [arXiv](https://arxiv.org/pdf/2410.05870)

**Optimistic assumptions in robot navigation (background for "optimistic replanning")**
- Koenig & Smirnov, "Sensor-Based Planning with the Freespace Assumption", ICRA 1997, pp. 3540–3545. Unknown terrain is assumed traversable. The robot plans a shortest path and replans when a blockage is sensed. The approach has good guarantees on restricted topologies such as grids but is not worst-case optimal in general. Basic-VECA gives guarantees within a constant factor. — [CMU RI](https://www.ri.cmu.edu/publications/sensor-based-planning-with-the-freespace-assumption)
- Nourbakhsh & Genesereth, "Assumptive Planning and Execution: a Simple, Working Robot Architecture", *Autonomous Robots* 3(1):49–67, 1996. The paper plans under simplifying assumptions and gives conditions under which this is sound and complete: never believe the problem is solved when it is not, and never take steps that make the problem unsolvable. The architecture was used in Dervish, winner of the 1994 AAAI robot competition. — [CMU RI](https://ri.cmu.edu/publications/assumptive-planning-and-execution-a-simple-working-robot-architecture)
- Tutorial on greedy online planning (Koenig) in robot navigation. — [idm-lab](https://idm-lab.org/greedyonline-tutorial.html)

### Inferences
- The PhD student's "online optimistic replanning" is, in planning-theoretic terms, a K-replanner or LW1-style approach: it determinizes sensing optimistically, replans when an assumption is falsified, and tracks beliefs with literals and invariants. It is also the symbolic analogue of the free-space assumption. The guarantees from the literature transfer if the warehouse model meets two conditions: (a) hidden facts such as item locations and table contents are static or width-1, and (b) the domain has no dead-ends, meaning moves are reversible and nothing irreversible is triggered by a wrong assumption. Under those conditions, completeness follows the K-replanner and LW1 argument. Each replanning episode either reaches the goal or learns at least one hidden literal, and there are finitely many, so the process terminates. This argument is my reconstruction and should be checked against the IJCAI 2011 paper.
- Domains where "sees only current room/cell" applies resemble colorballs, doors and unix: there is no informative long-range sensing. Brafman & Shani note there is no smart exploration strategy in such domains, so optimistic replanning loses little against full contingent reasoning in plan length. CLG was shorter in unix and doors by roughly 1.5–2x.
- If the warehouse has dead-ends (irreversible pick or drop, battery, one-way doors), SDR/K-replanner-style optimism can fail completely, as in Table 6. The candidates for that case are MPSR, the dead-end-aware CPOR variants (2023), or offline methods.

### Gaps
- I could not fetch the full text of the K-replanner (IJCAI 2011) and LW1 (AAAI 2014) papers, so the exact theorem statements and the LW1 benchmark tables are missing.
- I found no head-to-head published comparison of HCP against LW1 and the K-replanner with numbers.
- I found no 2024–2026 symbolic online contingent planner that clearly supersedes LW1, HCP or SDR. Recent work (2024–2026) has moved mostly toward POMDP, LLM and belief-state hybrids, for example "Belief-State Engine" (arXiv 2609.10036) and "PO-PDDL" (arXiv 2606.15654). I have not verified these as contingent planners.

---

## 3. Standard benchmark domains and what scaled; IPC tracks

### Takeaway
The de facto benchmark suite comes from the CLG, CFF and To et al. papers. It includes wumpus, doors, colorballs, unix, localize, ebtcs, elog and clog (logistics with sensing), medpks, and blocks with sensing. Brafman & Shani 2012 added RockSample, MasterMind and Wumpus variants with dead-ends or restarts. Bonet & Geffner added large Minesweeper and Battleship instances. No IPC track has ever been run for contingent (partially observable, sensing) planning. IPC 2006 and 2008 ran conformant and FOND tracks only.

### Cited Findings
- Domain descriptions:
  - Wumpus: grid with wumpuses or pits along the diagonal; stench and breeze sensing; no conditional effects.
  - Doors: walls with a single unlocked door each; the agent must try each door.
  - Colorballs: balls hidden in a grid must be delivered to bins by colour; large state space; no conditional effects.
  - Unix: find a file in a directory tree.
  - Localize: unknown position; wall sensing; conditional effects; uncertainty propagates.

  In all except localize, known facts do not become unknown again. — [Brafman & Shani 2012 §7](https://arxiv.org/pdf/1401.6048)
- Scale reached by online planners in Brafman & Shani's study (SDR variants): doors 17, localize 17, colorballs 9x9 with 7 balls (SDR-obs only), wumpus 20, unix 4, cloghuge, ebtcs-70. CLG execution failed beyond doors 13, localize 9 (its simulator was not usable there) and colorballs 9-3. — [Tables 1–2](https://arxiv.org/pdf/1401.6048)
- Wumpus as the standard benchmark has no pits, a known start and gold, and a wumpus below or left of each diagonal cell. It is fully solvable for n ≥ 3. Offline planners reach n = 7, online CLG and SDR reach n = 20, the K-replanner reaches n = 40, and beam tracking with a heuristic exceeds n = 100 in real time. — [Bonet & Geffner, arXiv 1909.13778 §9](https://arxiv.org/pdf/1909.13778)
- Minesweeper reaches 32x64 with 320 mines (80.3% wins, 2.9 s per game). Battleship reaches 40x40. — [arXiv 1909.13778 Tables 1–2](https://arxiv.org/pdf/1909.13778)
- RockSample (8x8, up to 14 rocks) and MasterMind were added. CLG cannot handle MasterMind (width > 1). — [Brafman & Shani 2012 Tables 8–9](https://arxiv.org/pdf/1401.6048)
- IPC 2006 non-deterministic track included a conformant subtrack, won by T0 (Palacios & Geffner). — [IPC-2006 probabilistic/non-det page](https://ipc06.icaps-conference.org/probabilistic/)
- IPC 2008 uncertainty part had a FOND track (entrants included Gamer by Edelkamp & Kissmann) and a conformant (NOND) track with CPA(H), CPA(C) and T0. A partially observable probabilistic track was planned "if at least 4 participants enter". I found no evidence that it ran. — [IPC-2008 results](https://ipc08.icaps-conference.org/probabilistic/wiki/index.php/Results.html); [Call for participation](https://ipc08.icaps-conference.org/probabilistic/wiki/index.php/Call_for_Participation.html)

### Inferences
- The warehouse gridworld is closest to a hybrid of colorballs (objects hidden in locations, delivered to targets) and doors or unix (exhaustive local sensing). These are exactly the domains where online replanners scale and where offline trees blow up.
- The SDR and CPOR benchmark sets, plus the dead-end domains (`DeadendDomains.zip` in the shanigu repo), are the practical source of PDDL benchmark files.

### Gaps
- medpks, ebtcs, elog and blocks with sensing are listed but I did not obtain their definitions or result tables. They originate from the CFF and PKS literature **[UNVERIFIED]**.
- I found no confirmation of who won the IPC-2008 FOND and conformant tracks.

---

## 4. Input languages

### Takeaway
There is no standard language. Planners share a de facto PDDL extension that came out of Conformant-FF, CFF and CLG: `(:init (unknown p) (oneof p q r) (or p q))` for initial uncertainty, plus `:observe` for sensing actions. Bonet & Geffner's tools use their own extended PDDL with invariants and sensors. PPDDL (probabilistic) is used by FOND planners such as PRP (via `oneof` effects), and Unified Planning has a contingent-problem class that can drive SDR and CPOR.

### Cited Findings
- Formal model used by the BGU line: ⟨P, A, φ_I, G⟩. φ_I is a propositional formula over the initial states, and each action has pre, effects and obs. Actions are deterministic. — [Brafman & Shani 2012 §2](https://arxiv.org/pdf/1401.6048)
- Bonet & Geffner's model uses multi-valued variables, initial clauses I, conditional and non-deterministic effects C → E1|…|En, observable variables V′ and sensor formulas W_a(Y=y), together with "defined variables" and "state constraints". — [arXiv 1909.13778 §2, §8](https://arxiv.org/pdf/1909.13778)
- K-replanner and cp2fsc read "an extended PDDL" using a parser based on Patrik Haslum's. — [GitHub cp2fsc-and-replanner](https://github.com/bonetblai/cp2fsc-and-replanner)
- PO-PRP consumes the K-replanner translation output. — [PO-PRP wiki](https://github.com/qumulab/planner-for-relevant-policies/wiki/PO-PRP)
- From memory **[UNVERIFIED in fetched sources]**: CLG and SDR syntax uses `(:action sense-x :observe (p))`, with `(unknown p)`, `(oneof …)` and `(or …)` in `:init`. MBP uses its own NuPDDL. POND reads PDDL with `oneof` and `:observation` and partly PPDDL. Unified Planning's `ContingentProblem` supports `SensingAction` and `add_oneof_initial_constraint` / `add_or_initial_constraint`.

### Gaps
- I did not fetch a grammar specification for any of the dialects. The exact keyword spellings for CLG, SDR and K-replanner should be checked against the benchmark files in each repository.

---

## 5. Available, maintained implementations (status October 2026)

### Takeaway
The only maintained, readily usable contingent planners are (1) the BGU/Shani C# code: SDR, MPSR, CPOR and dead-end handling in shanigu/ContingentPlanning, wrapped for Python in aiplan4eu/up-cpor as a Unified Planning engine; (2) PO-PRP inside QuMuLab/planner-for-relevant-policies, updated August 2026; and (3) Bonet's K-replanner and cp2fsc, which are legacy code (last updated December 2024). LW1, CLG, CFF, POND and MBP are research binaries that are hard to find or unmaintained.

### Cited Findings
- **aiplan4eu/up-cpor** (C#, 3 stars, last update 2025-02-10): a Unified Planning plugin. CPOR is exposed as `OneshotPlanner` (offline plan graph) and as a meta-engine (`MetaCPORPlanning[tamer]`). SDR is exposed as an `ActionSelector` loop (`get_action` / `apply` / `update`) with UP's `SimulatedEnvironment` or `SDRSimulator`. Setup: Python 3.8 conda environment and `pip install -r requirements.txt`, with notebooks for Windows and Colab. — [GitHub aiplan4eu/up-cpor](https://github.com/aiplan4eu/up-cpor); [fork guyazran/up-cpor](https://gitblind.noratr.app/guyazran/up-cpor)
- **shanigu/ContingentPlanning** (C#, updated February 2024) contains:
  - CPOR: latest code combining SDR, CPOR and unavoidable-dead-end handling; recommended.
  - SDR: online, with the SDR and MPSR translations.
  - CPOR.org: the original offline CPOR.
  - IMAP: multi-agent QDec-POMDP.
  - WriteKPlanner: the K-planner PPOS-to-FOND translation plus PO-PRP scripts.
  - `DeadendDomains.zip`.

  There are no build docs in the README. — [GitHub shanigu/ContingentPlanning](https://github.com/shanigu/ContingentPlanning)
- **QuMuLab/planner-for-relevant-policies** (PRP and PO-PRP, 33 stars, updated 2026-08-27). PO-PRP is on the `contingent` branch and runs via `src/poprp`. — [GitHub](https://github.com/QuMuLab/planner-for-relevant-policies); [PO-PRP wiki](https://github.com/qumulab/planner-for-relevant-policies/wiki/PO-PRP)
- **bonetblai/cp2fsc-and-replanner** (C++, exported from Google Code, 310 commits, updated 2024-12-05) contains k_replanner (classical planner selectable by parameter) and cp2fsc. Benchmarks include k_replanner instances on HOG grid maps (Sturtevant), which is relevant for gridworld navigation. — [GitHub](https://github.com/bonetblai/cp2fsc-and-replanner)
- **bonetblai/belief-tracking** (C++, updated January 2025) is presumably the beam-tracking code for Minesweeper, Battleship and Wumpus **[content UNVERIFIED]**. — [GitHub](https://github.com/bonetblai/belief-tracking)
- **aindilis/dnfct-frdcsa** (Perl wrapper, 2018/2024) is a third-party packaging of DNFct. — [GitHub](https://github.com/aindilis/dnfct-frdcsa)
- **kevinmcareavey/bcp** (Java) is a bounded-branching contingent planner. — [GitHub](https://github.com/kevinmcareavey/bcp)

### Inferences
- For a warehouse gridworld in Python, the lowest-friction path is Unified Planning with up-cpor. The model would be a `ContingentProblem` with `SensingAction`s such as "observe-table-contents-in-current-room". SDR then serves as the online baseline and CPOR as the offline baseline. The PRP `contingent` branch provides the FOND-translation baseline. The K-replanner (if it compiles) is the closest published analogue of the student's optimistic replanner and its natural baseline.
- LW1 does not seem to have a public GitHub release. It may sit inside cp2fsc-and-replanner (the README does not mention it), so it would have to be requested from the authors **[UNVERIFIED]**.

### Gaps
- I did not verify whether up-cpor installs on Linux with a current .NET or Mono runtime. The fork's notebooks target Windows and Colab.
- No public repositories were found for CLG, CFF, POND, MBP, HCP or LW1. Their binaries were historically distributed from the authors' pages, which could not be reached from this environment.
