# Assumption-Based Planning (planning under uncertainty by assuming unknown facts, planning as if they hold, monitoring and replanning)

Research-environment note: in this session only WebSearch and alphaXiv full text were reachable. Direct fetches of ojs.aaai.org, arxiv.org and the Geffner/RWTH host failed (DNS blocked). So full texts were read only for Pellier & Fiorino (arXiv 1810.08431) and Brafman & Shani SDR (arXiv 1401.6048). All other paper content comes from abstracts, bibliographic records and search snippets. Statements marked **[UNVERIFIED, from background knowledge]** were not confirmed against a source in this session.

## Q1. Who coined / formalized "assumption-based planning"? Key papers and exact bibliographic data

### Takeaway
No single paper coined "assumption-based planning". The idea shows up independently in four places:
- **Robotics:** Nourbakhsh & Genesereth's "assumptive planning" (1996).
- **Nondeterministic, partially observable planning (FBK/Trento):** Albore & Bertoli formalized *safe* assumption-based plans (AAAI 2004; LTL assumptions in ICAPS 2006).
- **Multi-agent HTN planning:** Pellier & Fiorino, a 2004 report later posted to arXiv.
- **Classical-planning compilation:** Davis-Mendelow, Baier & McIlraith (AAAI-12 workshop; AAAI 2013) compile assumption-based planning into classical planning by extending Palacios & Geffner's T0 (the planner "A0").

Related replanning methods also plan under assumptions: FF-Replan, CLG, the K-replanner, SDR, HCP and LW1.

### Cited Findings

**Bibliography (verified via bibliographic records)**

- **Nourbakhsh, I. & Genesereth, M. (1996).** "Assumptive Planning and Execution: a Simple, Working Robot Architecture." *Autonomous Robots* 3(1):49–67. DOI 10.1007/BF00162467 (the DOI appears in a Crossref URL in the search results).
  - The approach interleaves planning and execution by making *simplifying assumptions*.
  - Because assumptions can be wrong, the method must stop the robot from believing it has solved a problem it hasn't, and from taking steps that make a problem unsolvable.
  - The algorithm "preserves goal-reachability", and the paper gives conditions under which the architecture is sound and complete.
  - It was deployed on the robot Dervish, winner of the 1994 AAAI National Robot Competition.
  - Sources: [CMU RI page](https://ri.cmu.edu/publications/assumptive-planning-and-execution-a-simple-working-robot-architecture); [Crossref](https://api.crossref.org/works/10.1007%2FBF00162467).
- **Albore, A. & Bertoli, P. (2004).** "Generating Safe Assumption-Based Plans for Partially Observable, Nondeterministic Domains." *Proc. AAAI 2004*, pp. 495–500.
  - It describes "reactive planning using assumptions" as a well-known approach for nondeterministic, partially observable domains.
  - It formally defines *safe* plans (details in Q2).
  - It extends a planner inside a reactive platform to generate safe plans.
  - Experiments suggest this minimizes replanning while keeping plan generation efficient.
  - Sources: [ML Anthology](https://mlanthology.org/aaai/2004/albore2004aaai-generating); [FBK CRIS](https://cris.fbk.eu/handle/11582/2210).
- **Albore, A. & Bertoli, P. (2006).** "Safe LTL Assumption-Based Planning." *Proc. ICAPS 2006*, pp. 193–202 (DBLP).
  - Assumptions are expressed in LTL, so they can be about the evolution of the domain and not only about the initial state.
  - "Exploiting LTL assumptions highly improves plan-generation efficiency, and enforcing safety improves plan execution by avoiding costly replanning episodes."
  - Some listings give the venue as "AIPS 2006", which is DBLP's series key for ICAPS. They refer to the same paper.
  - Sources: [FBK CRIS](https://cris.fbk.eu/handle/11582/11488); [DBLP Albore](https://dblp.uni-trier.de/pid/05/1344.html); [LAAS TrustMeIA document citing pp.193–202](https://sites.laas.fr/projects/trustmeia/sites/www.laas.fr.projects.trustmeia/files/u173/3.Stage_TrustMeIA_AA_CC.pdf).
- **Pellier, D. & Fiorino, H. (2004/2018).** "Assumption-Based Planning." Technical report, Laboratoire Leibniz (CNRS-INPG-IMAG), dated November 2004; posted as arXiv:1810.08431 (19 Oct 2018).
  - This is a multi-agent, Lakatos-inspired "conjecture/refutation" planner built on SHOP (HTN).
  - A *conjecture* is χ = (⟨H_α1, α1⟩, …, ⟨H_αn, αn⟩), where H_αi is the set of unsatisfied preconditions *assumed* before executing αi.
  - States progress as E_0 = E ∪ H_α1 and E_i = ((E_{i−1} ∪ H_αi) − Del) ∪ Add.
  - Branch-and-bound search minimizes the number of assumptions. Only predicates declared "hypothetical" may be assumed.
  - With an assumption bound of 0, the planner reduces to SHOP.
  - Assumptions become new goals for teammates.
  - Source: [arXiv 1810.08431](https://arxiv.org/abs/1810.08431) (full text read via alphaXiv).
- **Davis-Mendelow, S., Baier, J. A. & McIlraith, S. A. (2012).** "Making Reasonable Assumptions to Plan with Incomplete Information: Abridged Report." AAAI-12 Workshop on Problem Solving Using Classical Planners, AAAI Technical Report WS-12-12, pp. 16–22.
  - Record listings disagree on the venue: one says "Workshop on Classical Planning for Problem Solving", another says "National Conference on AI, WS-12-12".
  - The report describes the planner as a T0 variant.
  - Sources: [aitopics](https://aitopics.org/doc/conferences:D1B6392E); [BibBase](https://bibbase.org/network/publication/davismendelow-baier-mcilraith-makingreasonableassumptionstoplanwithincompleteinformationabridgedreport-2012); [ANID](https://investigadores.anid.cl//en/public_search/work?id=448252).
- **Davis-Mendelow, S., Baier, J. A. & McIlraith, S. A. (2013).** "Assumption-Based Planning: Generating Plans and Explanations under Incomplete Knowledge." *Proc. AAAI-13* (27th AAAI), pp. 209–216. DOI 10.1609/aaai.v27i1.8687.
  - Unlike conformant planning, which requires success under *all* completions of the initial state, an assumption-based plan may assert additional assumptions about the world. This often yields high-quality plans where no conformant plan exists.
  - Costs are attached to both assumptions and actions, so the planner finds *preferred* assumption-based plans.
  - It exploits the correspondence with conformant planning to compile into classical planning by **extending Palacios & Geffner's T0 translation**. The resulting planner is **A0**.
  - It is evaluated on IPC-derived domains.
  - The authors propose applications to **explanations, diagnoses and counter-examples**.
  - Sources: [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/8687); [BibBase](https://bibbase.org/network/publication/davismendelow-baier-mcilraith-assumptionbasedplanninggeneratingplansandexplanationsunderincompleteknowledge-2013).

**Knowledge-level translations and the replanners built on them (bibliography)**

- **Palacios, H. & Geffner, H. (2009).** "Compiling Uncertainty Away in Conformant Planning Problems with Bounded Width." *JAIR* 35:623–675. DOI 10.1613/jair.2708. arXiv:1401.3468 (posted 2014).
  - The translation is polynomial in the conformant width, and the width is bounded in most benchmarks.
  - The planner T0 was the best performer in the IPC-2006 conformant track.
  - It is restricted to deterministic actions with uncertainty only in the initial state.
  - Sources: [JAIR](https://jair.org/index.php/jair/article/view/10618); [DBLP](https://dblp.uni-trier.de/rec/journals/jair/PalaciosG09.html).
- **Albore, A., Palacios, H. & Geffner, H. (2009).** "A Translation-Based Approach to Contingent Planning." *Proc. IJCAI 2009*, pp. 1623–1628.
  - The contingent problem P maps to a nondeterministic problem X(P). A classical relaxation X⁺(P) selects actions; this is the planner CLG.
  - The translation is sound, polynomial and complete for bounded contingent width, and almost all benchmarks have width 1.
  - Sources: [IJCAI](https://www.ijcai.org/Abstract/09/271); [DBLP](https://dblp.uni-trier.de/rec/conf/ijcai/AlborePG09.html).
- **Bonet, B. & Geffner, H. (2011).** "Planning under Partial Observability by Classical Replanning: Theory and Experiments." *Proc. IJCAI 2011*, pp. 1936–1941. DOI 10.5591/978-1-57735-516-8/IJCAI11-324. This is the **K-replanner**.
  - Sources: [IJCAI](https://ijcai.org/Abstract/11/324); [UPF repository](https://repositori.upf.edu/items/3cb5516d-a66f-40e7-8c82-af88db57376b); [poster](https://bonetblai.github.io/reports/IJCAI11-replanning-poster.pdf).
- **Shani, G. & Brafman, R. I. (2011).** "Replanning in Domains with Partial Information and Sensing Actions." *Proc. IJCAI 2011*, pp. 2021–2026. DOI 10.5591/978-1-57735-516-8/IJCAI11-337. This is **SDR**.
  - Source: [ML Anthology](https://mlanthology.org/ijcai/2011/shani2011ijcai-replanning).
- **Brafman, R. I. & Shani, G. (2012).** Same title, *JAIR* 45:565–600. DOI 10.1613/jair.3711. arXiv:1401.6048.
  - The author order is reversed relative to the IJCAI version.
  - Sources: [JAIR](https://jair.org/index.php/jair/article/view/10790); [DOI](https://www.doi.org/10.1613/JAIR.3711).
- **Maliah, S., Brafman, R. I., Karpas, E. & Shani, G. (2014).** "Partially Observable Online Contingent Planning Using Landmark Heuristics." *Proc. ICAPS 2014*, pp. 163–171. DOI 10.1609/icaps.v24i1.13632. This is the planner **HCP**.
  - HCP solves intermediate conformant subproblems by projection plus classical planning, with no translation.
  - Journal extension: Maliah, Shani & Brafman, "Landmark-based heuristic online contingent planning," *JAAMAS*, DOI 10.1007/s10458-018-9389-9.
  - Sources: [AAAI OJS](https://ojs.aaai.org/index.php/ICAPS/article/view/13632); [DBLP](https://dblp.dagstuhl.de/rec/conf/aips/MaliahBKS14.html).
- **Bonet, B. & Geffner, H. (2014).** "Flexible and Scalable Partially Observable Planning with Linear Translations." *Proc. AAAI 2014*, pp. 2235–2241. DOI 10.1609/aaai.v28i1.9047. This is the planner **LW1**.
  - CLG is complete for width-1 problems but its translation is quadratic. The K-replanner's translation is linear but less general. LW1 is linear *and* complete for width 1.
  - Sources: [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/9047); [DBLP](https://dblp.uni-trier.de/rec/conf/aaai/BonetG14.html).

**Determinization and optimism roots (bibliography)**

- **Yoon, S., Fern, A. & Givan, R. (2007).** "FF-Replan: A Baseline for Probabilistic Planning." *Proc. ICAPS 2007*, pp. 352–359. Only the start page, 352, is confirmed by DBLP.
  - It determinizes the domain, with each probabilistic outcome becoming a separate deterministic action and probabilities ignored. It plans with FF and replans on unexpected outcomes.
  - It outperformed the IPPC-06 entrants although it did not enter.
  - Sources: [PDF](https://engineering.purdue.edu/~givan/papers/icaps07-ffreplan.pdf); [DBLP](https://dblp.org/rec/conf/aips/YoonFG07).
- **Koenig, S. & Smirnov, Y. (1997).** "Sensor-Based Planning with the Freespace Assumption." *Proc. ICRA 1997*.
  - Planning with the freespace assumption gives good guarantees on some restricted topologies (e.g., grids) but is not worst-case optimal in general.
  - Source: [CMU RI](https://www.ri.cmu.edu/publications/sensor-based-planning-with-the-freespace-assumption).
- **Koenig, S., Tovey, C. & Smirnov, Y. (2003).** "Performance Bounds for Planning in Unknown Terrain." *Artificial Intelligence* (2003).
  - Greedy Mapping and Dynamic A* are not worst-case optimal, by a factor of Ω(log|V|/log log|V|).
  - Source: [idm-lab abstract](https://idm-lab.org/bib/abstracts/Koen03a.html).
- **Koenig, S. & Likhachev, M. (2005).** "Fast Replanning for Navigation in Unknown Terrain." *IEEE Transactions on Robotics* (2005).
  - D* Lite under the freespace assumption.
  - Source: search snippet only; see the [Koenig search results](https://idm-lab.org/bib/abstracts/Koen03a.html).
- **Son, T. C. & Baral, C. (2001).** "Formalizing Sensing Actions — A Transition Function Based Approach." *Artificial Intelligence* 125(1–2):19–91.
  - It introduces the **0-approximation**: approximate states are pairs of true/false fluent sets, with a sound, incomplete transition function.
  - Sources: [PDF](https://www.cs.nmsu.edu/~tson/papers/SonB01.pdf); [DBLP](https://dblp.uni-trier.de/rec/journals/ai/SonB01.html).

### Inferences
- The term "assumption-based planning" is used for at least three technically different things:
  1. **Albore & Bertoli:** online plans under assumptions on nondeterminism and observations, with execution monitors.
  2. **Pellier & Fiorino:** relaxing preconditions into hypotheses that become goals for other agents.
  3. **Davis-Mendelow et al.:** offline, cost-based choice of assumptions on the initial state, compiled to classical planning.
- A synthesis should name the specific sense and not merge them.
- Earliest priority for the *idea* goes to Nourbakhsh & Genesereth (1996, "assumptive planning"). The *term* "assumption-based planning" with a formal safety notion goes to Albore & Bertoli (2004).

### Gaps
- I could not read the full Albore & Bertoli (2004/2006) or Davis-Mendelow et al. (2013) texts because the hosts were blocked. Formal definitions below are paraphrased from abstracts.
- Exact end page of FF-Replan is unverified (352–359 from memory).
- The volume and pages of Koenig, Tovey & Smirnov (AIJ 2003) were not retrieved. **[UNVERIFIED, from background knowledge]:** AIJ 147(1–2):253–279.
- A conference version of Nourbakhsh & Genesereth (e.g., an earlier workshop or IJCAI paper on "assumptive planning") was not verified.
- Hanks-style "plausible"/"reasonable assumptions" planning and Goldman & Boddy were not checked. I found no source in this session for those roots.

## Q2. What is "safety" of an assumption-based plan; how are assumptions monitored; relation to explanations and diagnosis

### Takeaway
In Albore & Bertoli's work, a **safe** assumption-based plan has two properties:
- it reaches the goal whenever the assumptions hold;
- it carries a monitor guaranteeing that any violation of an assumption which would endanger success is *detected* during execution.

Replanning therefore happens exactly when needed. SDR and the K-replanner implement the same pattern operationally: they track the belief or knowledge state and check that the next action is still applicable ("safe") in all states still possible. Davis-Mendelow et al. reuse assumption-based planning to generate *explanations, diagnoses and counter-examples*, since an explanation is a set of assumptions that would make the observations/goal hold.

### Cited Findings
- **Albore & Bertoli 2004, the safety definition.**
  - Safe plans "succeed when the assumption holds and ensure that any failure of a success-endangering assumption is caught by a monitor, so replanning happens only when it is needed."
  - The motivating problem: "assumptions may be wrong; this may cause an assumption-based plan to fail" (paraphrased by the search tool from the abstract).
  - Sources: [ML Anthology](https://mlanthology.org/aaai/2004/albore2004aaai-generating); [FBK CRIS](https://cris.fbk.eu/handle/11582/2210).
- **Albore & Bertoli 2006.**
  - It extends assumptions to LTL formulae over executions.
  - Enforcing safety avoids "costly replanning episodes", and LTL assumptions "highly improve plan-generation efficiency".
  - Source: [FBK CRIS](https://cris.fbk.eu/handle/11582/11488).
- **Nourbakhsh & Genesereth 1996.**
  - Safety-like conditions: the robot must not falsely believe it has solved the problem, and must not take actions that make the problem unsolvable. The method preserves goal-reachability.
  - Soundness and completeness hold under stated conditions.
  - Source: [CMU RI](https://ri.cmu.edu/publications/assumptive-planning-and-execution-a-simple-working-robot-architecture).
- **SDR "safe"/"unsafe" actions (Brafman & Shani 2012, full text read).**
  - SDR executes the plan "as long as it is safe to do so. When this is no longer the case, we replan."
  - Under the sampled translation, Kp means p is known in all *sampled* states only. Actions whose preconditions are not known in all *possible* states are "unsafe" and "must never be executed".
  - SDR therefore keeps a true belief-state representation, queried lazily by regression.
  - If an action is unsafe, the witness state is added to the sample and SDR replans.
  - Source: [arXiv 1401.6048 / JAIR 45](https://arxiv.org/abs/1401.6048).
- **SDR monitoring by observations.**
  - The plan is executed until a sensing action reveals an outcome different from the one predicted by the sampled state s′. Belief states inconsistent with the observation are then removed.
  - Theorem (idealized SDR, with complete belief tracking and belief-connectedness): the goal is reached after at most |initial belief| replanning episodes, since each failure eliminates at least s′.
  - Source: [arXiv 1401.6048](https://arxiv.org/abs/1401.6048).
- **K-replanner monitoring.** The classical plan is optimized "to the shortest prefix that either reaches the goal or yields an observation refuting an action assumption". Execution stops when an assumption is refuted, and the planner replans.
  - Source: [IJCAI-11 poster](https://bonetblai.github.io/reports/IJCAI11-replanning-poster.pdf).
- **Explanations and diagnosis (Davis-Mendelow et al. 2013).**
  - Assumption-based planning is "of great utility in the generation of explanations, diagnoses, and counter-examples, tasks that share a computational core with planning".
  - Costs on assumptions express preference among explanations.
  - Source: [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/8687).

### Inferences
- Davis-Mendelow et al. focus on choosing *which* assumptions to make (cost/preference). They do not focus on monitoring.
- Albore & Bertoli focus on guaranteeing that a *violated* assumption is *observable in time*. That is a condition coupling the plan with the observation model.
- The two are complementary. A "safe, preferred" assumption-based plan combines both. I found no paper doing both explicitly.
- Safety in the Albore & Bertoli sense means monitorability, i.e., detectability of assumption failure. SDR's notion of "safe action" is narrower: an action is safe when its preconditions are known in the true belief state. It is not about detecting assumption violations in general.

### Gaps
- I did not obtain the precise formal definition from Albore & Bertoli: plan language, nondeterministic domain model, or the monitor as a finite-state observer. **[UNVERIFIED, from background knowledge]:** the plans are built in the MBP/symbolic BDD framework of the FBK group, with monitors that track the belief under the assumption and flag when the observed history becomes inconsistent with it.
- I did not verify whether Davis-Mendelow et al. formally relate "explanation" to abductive diagnosis (e.g., Reiter-style or McIlraith's earlier diagnosis-as-planning work).

## Q3. Optimism in the face of uncertainty: free-space assumption, determinization, optimistic sensing

### Takeaway
"Plan as if the unknown were favourable, then replan on contradiction" is a recurring pattern:
- **Navigation:** the freespace assumption, using D*/D* Lite (Koenig et al.).
- **Probabilistic planning:** all-outcomes determinization in FF-Replan.
- **Contingent planning:** CLG's optimistic assumption on sensed values; the K-replanner's assumptions on observation outcomes; SDR's sampled "witness" state.

All of these are instances of assumption-based planning with monitoring. They differ in what is assumed (map cells, action outcomes, observation values, initial state) and in whether a safety/reachability guarantee is provided.

### Cited Findings
- **Freespace assumption (Koenig & Smirnov, ICRA 1997).**
  - Unknown terrain is assumed traversable. The robot plans the shortest path under this assumption, moves, senses, and replans.
  - Guarantees are good on grids but not worst-case optimal in general. The paper proposes Basic-VECA as an alternative.
  - Source: [CMU RI](https://www.ri.cmu.edu/publications/sensor-based-planning-with-the-freespace-assumption).
- **Freespace assumption (Koenig, Tovey & Smirnov, AIJ 2003).** Greedy Mapping and Dynamic A* are Ω(log|V|/log log|V|) worse than optimal in the worst case.
  - Source: [idm-lab abstract](https://idm-lab.org/bib/abstracts/Koen03a.html).
- **FF-Replan:** "determinizes the input domain, removing all probabilistic information, plans deterministically, and replans when unexpected outcomes occur". It was criticized for ignoring uncertainty in "probabilistically interesting" domains (Little & Thiébaux 2007).
  - Sources: [FF-Replan PDF](https://engineering.purdue.edu/~givan/papers/icaps07-ffreplan.pdf); [DBLP](https://dblp.org/rec/conf/aips/YoonFG07).
- **CLG's optimistic sensing.** If action a needs precondition p and p can be sensed, CLG plans as if p is true provided a sensing action a′ for p comes before a. "Thus, it makes optimistic assumptions regarding future outcomes of sensing actions." CLG replans if the actual outcome differs.
  - Technically, CLG uses propositions Ap ("the value of p was sensed") instead of Kp as the precondition, which "results in an optimistic choice of values for sensed propositions".
  - Source: SDR paper's description of CLG, [arXiv 1401.6048](https://arxiv.org/abs/1401.6048).
- **SDR replaces optimism with sampling.** "We also use sampling to remove the optimistic bias of CLG: we sample an arbitrary initial state s_I, and assume that observation values are those obtained when s_I is the true initial state."
  - Importantly, SDR does *not* plan as if s′ were the actual state. It reasons about the *belief state* assuming the observations come from s′, and it seeks a belief where the goal is *known*.
  - Source: [arXiv 1401.6048](https://arxiv.org/abs/1401.6048).
- **Nourbakhsh & Genesereth (1996)** present "making simplifying assumptions" as a general robot architecture, implemented on Dervish.
  - Source: [CMU RI](https://ri.cmu.edu/publications/assumptive-planning-and-execution-a-simple-working-robot-architecture).

### Inferences
- What each method assumes:
  - the freespace assumption fixes an unknown *initial state* to its most favourable value;
  - FF-Replan fixes *action outcomes*;
  - CLG and the K-replanner fix *observation outcomes*, optimistically and chosen by the planner;
  - SDR fixes the *initial state* by sampling (not chosen to help the planner), which induces observation outcomes.
- **Optimism with replanning versus "optimism in the face of uncertainty" in RL.** Optimism plus replanning is closely analogous to optimism in exploration. The SDR completeness proof explicitly borrows from PAC-RL E³ and R-max (Kearns & Singh 2002; Brafman & Tennenholtz 2003), as the full text states.
  - Source for the borrowing: [arXiv 1401.6048](https://arxiv.org/abs/1401.6048).

### Gaps
- I found no source for a Koenig & Likhachev paper specifically titled "planning with optimism". Their related item is "Fast Replanning for Navigation in Unknown Terrain" (IEEE T-RO 2005), seen only as a search snippet.
- **[UNVERIFIED, from background knowledge]:** IEEE T-RO 21(3):354–363.

## Q4. Formal relation between assumption-based planning and knowledge-level planning (K0/tags, K-replanner, 0-approximation, and a Π⁺/Π⁻ theory)

### Takeaway
Knowledge-level translations make assumption-based planning directly expressible:
- In Palacios & Geffner's K_{T,M} / K_0 translations, the classical state contains literals KL / K¬L, and tags t encode "L holds if the initial state is compatible with t" (KL/t).
- An **assumption** is then an action that adds KL (or KL/t) without justification, possibly at a cost. Davis-Mendelow et al.'s A0 extends T0 in exactly this direction.
- The K-replanner and CLG add assumption-style actions for *observation outcomes*. Execution monitors these assumptions and replans when an observation refutes one.

An agent that maintains (Π⁺, Π⁻), the sets of fluents known true and known false, is maintaining a **0-approximation** (Son & Baral 2001). This is the same object as the set {KL} ∪ {K¬L} in K_0 / K-replanner states. Planning on top of it while treating unknown fluents as assumed values is assumption-based planning at the knowledge level.

### Cited Findings
- **Palacios & Geffner 2009** compile conformant planning to classical planning with a translation whose size is polynomial in the conformant width. T0 is built on it.
  - Source: [JAIR](https://jair.org/index.php/jair/article/view/10618).
- The SDR paper describes the knowledge-level encoding as follows.
  - "This is achieved by extending the language with propositions of the form Kp, and K¬p, denoting the fact that the agent 'knows' that p is true and false, respectively."
  - On extension to sensing: these ideas "result in non-deterministic planning problems because the effect of a sensing action on an agent's belief state cannot be known offline". Determinizing that step is where assumptions enter.
  - SDR's translation is "inspired by the K^S_0 translation introduced by Palacios and Geffner (2009)". It uses p/s propositions (p conditioned on initial state s); grouping these gives the *tags* of Palacios & Geffner and Albore et al.
  - Source: [arXiv 1401.6048](https://arxiv.org/abs/1401.6048).
- **Davis-Mendelow et al. 2013** compute assumption-based plans "via a translation to classical planning" that "extends the Palacios and Geffner approach used in their T0 planner". The planner is A0.
  - Source: [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/8687).
- **K-replanner.**
  - It computes a classical plan for the current *knowledge state* and stops at the first observation that refutes an action assumption.
  - It is sound and complete for "simple" problems: completeness needs a connected state space and initial clauses in prime-implicate form.
  - Experiments used FF on freespace, doors, wumpus and colored-balls.
  - Source: [IJCAI-11 poster](https://bonetblai.github.io/reports/IJCAI11-replanning-poster.pdf).
- **LW1 (Bonet & Geffner 2014)** characterizes the trade-off: the K-replanner uses *linear* translations and scales, but is less general than CLG.
  - Source: [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/9047).
- **0-approximation (Son & Baral 2001).** It is defined by a transition function over approximate states, i.e., pairs of sets of fluents known true and known false. It is sound but incomplete with respect to possible-worlds semantics, and it has a much smaller state space.
  - Source: [Son & Baral PDF](https://www.cs.nmsu.edu/~tson/papers/SonB01.pdf).

**K-replanner details [UNVERIFIED, from background knowledge]**
- "Simple" problems: hidden (unknown) fluents are *static*, i.e., not affected by actions, and the non-unary initial clauses involve only static fluents.
- Under these conditions, belief tracking reduces to unit resolution over the KL literals: progression in K(P), plus unit propagation over the prime-implicate clauses. So tracking is linear and close to a 0-approximation enriched with static clauses.
- Planning uses the translation K(P) with literals KL/K¬L. For each sensing outcome it adds "assumption" actions: the classical planner may make L known when ¬K¬L holds, i.e., when L is not known false and is observable. Executing the plan tests these assumptions.
- I could not retrieve the full paper to confirm the exact action schema names and preconditions.

**The (Π⁺, Π⁻) PhD thesis**
- I could not identify the specific thesis. Searches were not targeted because no author or title was given, and repository hosts were partly unreachable.
- Formally:
  - its (Π⁺, Π⁻) theory equals the 0-approximation state of Son & Baral;
  - it also equals the set of K-literals in K_0 (Palacios & Geffner's translation without tags) and the K-replanner's knowledge state without clauses;
  - "observation rules" that move literals into Π⁺ or Π⁻ play the role of the K-replanner's/CLG's observation and ramification axioms.

### Inferences
- **Two ways of placing assumptions in a KL/K¬L state:**
  - *Assumption-as-action:* add KL (or KL/t) at cost c. This is the Davis-Mendelow A0 style: offline, about the initial state.
  - *Assumption-on-observation:* determinize sensing so that observing L yields KL. This is the CLG / K-replanner style: online, with refutation by actual observation.
- **What this means for a (Π⁺, Π⁻) agent:**
  - Plans computed by assuming unknown literals, i.e., fluents in neither Π⁺ nor Π⁻, are *assumption-based plans*.
  - Literals entering Π⁻ (or Π⁺) by observation rules are exactly the monitoring events that refute assumptions and trigger replanning.
  - Safety in the Albore & Bertoli sense requires that every assumed literal be observable (decidable into Π⁺ or Π⁻) before the plan step that depends on it.
- **Incompleteness of 0-approximation tracking.** Tracking via 0-approximation (Π⁺, Π⁻) is incomplete when knowledge is disjunctive, e.g. "the Wumpus is at (3,4) or (4,3)". This is exactly why the K-replanner adds static clauses/prime implicates and why CLG and LW1 use tags/width.
  - Consequently, a (Π⁺, Π⁻)-only agent can fail to detect that an assumption has become impossible. It can be "unsafe" in the Albore & Bertoli sense unless the domain is width-1 / simple-like.
- **Completeness under replanning.** The completeness argument of SDR (each refuted assumption removes ≥1 initial state; belief-connectedness ensures progress) and that of the K-replanner (each refutation adds ≥1 known literal; finitely many literals) are the standard templates. They can be reused to prove termination of a (Π⁺, Π⁻) replanning agent, since Π⁺ ∪ Π⁻ grows monotonically for static hidden fluents.

### Gaps
- I could not read the original IJCAI-11 K-replanner text to quote the K(P) translation and the "simple problems" definition (host blocked). The poster summary is the only verified source.
- I could not identify the Π⁺/Π⁻ thesis. The caller should supply author and title if a precise comparison is needed.
- I found no paper that formally proves the equivalence "K_0 knowledge state = 0-approximation". The correspondence is my inference from the two definitions.
- Not checked: Brafman & Shani's later belief-tracking-by-regression work, Bonet & Geffner's JAIR 2014 width/belief-tracking paper, and work by Grastien & Scala. They may be relevant for formal completeness of tracking.
