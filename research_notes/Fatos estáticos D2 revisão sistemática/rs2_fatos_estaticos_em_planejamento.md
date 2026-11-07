# Systematic review (facet RS2): static facts and instance structure inside automated planning, as meta-structure, as hidden information, and as something inferred from the state space

Review date: 2026-11-07. Single reviewer (research agent). This review extends `../Diferenciais da tese revisão sistemática/rs_fatos_estaticos.md` (RS1, 2026-10-09). RS1 asked whether anyone learns static facts online with known schemas. This facet (RS2) asks the planning-internal question: who treats static facts as the thing that selects an instance's transition system, who plans with them hidden, and who infers them from state-space structure. Items already graded in RS1 are only re-cited here when RS2 adds new full-text evidence.

Reading labels:
- **[full text]**: read in full or in targeted passages in this session (via alphaXiv).
- **[abstract/snippet]**: search snippet or abstract only.
- **[RS1]**: carried over from RS1, not re-read.
- **[memory, unverified]**: background knowledge not confirmed in this session.

Thesis differential under test (D2, Fernando Chiu, USP): given fixed lifted action templates, the static facts (predicates no action changes, e.g. `connected(r1,r2)`) are a **meta-structure**. They select which transition system holds. They are **hidden** from the agent and **identified online by interaction**, while the agent tracks knowledge (Π⁺,Π⁻) with observation rules. The claim to test is that this combined view is new.

---

## Q1. Protocol, search log and PRISMA-style counts

### Takeaway
The protocol was fixed before searching. I ran 15 distinct search strings on 2 tools (12 WebSearch, 3 alphaXiv `discover_papers`) and did 7 targeted full-text reads through alphaXiv `answer_pdf_queries`. I snowballed from 4 seeds. WebFetch was blocked for the two university hosts I tried, so those PDFs were read through alphaXiv instead.

### Protocol (written before searching)
- **Facet research question.** In symbolic automated planning (not learning from traces), how have static predicates and instance-level structure been treated? Three readings:
  - (a) as structural or meta information: grounding, invariants, types, symmetry, domain analysis;
  - (b) as hidden information during planning: contingent planning with static hidden variables, CTP, unknown maps, open world;
  - (c) as something inferred from the structure of the state space or from images.
  - Sub-question (d): generalized planning where instances differ only in objects and static facts.
- **Grading target.** For each included study, grade it against D2 = (i) meta-structure selecting the transition system given templates + (ii) hidden from the agent + (iii) identified online by interaction with (Π⁺,Π⁻)-style knowledge and observation rules.
- **Inclusion.**
  - I1: published 1990–2026, peer-reviewed or arXiv;
  - I2: symbolic planning (classical, contingent, or graph-based online planning);
  - I3: explicitly treats static facts or instance structure as structural, hidden, or inferred.
- **Exclusion.**
  - E1: pure motion planning or purely continuous control;
  - E2: not in English;
  - E3: duplicates and mirror pages;
  - E4: pure combinatorics of games (e.g. Minesweeper phase transitions) with no planning model;
  - E5: LLM-only planners.
- **Verdict scale.** IDÊNTICO (identical), SOBREPOSIÇÃO FORTE (strong overlap), PARCIAL (partial), DISTANTE (distant). Each verdict is given against the whole D2 claim, with a note on which component (i/ii/iii) overlaps.

### Search log
"Hits" means the records the tool returned. Web search returns about 10 links per query, so these are not database hit counts.

| # | String (as typed) | Tool | Hits | Screened | Retained |
|---|---|---|---|---|---|
| W1 | `Bonet Geffner "simple problems" K-replanner hidden fluents static partial observability` | WebSearch | 10 | 10 | 3 (IJCAI-11 paper, poster, LW1 AAAI-14) |
| W2 | `Canadian Traveler Problem Bnaya Felner Shimony sensing unknown edges` | WebSearch | 10 | 10 | 2 (Bnaya et al. IJCAI-09; arXiv 1702.07001 screened, not retained) |
| W3 | `Rodriguez Bonet Romero Geffner learning first-order representations black-box states KR 2021` | WebSearch | 10 | 10 | 1 (KR-21, mirrors deduplicated) |
| W4 | `Helmert 2009 concise finite-domain representations static predicates grounding invariants` | WebSearch | 20 (2 sets) | 20 | 3 (Helmert AIJ-09; Lauer et al. IJCAI-21; Fišer lifted mutex mention) |
| W5 | `Fox Long TIM type inference planning domain analysis static relations` | WebSearch | 10 | 10 | 3 (TIM JAIR-98; AIPS-00 extension; Cresswell-Fox-Long ADL TIM) |
| W6 | `Talamadupula open world planning unknown objects open world quantified goals 2010` | WebSearch | 10 | 10 | 2 (AAAI-10; ACM TIST-10) |
| W7 | `generalized planning instances same domain different static facts general policies features Bonet Geffner Francès 2021 "static"` | WebSearch | 10 | 10 | 3 (Francès et al. AAAI-21; Bonet & Geffner IJCAI-18; Bonet & Geffner AAAI-21 width) |
| W8 | `Etzioni Golden Weld local closed world information sound and efficient closed-world reasoning planning unknown objects` | WebSearch | 10 | 10 | 2 (LCW AIJ; XII AAAI-94) |
| W9 | `lifted planning static predicates successor generation Powerlifted Corrêa Pommerening Helmert static atoms preprocessing` | WebSearch | 10 | 10 | 1 (Lauer et al. IJCAI-21, duplicate of W4) |
| W10 | `online graph exploration unknown graph Deng Papadimitriou 1990 exploring unknown environment competitive` | WebSearch | 10 | 10 | 1 (Deng & Papadimitriou FOCS-90 / JGT-99) |
| W11 | `Asai FOSAE "first-order state autoencoder" unsupervised grounding objects predicates images planning` | WebSearch | 10 | 10 | 1 (Asai ICAPS-19) |
| W12 | `Papadimitriou Yannakakis 1991 shortest paths without a map Canadian traveller problem` | WebSearch | 10 | 10 | 1 (TCS 84, 1991) |
| A1 | Q: learning first-order symbolic representations (objects, static predicates, schemas) from state-space structure or images; kw: static predicates, state space structure, first-order representations, Latplan, FOSAE | alphaXiv | 15 | 15 | 4 (1902.08093, 1909.05546, 2107.00110, 2602.23148) |
| A2 | Q: planning with hidden map/topology or static facts discovered during execution (CTP, contingent, Wumpus, Minesweeper, Battleship, unknown graphs) | alphaXiv | 14 | 14 | 4 (2403.01872, 2610.13712, 1909.13778, 2303.07397) |
| A3 | Q: general policies or models generalizing over instances differing in objects and static relations (GNN, Occhipinti, Srivastava) | alphaXiv | 14 | 14 | 4 (2411.08040, 2109.10129, 2409.15892, 2602.23148, the last a duplicate) |
| — | WebFetch of `www-i6.informatik.rwth-aachen.de` (K-replanner PDF) and `ai.dmi.unibas.ch` (Helmert PDF) | WebFetch | blocked (ENOTFOUND) | — | read through alphaXiv instead |

Full-text reads (alphaXiv `answer_pdf_queries`):
1. Bonet & Geffner 1909.05546 (ECAI 2020)
2. Rodriguez et al. 2105.10830 (KR 2021)
3. Bonet & Geffner 1909.13778 (causal belief decomposition)
4. K-replanner IJCAI-11 poster (Bonet's site)
5. Helmert AIJ 2009 (pp. 1–13, 32–35, 46–50)
6. Haslum & Corrêa 2411.08040
7. Occhipinti, Bonet & Geffner 2204.11902

### Snowballing (seeds)
- **S-a: Bonet & Geffner, ECAI 2020** (1909.05546).
  - Backward, from its related work: OO-MDPs; Asai 2019 (FOSAE); Konidaris-style PDDL from options.
  - Forward: Rodriguez et al. KR 2021; Occhipinti et al. 2022; Gösgens et al. 2024 [RS1].
- **S-b: Rodriguez, Bonet, Romero & Geffner, KR 2021** (2105.10830).
  - Backward, from its related work: generalized planning (Srivastava, Immerman & Zilberstein 2008; Bonet, Palacios & Geffner 2009; Hu & De Giacomo 2011; Belle & Levesque 2016; Francès, Bonet & Geffner 2021). These cover sub-question (d).
  - Forward: Occhipinti et al. (it builds on the KR-21 ASP code).
- **S-c: Bonet & Geffner, IJCAI 2011, K-replanner.**
  - Backward, from the poster's "Related Work and Ideas": Koenig's freespace assumption; Likhachev & Stentz's preferred values; Albore & Bertoli's safe assumptions; Petrick & Bacchus (PKS).
  - Forward: causal belief tracking / beam tracking (1909.13778), which explicitly compares itself to the K-replanner on Wumpus; LW1 (AAAI 2014); PO-PRP [prior notes].
- **S-d: Bnaya, Felner & Shimony, IJCAI 2009, "CTP with Remote Sensing".**
  - Backward: Papadimitriou & Yannakakis 1991.
  - Forward (alphaXiv A2): k-CTP on outerplanar graphs (2403.01872); tight competitive ratio on unit grids (2610.13712); scout-assisted robot teams (2605.22693, excluded under E1).

### PRISMA-style counts (approximate, deduplicated by hand)
- **Identified:** 182 (139 WebSearch + 43 alphaXiv), plus 34 carried from RS1, for 216 in total.
- **Duplicates or mirrors removed:** about 38. For example: Rodriguez KR-21 ×7 mirrors; Helmert AIJ-09 ×5; CTP Wikipedia ×3; 1909.05546 also in RS1; 2602.23148 ×2.
- **Screened:** about 178.
- **Excluded at screening:** about 132. Reasons:
  - E1: robotics or motion (e.g. 2605.22693, 2609.19541, 2608.00792);
  - E4: Minesweeper combinatorics or NN solvers (2506.01634, 2404.14519, 2212.10446, 2311.07387);
  - off-topic (e.g. 2510.03845, VLM predicate invention papers 2410.23156, 2501.00296, 2511.18203);
  - E5.
- **Assessed for eligibility:** 46.
- **Included in RS2 synthesis:** 30 (table in Q6). 7 were read in full or in targeted full-text passages this session.

### Gaps
- Citation indexes (Semantic Scholar, OpenAlex, DBLP) were not retried after RS1 found them blocked. Forward snowballing is again approximate.
- Two university hosts were DNS-blocked for WebFetch. alphaXiv served both PDFs, but the full K-replanner paper was not read, only the poster.

---

## Q2. Who has said that static facts define or select an instance's transition system, given a domain?

### Takeaway
Several works state this, explicitly and formally, at the **meta-structure** level (component i), but only in offline settings where the facts are given or learned in batch:
- **Helmert (AIJ 2009).** Static facts decide which ground operators exist; grounding over static preconditions is a CSP whose constraints are the static facts.
- **Bonet & Geffner (ECAI 2020).** Static atoms "control the grounding", and absent ground actions are explained by static predicates filtering them out.
- **Rodriguez et al. (KR 2021).** Domain D plus instance information I "determine state space graphs". The "verification problem" fixes D and infers only I.
- **Haslum & Corrêa (2024).** In the extreme case, static facts alone can encode any propositional transition system under one universal lifted domain.

So component (i) of D2 is **not new**. It is well established and can be cited.

### Cited Findings
- **Helmert 2009, static predicates as implicit types.** The unary predicates `airplane`, `airport`, `city`, etc. "are used as 'implicit types': they are static (do not appear in operator effects) and serve to restrict the possible instantiations of the operator parameters" [full text]. — [Helmert, AIJ 173:503–535, 2009](https://ai.dmi.unibas.ch/papers/helmert-aij2009.pdf)
- **Helmert 2009, static preconditions as a CSP.** For an operator whose precondition is a conjunction of static atoms, this "defines a constraint satisfaction problem … the constraint schemas are given by the facts for the static predicates in the initial state". Deciding whether a valid instantiation exists is NP-complete, and enumerating all instantiations is #P-hard even in restricted cases [full text]. — [Helmert 2009](https://ai.dmi.unibas.ch/papers/helmert-aij2009.pdf)
- **Helmert 2009, constant atoms compiled away.** Ground atoms are partitioned into fluent atoms Pf and "constant predicates" Pc. Conditions on Pc atoms "have constant truth values", so operators whose condition becomes false are removed [full text]. In other words, the static facts prune the ground operator set and thereby fix the FDR state space S(Π). — [Helmert 2009](https://ai.dmi.unibas.ch/papers/helmert-aij2009.pdf)
- **Helmert 2009, no domain/instance distinction after grounding.** "There is no distinction between domain constants and objects of the problem instance, or indeed between the domain and problem instance specification in general. At the level of individual problem instances … there is no need for such a distinction" [full text]. — [Helmert 2009](https://ai.dmi.unibas.ch/papers/helmert-aij2009.pdf)
- **Bonet & Geffner 2020, static atoms control grounding.** "A predicate p is static if all p-atoms are static, meaning that they do not appear in the effects of any action. Static atoms are used as preconditions to control the grounding of action schemas, and in the SAT encoding, they are treated differently than the other (fluent) atoms" [full text]. — [arXiv 1909.05546](https://www.alphaxiv.org/abs/1909.05546); [ECAI PDF](https://ecai2020.eu/papers/894_paper.pdf)
- **Bonet & Geffner 2020, absent ground actions.** "If a ground action a(ō) is never applied … it must be because the static predicates filter it out" (discussion of formula 58) [full text]. — [arXiv 1909.05546](https://www.alphaxiv.org/abs/1909.05546)
- **Rodriguez et al. 2021, domain plus instance determine the graph.** The learning problem is the "search for a simplest first-order domain description D that along with information about instances Iᵢ (number of objects and initial state) determine state space graphs G(Pᵢ) that match the observed state graphs" [full text]. — [KR 2021, pp. 539–548](https://proceedings.kr.org/2021/51/); [arXiv 2105.10830](https://www.alphaxiv.org/abs/2105.10830)
- **Rodriguez et al. 2021, the verification problem.** "The verification problem is actually a subproblem of the learning problem where the domain D is not learned but fixed, and just the instance information needs to be inferred" (Definition 2) [full text]. — [arXiv 2105.10830](https://www.alphaxiv.org/abs/2105.10830)
- **Rodriguez et al. 2021, static atoms are state-independent.** "The static predicates and atoms refer to those that control the grounding; i.e., they appear in the precondition of action schemas but do not appear in the effects, so their value do not depend on the state." The ASP code gives static atoms a single valuation `val((P,OO),V)` with no state argument [full text]. — [arXiv 2105.10830](https://www.alphaxiv.org/abs/2105.10830)
- **Haslum & Corrêa 2024, the universal domain.** A single lifted domain exists in which "any propositional planning problem instance, from any domain, becomes an instance". The ground actions are encoded as initial-state facts `(pre a p)`, `(add a p)`, `(del a p)`, and "all predicates in effect conditions are static". Consequence: one action schema and one fluent predicate suffice for PSPACE-hardness, and generalized planning over PDDL's notion of domain is "in the general case … futile" [full text]. — [arXiv 2411.08040](https://www.alphaxiv.org/abs/2411.08040)
- **Haslum & Corrêa 2024, PDDL's notion of domain.** The paper also reports that Grundke, Röger & Helmert (ICAPS 2024) and Haslum & Scholz (2003) argue that "PDDL's notion of domain is too weak" to state constraints on valid initial states. — [arXiv 2411.08040](https://www.alphaxiv.org/abs/2411.08040) (secondary mention; Grundke et al. not read)
- **TIM (Fox & Long 1998).** TIM infers types and state invariants with no annotation, through finite state machines per type. "Type restrictions on the parameters of an operator are transformed into additional static preconditions on the operators." STAN uses the inferred structure to prune action instances. Long's 2003 overview also mentions symmetric objects used for static and dynamic pruning [abstract/snippet]. — [JAIR 9:367–421](https://jair.org/index.php/jair/article/view/10216); [AIPS-00](https://cdn.aaai.org/AIPS/2000/AIPS00-021.pdf); [Long 2003 abstract](https://arw.csc.liv.ac.uk/year/2003/abstracts/long.pdf)
- **Lauer et al., IJCAI 2021.** A lifted delete-relaxation heuristic polynomial in the PDDL input size "leverages some arity-two information from static predicates" [snippet]. Static facts are used as instance structure inside lifted heuristics. — [IJCAI 2021 PDF](https://homes.cs.aau.dk/~alto/papers/ijcai21.pdf)

### Inferences
- Component (i) of D2, "given fixed templates, static facts select the transition system", has at least four explicit formal precedents. The strongest is Haslum & Corrêa: static facts can carry the *entire* transition system, so "the static facts select the TS" holds in a maximally strong sense. A corollary for the thesis: without restrictions on which static configurations are admissible, a lifted domain imposes almost no constraint on the induced TS. This is an argument for D2 to name an admissible class of static configurations, which is the point Grundke et al. raise.
- Rodriguez et al.'s "verification problem" (D fixed, infer I from the graph) is the **offline, full-graph twin** of D2's online identification. This is the single most important citation for repositioning D2.

### Gaps
- The phrase "meta-structure" itself was not found in any planning paper retrieved here. The concept is present, but the label appears to be the thesis's own.
- I did not read Grundke, Röger & Helmert (ICAPS 2024), "Formal representations of classical planning domains". It may formalize classes of instances by constraints on static facts and should be read.

---

## Q3. Planning with hidden static variables: what is assumed known (templates) and what is unknown (static facts)?

### Takeaway
Planning with hidden static structure is a mature topic. The main lines are:
- **Contingent planning:** K-replanner benchmarks such as freespace, doors, wumpus and trail; Minesweeper, Battleship and Wumpus in causal belief tracking.
- **Canadian Traveller Problem:** a known graph with hidden, static edge status.
- **Unknown-graph exploration.**

The action model is always known. The hidden part is a set of ground variables that never change, and the field already exploits their staticness. In causal belief tracking, "static variables are memory variables", and that is what makes the algorithm complete. The K-replanner's KL/K¬L knowledge literals are a (Π⁺,Π⁻)-style representation. However, these works are **propositional or graph-specific**: none phrases the hidden part as "the static predicates of a lifted domain that select the TS".

### Cited Findings
- **K-replanner, "simple" problems.** Two restrictions: "non-unary clauses in I are invariant" and "no hidden fluent appears in the body of a conditional effect". Theorem: "reachable beliefs b for simple problems represented by just the literals true in b + invariants" [full text of poster]. — [IJCAI-11 poster](https://bonetblai.github.io/reports/IJCAI11-replanning-poster.pdf); [IJCAI 2011 proceedings](https://ijcai.org/proceedings/2011/324)
- **K-replanner, translation K(P).** Fluents F′ = {KL, K¬L}; supports, cancellations and invariant operators; and sensor "assumption" actions A(C,L) with precondition KC, ¬KL, ¬K¬L. The replanner executes the plan prefix up to a goal or a refuting observation, adds the observations, and closes the state under the invariants [full text of poster]. — [poster](https://bonetblai.github.io/reports/IJCAI11-replanning-poster.pdf)
- **K-replanner, guarantees.** It is sound for simple problems, and complete "if P is a simple problem with a connected space and the non-unary clauses in I are in prime implicate form" [full text of poster]. — [poster](https://bonetblai.github.io/reports/IJCAI11-replanning-poster.pdf)
- **K-replanner, benchmarks.** The hidden information is static structure. In freespace, "each cell of the grid is either blocked or clear. As the agent moves, it senses the status of adjacent cells". Doors have positions only partially known. Wumpus positions are sensed by stench. In trail, "the shape and length of the trail are not known". The poster lists Koenig's freespace assumption as a related idea [full text of poster]. — [poster](https://bonetblai.github.io/reports/IJCAI11-replanning-poster.pdf)
- **Correction to an earlier note.** The "simple" condition does **not** literally say that hidden fluents are static. It says that hidden fluents do not appear in conditional-effect bodies and that non-unary initial clauses are invariant. In practice the hidden fluents in the benchmarks are static. — [poster](https://bonetblai.github.io/reports/IJCAI11-replanning-poster.pdf)
- **Causal belief tracking, memory variables.** A state variable is a *memory variable* if its value at any time can be determined from its value at any other time, the actions and the initial belief. "Thus, static variables are memory variables." CBT is complete for "causally decomposable" problems, where beams intersect only on memory variables (Theorem 18). Minesweeper, Battleship and Wumpus fall in this class [full text]. — [Bonet & Geffner, arXiv 1909.13778](https://www.alphaxiv.org/abs/1909.13778) (venue IJCAI 2013 [memory, unverified])
- **Causal belief tracking, encodings.** Minesweeper uses "3mn boolean state variables mine_{x,y}, opened_{x,y}, flagged_{x,y}". Wumpus uses "hidden boolean variables for each cell that tell whether there is a pit, a wumpus, or nothing" and a hidden gold position. Battleship uses a hidden ship size and orientation per cell, with state constraints between neighbouring cells [full text]. — [arXiv 1909.13778](https://www.alphaxiv.org/abs/1909.13778)
- **Causal belief tracking, scale.** Beam tracking solves Wumpus grids with n > 100 in real time. "The K-replanner manages to scale up to n = 40" on the simplified Wumpus [full text]. — [arXiv 1909.13778](https://www.alphaxiv.org/abs/1909.13778)
- **Causal belief tracking, "determined" variables.** A set of variables is *determined* when it is initially known, appears only in deterministic effect heads, and so on. Determined variables do not count toward width [full text]. This is a formal width notion in which known static facts cost nothing. — [arXiv 1909.13778](https://www.alphaxiv.org/abs/1909.13778)
- **CTP with remote sensing.** "The agent starts with a known graph and some edges may be blocked with known probability". Sensing can be done remotely at a cost. Optimal policies are intractable, so heuristics are used [abstract/snippet]. — [Bnaya, Felner & Shimony, IJCAI 2009](https://www.ijcai.org/Abstract/09/080)
- **CTP origin.** Papadimitriou & Yannakakis, "Shortest paths without a map", TCS 84:127–150 (1991; ICALP 1989 version). They show the adversarial variant with an unbounded number of blockages to be PSPACE-complete. Secondary sources disagree on which variant is #P-hard versus PSPACE-complete [snippet; conflicting secondary sources]. — [MaRDI record](https://portal.mardi4nfdi.de/wiki/Publication:809612); [UCSB copy](https://sites.cs.ucsb.edu/~suri/cs235/Rlist/noMapNavigation.pdf); [Wikipedia (secondary)](https://en.wikipedia.org/wiki/Canadian_traveller_problem)
- **k-CTP.** The k-CTP formulation has "a hidden input E* ⊊ E" of blocked edges, revealed online [abstract]. — [arXiv 2403.01872](https://www.alphaxiv.org/abs/2403.01872); [arXiv 2610.13712](https://www.alphaxiv.org/abs/2610.13712)
- **Unknown-graph exploration.** Deng & Papadimitriou (FOCS 1990; J. Graph Theory 32(3), 1999) study exploring every edge of an unknown, strongly connected directed graph. The competitive ratio is 2 for Eulerian graphs and is bounded when the deficiency is bounded [snippet]. — [arXiv 2002.10958 (citing work)](https://arxiv.org/pdf/2002.10958); [JGT record](https://acnpsearch.tweb-dev.unibo.it/singlejournalindex/66105265)
- **Latent graphs with aliased observations.** Recovering a latent graph from aliased observations while traversing it is a non-symbolic analogue [abstract]. — [arXiv 2303.07397](https://www.alphaxiv.org/abs/2303.07397)

### Inferences
- **Component (ii), hidden static facts, is standard.** Every one of these lines assumes known dynamics and unknown, fixed instance structure. The novelty claim for (ii) cannot be "hidden static facts" as such.
- **Component (iii), online identification with (Π⁺,Π⁻), has a near-isomorphic mechanism in the K-replanner.**
  - KL/K¬L literals restricted to static atoms are exactly a pair (Π⁺,Π⁻).
  - O_inv plus sensors (C,L) play the role of observation rules.
  - The completeness condition (connected space, prime implicates) is an identifiability-flavoured result.
  - CBT's "static ⇒ memory variable ⇒ complete decomposition" is a formal statement that staticness makes belief tracking easier, which D2 may be implicitly re-deriving.
- **What remains distinctive for D2** (to verify against the thesis text): the hidden variables are explicitly the *lifted static predicates* of the domain, with the static/fluent split read off the templates. The observation rules may also be derived from the templates (applicability or effects of actions), rather than given as hand-written sensors (C,L).

### Gaps
- I read only the K-replanner poster, not the full paper. Whether the paper discusses static hidden fluents by name is unverified.
- Exploration under the freespace assumption (Koenig) was not re-read; see RS1.

---

## Q4. Learning the static structure (objects, static predicates such as adjacency) from the state graph or from images

### Takeaway
The Bonet–Geffner line infers objects and static predicates from the bare state graph:
- ECAI 2020: SAT;
- KR 2021: ASP;
- Occhipinti et al. 2022: parsed images (O2D).

They recover adjacency in Grid as static binary predicates and handle static atoms with a single per-instance valuation. Rodriguez et al. even isolate the sub-problem "D fixed, infer the instance". The guarantees are encoding correctness (an instance accounts for the graph iff the theory is satisfiable) and empirical generalization. The setting is offline, needs a complete graph or a dataset, and has no epistemic state or online interaction.

Asai's FOSAE and Latplan learn predicates from images but do not separate static from fluent in a symbolic way [abstract only].

### Cited Findings
- **Bonet & Geffner 2020, adjacency recovered.** In Grid with 2 labels, "the inferred static predicates B2 and B0 capture the horizontal and vertical adjacency relations respectively". In Hanoi, the static predicates BIGGER and NEQ are learned. In Gripper, the static predicates CONN and PAIR are learned [full text]. — [arXiv 1909.05546](https://www.alphaxiv.org/abs/1909.05546)
- **Bonet & Geffner 2020, hyperparameters.** The vector α includes "the total number of unary and binary static predicates". The bound used is "max number of static predicates set to 5". The SAT variables r(u,o) and s(b,o,o′) give the truth of static unary and binary atoms per instance layer [full text]. — [arXiv 1909.05546](https://www.alphaxiv.org/abs/1909.05546)
- **Bonet & Geffner 2020, guarantee.** Theorem 3: the instances Pᵢ = ⟨D,Iᵢ⟩ "account for the input labeled graphs … applying every ground action at least once iff there is a satisfying assignment". The learned domain is "correct for these instances but not necessarily for other instances" [full text]. — [arXiv 1909.05546](https://www.alphaxiv.org/abs/1909.05546)
- **Bonet & Geffner 2020, input requirement.** The input is a complete graph, not traces. The authors note that the encoding "can be adjusted … to account for incomplete graphs where only certain nodes are marked" [full text]. — [arXiv 1909.05546](https://www.alphaxiv.org/abs/1909.05546)
- **Rodriguez et al. 2021, ranking.** Solutions are ranked lexicographically by action arity, then non-static predicate arity, then **static predicate arity** (N_s), then true atoms per state. The approach extends to partial graphs and to noise that makes some states indistinguishable [full text]. — [arXiv 2105.10830](https://www.alphaxiv.org/abs/2105.10830)
- **Rodriguez et al. 2021, framing.** "This is a learning problem and not a synthesis problem; namely, the representations Pᵢ = ⟨D,Iᵢ⟩ are not deducible from the input graphs; they are inferred under suitable regularity (simplicity) assumptions" [full text]. — [arXiv 2105.10830](https://www.alphaxiv.org/abs/2105.10830)
- **Occhipinti et al. 2022, per-instance statics.** "For each instance I, we compute the predicates that are static over that instance … and encode their truth value V in all states of that instance with a single fact val(I,(P,OO2),V)". A predicate is static for the domain if it is static in all instances (`p_static(P) :- pred(P), f_static(I,P) : instance(I)`) [full text]. — [arXiv 2204.11902](https://www.alphaxiv.org/abs/2204.11902)
- **Occhipinti et al. 2022, completeness.** Completeness is stated relative to a "hidden" domain D (Theorem 4). New instances are given as pairs of parsed images. The learned Sokoban and Grid schemas use the static predicates `below`/`INV[left]` as cell adjacency [full text]. — [arXiv 2204.11902](https://www.alphaxiv.org/abs/2204.11902)
- **Asai 2019, FOSAE.** An unsupervised architecture "for grounding the first-order logic predicates and facts" from object feature vectors; object detection is out of scope. It was evaluated on the 8-puzzle and a photo-realistic Blocksworld [abstract/snippet]. — [ICAPS 2019](https://ojs.aaai.org/index.php/ICAPS/article/view/3525); [arXiv 1902.08093](https://www.alphaxiv.org/abs/1902.08093)
- **Latplan.** "Classical Planning in Deep Latent Space" learns propositional latent states and action models from images [abstract]. — [arXiv 2107.00110](https://www.alphaxiv.org/abs/2107.00110)

### Inferences
- **Component (i) appears again, plus identification.** In these works the static facts are *what is inferred per instance*, while the domain is shared. Rodriguez et al.'s verification problem is "templates known, identify instance static facts from TS structure". It is offline and needs (near-)complete graphs. D2's differential relative to this line has to be the **online, partial-information, epistemic** version: identifying statics from a single trajectory of interaction while acting, with (Π⁺,Π⁻) bookkeeping.
- None of the papers read gives identifiability conditions for the static facts alone, under known templates, from partial interaction. This remains the most defensible technical gap; it is consistent with RS1, claim (b).

### Gaps
- Ståhlberg et al. (GNN general policies, 2109.10129; symmetries, 2409.15892) were retrieved but not read. I could not verify whether they treat static predicates specially in the relational encoding.
- FOSAE and Latplan were read at abstract level only.

---

## Q5. Open world, unknown objects, and generalized planning over instances that differ in static facts

### Takeaway
- **Open-world planning** (Etzioni, Golden & Weld's LCW and XII; Talamadupula et al.'s OWQGs) handles unknown *objects* and facts through sensing and local closed-world bookkeeping. It treats completeness of knowledge as the variable, not static facts as a selector of the TS.
- **Generalized planning** (Srivastava et al.; Bonet & Geffner; Francès et al.) assumes instances share a domain and differ in objects and initial state, including static facts. Policies over features must work across all such instances. I found no explicit statement that "instances differ only by static facts".

### Cited Findings
- **LCW.** Classical planners treat absent facts as false, and open-world planners as unknown. LCW gives complete knowledge of a specified part of the world, with "tractable update rules for the case of conjunctive LCW" and a link to circumscription. XII uses LCW to avoid redundant sensing [snippet]. — [Golden, Etzioni & Weld TR 94-01-03](https://homes.cs.washington.edu/~weld/papers/tr94-01-03.pdf); [XII page](https://homes.cs.washington.edu/~etzioni/papers/xii.html); [AIJ record](https://portal.mardi4nfdi.de/wiki/Publication:1399125)
- **PSIPLAN.** Babaian & Schmolze's PSIPLAN addresses the case where the set of objects is unknown or infinite [snippet, secondary]. — [same search; AIPS-00](https://cdn.aaai.org/AIPS/2000/AIPS00-031.pdf)
- **OWQGs.** Talamadupula et al. introduce Open World Quantified Goals because "the planner … does not have complete knowledge of all objects (e.g., victims)". They argue that sensing everything first is "clearly infeasible in open worlds where we do not even know how many objects may be there" [abstract/snippet]. — [AAAI 2010](https://hrilab.tufts.edu/publications/talamadupulaetal10aaai.pdf); [ACM TIST 2010](https://hrilab.tufts.edu/publications/talamadupulaetal10tist.pdf)
- **Francès, Bonet & Geffner, AAAI 2021.** General policies are learned from small instances over a feature pool derived from the domain predicates, via Weighted Max-SAT. Instances "can have different objects and ground actions", which general features abstract away [snippet]. — [AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/17402); [arXiv 2101.00692](https://arxiv.org/abs/2101.00692v2)
- **Bonet & Geffner, IJCAI 2018.** "Features, Projections, and Representation Change for Generalized Planning": abstract actions over features are shared across instances [snippet]. — [IJCAI 2018](https://ijcai.org/proceedings/2018/649)
- **Haslum & Corrêa 2024, on generalized planning.** With PDDL's notion of domain, generalized planning is in general futile, because static facts can encode arbitrary propositional problems [full text]. — [arXiv 2411.08040](https://www.alphaxiv.org/abs/2411.08040)
- **Lineage cited by Rodriguez et al.** Generalized planning lineage: Srivastava, Immerman & Zilberstein 2008/2011; Bonet, Palacios & Geffner 2009; Hu & De Giacomo 2011; Belle & Levesque 2016 [reference list, full text]. — [arXiv 2105.10830](https://www.alphaxiv.org/abs/2105.10830)

### Inferences
- **Generalized planning is the "static facts vary across instances" setting, seen from outside.** The policy must be correct under every admissible static configuration. D2's agent faces the same family but resolves *which* member holds online. Framing D2 as "online selection within a generalized-planning family" would be accurate and well anchored.
- Haslum & Corrêa's futility result implies that D2 needs a restricted class of static configurations (e.g. grid adjacency) for identification to be tractable or meaningful.

### Gaps
- Srivastava et al. 2011 (AIJ) and Hu & De Giacomo 2011 were not opened. Their definitions of "problem class" may parametrize instances by static facts explicitly.

---

## Q6. Included studies and verdicts on D2

### Takeaway
No study is IDÊNTICO.
- Two are **SOBREPOSIÇÃO FORTE**:
  - Rodriguez et al. 2021: templates fixed, infer instance static facts so that the induced TS matches; offline.
  - K-replanner 2011: hidden static benchmark structure, KL/K¬L knowledge, sensors plus invariants, online; propositional, not lifted statics.
- Haslum & Corrêa 2024 and Bonet & Geffner 2020 are FORTE on component (i) only.
- D2 survives only as a **combination** of the three components: lifted static predicates, hidden, identified online with (Π⁺,Π⁻) and template-derived observation rules. The thesis should present it as a synthesis, citing these precedents explicitly.

### Included studies

| # | Study (bibliographic data and URL) | Read | Role of static facts | Known vs unknown | Guarantees | D2 verdict (components) |
|---|---|---|---|---|---|---|
| 1 | Rodriguez, Bonet, Romero & Geffner, "Learning First-Order Representations for Planning from Black-Box States: New Results", KR 2021, 539–548. [KR](https://proceedings.kr.org/2021/51/), [arXiv 2105.10830](https://www.alphaxiv.org/abs/2105.10830) | full | learned per instance; structural (control grounding) | Learning: nothing known. Verification: D known, I (objects, static atoms) inferred | ASP correctness theorem; optimality under lexicographic cost | **SOBREPOSIÇÃO FORTE** (i strong; iii offline, no epistemic state) |
| 2 | Bonet & Geffner, "Planning under Partial Observability by Classical Replanning: Theory and Experiments" (K-replanner), IJCAI 2011, 1936–1941. [IJCAI](https://ijcai.org/proceedings/2011/324), [poster](https://bonetblai.github.io/reports/IJCAI11-replanning-poster.pdf) | poster full | hidden (benchmarks: static cells, doors, wumpus, trail) | Actions, sensors and invariants known; hidden initial literals | sound; complete if connected space and prime implicates | **SOBREPOSIÇÃO FORTE** (ii, iii; not i at the lifted level) |
| 3 | Haslum & Corrêa, "The Universal PDDL Domain", arXiv 2411.08040 (2024) | full | structural: static facts encode the entire TS | Lifted schema fixed; static facts = ground actions | PSPACE-hardness with 1 schema + 1 fluent predicate; no polynomial lifted universal domain | **SOBREPOSIÇÃO FORTE on (i)**; DISTANTE on ii/iii |
| 4 | Bonet & Geffner, "Learning First-Order Symbolic Representations for Planning from the Structure of the State Space", ECAI 2020. [arXiv 1909.05546](https://www.alphaxiv.org/abs/1909.05546) | full | learned; static atoms control grounding | Nothing known; complete graph given | SAT correctness theorem (Th. 3) | FORTE on (i); PARCIAL overall |
| 5 | Occhipinti, Bonet & Geffner, "Learning First-Order Symbolic Planning Representations That Are Grounded", [arXiv 2204.11902](https://www.alphaxiv.org/abs/2204.11902) (2022) | targeted full | learned per instance (f_static per instance) | Pool of O2D predicates known; domain learned | completeness w.r.t. a hidden domain (Th. 4) | PARCIAL |
| 6 | Bonet & Geffner, "Causal Belief Decomposition for Planning with Sensing", [arXiv 1909.13778](https://www.alphaxiv.org/abs/1909.13778) (IJCAI 2013 [memory, unverified]) | full | hidden; "static variables are memory variables" | Model known; hidden static cell contents | CBT sound, complete for causally decomposable problems | **PARCIAL→FORTE** on (ii) and the formal role of staticness; not lifted |
| 7 | Helmert, "Concise finite-domain representations for PDDL planning tasks", AIJ 173 (2009) 503–535. [PDF](https://ai.dmi.unibas.ch/papers/helmert-aij2009.pdf) | targeted full | given; structural (implicit types, CSP grounding, constant atoms removed) | All known | NP-/#P-hardness of static-precondition grounding | PARCIAL (i only) |
| 8 | Fox & Long, "The Automatic Inference of State Invariants in TIM", JAIR 9 (1998) 367–421. [JAIR](https://jair.org/index.php/jair/article/view/10216) | snippet | structural (types → static preconditions; invariants; symmetry) | All known | — | PARCIAL (i) |
| 9 | Fox & Long, TIM extensions, AIPS 2000. [PDF](https://cdn.aaai.org/AIPS/2000/AIPS00-021.pdf); Cresswell, Fox & Long, ADL TIM. [Strathprints](https://strathprints.strath.ac.uk/31981) | snippet | structural | All known | — | DISTANTE |
| 10 | Lauer et al., "Polynomial-time in PDDL input size: making the delete relaxation feasible for lifted planning", IJCAI 2021. [PDF](https://homes.cs.aau.dk/~alto/papers/ijcai21.pdf) | snippet | structural (arity-2 static info in lifted heuristic) | All known | tractability for at-most-unary delete-free | DISTANTE |
| 11 | Bnaya, Felner & Shimony, "Canadian Traveler Problem with Remote Sensing", IJCAI 2009. [IJCAI](https://www.ijcai.org/Abstract/09/080) | abstract | hidden (edge blocked status, static) | Graph and probabilities known; status unknown | intractability; heuristics; special-case optima | PARCIAL (ii, online sensing; not lifted, no templates) |
| 12 | Papadimitriou & Yannakakis, "Shortest paths without a map", TCS 84 (1991) 127–150. [MaRDI](https://portal.mardi4nfdi.de/wiki/Publication:809612) | snippet | hidden map | Partial map | PSPACE-completeness (variant-dependent; sources conflict) | PARCIAL (ii) |
| 13 | k-CTP on outerplanar graphs, [arXiv 2403.01872](https://www.alphaxiv.org/abs/2403.01872); unit-weighted grids, [arXiv 2610.13712](https://www.alphaxiv.org/abs/2610.13712) | abstract | hidden E* ⊊ E | Graph known | competitive ratios | DISTANTE |
| 14 | Deng & Papadimitriou, "Exploring an unknown graph", FOCS 1990 / JGT 32(3) 1999. [JGT](https://acnpsearch.tweb-dev.unibo.it/singlejournalindex/66105265) | snippet | the whole graph is unknown | Nothing about the graph | competitive ratio 2 (Eulerian); bounded under bounded deficiency | PARCIAL (ii, iii-as-exploration; no templates) |
| 15 | Fast exploration and learning of latent graphs with aliased observations, [arXiv 2303.07397](https://www.alphaxiv.org/abs/2303.07397) | abstract | latent graph learned | Observations aliased | — | DISTANTE |
| 16 | Golden, Etzioni & Weld, LCW / XII (AAAI-94; AIJ). [TR](https://homes.cs.washington.edu/~weld/papers/tr94-01-03.pdf) | snippet | unknown facts and objects; local completeness | Actions known | tractable LCW updates | PARCIAL (ii, iii with knowledge bookkeeping; not static-specific) |
| 17 | Talamadupula et al., AAAI 2010 [PDF](https://hrilab.tufts.edu/publications/talamadupulaetal10aaai.pdf); ACM TIST 1(2) 2010 [PDF](https://hrilab.tufts.edu/publications/talamadupulaetal10tist.pdf) | abstract | unknown objects (OWQG) | Domain known | — | DISTANTE→PARCIAL |
| 18 | Babaian & Schmolze, PSIPLAN (unknown or infinite object sets) | secondary snippet | unknown objects | Domain known | — | DISTANTE |
| 19 | Asai, "Unsupervised Grounding of Plannable First-Order Logic Representation from Images" (FOSAE), ICAPS 2019, 583–591. [ICAPS](https://ojs.aaai.org/index.php/ICAPS/article/view/3525) | abstract | predicates learned from images | Objects given as feature vectors | — | DISTANTE |
| 20 | Asai et al., "Classical Planning in Deep Latent Space" (Latplan), [arXiv 2107.00110](https://www.alphaxiv.org/abs/2107.00110) | abstract | propositional latent | Nothing known | — | DISTANTE |
| 21 | Francès, Bonet & Geffner, "Learning General Planning Policies from Small Examples Without Supervision", AAAI 2021. [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/17402) | snippet | vary across instances (implicit) | Domain known | policies verified on instances | PARCIAL (i-as-family) |
| 22 | Bonet & Geffner, "Features, Projections, and Representation Change for Generalized Planning", IJCAI 2018. [IJCAI](https://ijcai.org/proceedings/2018/649) | snippet | implicit | Domain known | soundness of projections [memory, unverified] | DISTANTE |
| 23 | Bonet & Geffner, "General Policies, Representations, and Planning Width", AAAI 2021. [mlanthology](https://mlanthology.org/aaai/2021/bonet2021aaai-general) | snippet | implicit | Domain known | width ↔ general policies | DISTANTE |
| 24 | Ståhlberg, Bonet & Geffner, GNN general policies, [arXiv 2109.10129](https://www.alphaxiv.org/abs/2109.10129); symmetries, [arXiv 2409.15892](https://www.alphaxiv.org/abs/2409.15892) | title/abstract | relational encoding of instances (static handling unverified) | Domain known | expressivity (C2 logic) [memory, unverified] | DISTANTE |
| 25 | Sample-efficient generalized planning via learned transition models, [arXiv 2602.23148](https://www.alphaxiv.org/abs/2602.23148) | abstract | shared transition function γ across instances | — | — | DISTANTE |
| 26 | Bonet & Geffner, LW1, AAAI 2014. [OJS](https://ojs.aaai.org/index.php/AAAI/article/view/9047) | prior notes | hidden | Model known | complete for width 1 | PARCIAL (ii, iii) |
| 27 | Brafman & Shani, SDR, JAIR 2012 [RS1] | RS1 | hidden initial state | Model known | — | PARCIAL |
| 28 | Koenig & Smirnov, freespace assumption, ICRA 1997 [RS1] | RS1 | hidden static map | Dynamics known | — | PARCIAL |
| 29 | Gregory & Cresswell, LOP, ICAPS 2015 [RS1] | RS1 | static relations learned offline (domain-level) | Traces | — | PARCIAL |
| 30 | Grundke, Röger & Helmert, "Formal representations of classical planning domains", ICAPS 2024 | reference only (via #3) | valid-instance constraints | — | — | not graded (to read) |

### Inferences
- **Component (i), static facts as meta-structure selecting the TS given templates.** Established: Helmert 2009; Bonet & Geffner 2020; Rodriguez et al. 2021; Haslum & Corrêa 2024. *Not new.*
- **Component (ii), hidden static facts.** Established: CTP, K-replanner benchmarks, CBT/beam tracking, freespace, Minesweeper/Battleship/Wumpus. *Not new.*
- **Component (iii), identified online with a two-sided knowledge state.** The mechanism exists in propositional form: K-replanner KL/K¬L plus sensors, invariants and replanning; LCW in open-world planning. *Not new as a mechanism.*
- **Plausibly new: the specific conjunction.** Hidden facts = the static predicates of a *lifted* domain, syntactically identified from the templates. Observation rules derived from the templates. (Π⁺,Π⁻) restricted to that static layer. The goal is to identify which member of the template-induced TS family holds, during execution. I found no paper that states this conjunction. The closest pair is **Rodriguez et al.'s verification problem (offline) + K-replanner (online, propositional)**.
- **Recommended wording for the thesis.** "D2 combines the view of static facts as instance information that, together with a fixed domain, determines the state graph (Helmert 2009; Bonet & Geffner 2020; Rodriguez et al. 2021; Haslum & Corrêa 2024) with online knowledge-level identification of hidden static structure (K-replanner; causal belief tracking; CTP), lifting the latter to the static predicates of a PDDL domain." Avoid "new view of static facts as meta-structure".

### Gaps
- No item was rated IDÊNTICO, but recall is limited (see Q7). The thesis text itself was not available to check whether its observation rules are template-derived, which is the point on which the distinction from the K-replanner rests.

---

## Q7. Limitations

### Takeaway
The negative finding (no identical prior work) has moderate confidence. The positive findings (strong prior art for each component) have high confidence, because the key ones were read in full text.

### Cited Findings
- The WebFetch failures (ENOTFOUND for two university hosts) are recorded in this session's tool log. There is no external source.

### Inferences
- **Single reviewer**, with no second screener.
- **Recall is unknown.** Web search returns about 10 results per query, and citation indexes were not used.
- **Full-text coverage is partial.** 7 sources were read in full or in targeted passages; the remaining verdicts rest on abstracts or snippets. The K-replanner paper was read only through its poster.
- **Terminology.** "Rigid" predicates (situation calculus, KR), "constant predicates" (Helmert), "memory variables" (CBT), "determined variables", "implicit types" and "static atoms" all name overlapping concepts. A dedicated "rigid fluents/predicates" query in KR and epistemic planning was not run.
- **Unverified metadata:** CBT venue and year; Occhipinti venue; Ståhlberg contents; the P&Y complexity results per variant (secondary sources conflict).

### Gaps
- Follow-ups:
  1. Read the full K-replanner paper (IJCAI 2011) and Grundke, Röger & Helmert (ICAPS 2024).
  2. Read Srivastava, Immerman & Zilberstein (AIJ 2011) for "problem class" definitions.
  3. Run queries on "rigid predicates" plus planning/epistemic, and on "identifiability of initial state" in contingent planning.
  4. Do Google Scholar forward citations of Rodriguez et al. 2021, filtering for "online", "partial observability" and "verification".
