# Linhagem das traduções (compilações) K0 / K_{T,M} / K_i e dos parâmetros estruturais de largura

> Escopo: genealogia técnica a partir de Palacios & Geffner (JAIR 2009). O relatório anterior (`reports/Planejamento contingente estado da arte.md`) já cobre complexidade, desempenho empírico de SDR/CLG/K-replanner/beam tracking e a equivalência (Γ⁺,Γ⁻) ≡ K0. Estas notas se concentram no que cada trabalho acrescenta à ideia de etiquetas (*tags*), *merges* e largura, e em qual condição de completude cada um usa.
> Limitação de acesso: dblp.org, arxiv.org, bonetblai.github.io e www-i6.informatik.rwth-aachen.de (página de Geffner) não resolvem nesta sessão. Os dados bibliográficos vêm de WebSearch (OJS AAAI, IJCAI, JAIR, mlanthology, CRIS) e do texto integral do JAIR 2009 obtido via alphaXiv. Para conferência final, é preciso liberar esses domínios.

## 1. O núcleo: Palacios & Geffner 2006 → 2007 → JAIR 2009 (K0, K_{T,M}, K_S0, K_models, K_i, largura conformante)

### Takeaway
A ideia nasce em AAAI 2006 (K0, correto mas incompleto), ganha o esquema geral com etiquetas e *merges* e as condições de completude em ICAPS 2007, e se consolida no JAIR 2009. Nele, a largura conformante w(P) é definida via coberturas de cláusulas relevantes, e K_i é correto, polinomial para i fixo e completo quando w(P) ≤ i. O planejador T0, baseado nessa tradução, venceu a trilha conformante do IPC5 (2006).

### Cited Findings
- **AAAI 2006**: H. Palacios & H. Geffner, "Compiling Uncertainty Away: Solving Conformant Planning Problems using a Classical Planner (Sometimes)", Proc. AAAI-06 (21st National Conference on AI). Introduz o mapeamento de fluentes em contrapartes epistêmicas KL/K¬L. É correto, mas incompleto ("sometimes"). Páginas não verificadas. — [ResearchGate](https://www.researchgate.net/publication/221603901_Compiling_Uncertainty_Away_Solving_Conformant_Planning_Problems_using_a_Classical_Planner_Sometimes); [Semantic Scholar](https://www.semanticscholar.org/paper/Compiling-Uncertainty-Away:-Solving-Conformant-a-Palacios-Geffner/0f8bb5fb350920f4f0493280f32813324272f348)
- **ICAPS 2007**: H. Palacios & H. Geffner, "From Conformant into Classical Planning: Efficient Translations that may be Complete Too", Proc. 17th ICAPS, Providence, set. 2007. As páginas não foram verificadas; o PDF da AAAI é ICAPS07-034. — [PDF AAAI](https://cdn.aaai.org/ICAPS/2007/ICAPS07-034.pdf); [versão DC ICAPS07](https://icaps07-satellite.icaps-conference.org/dc/dc-21.pdf)
- **JAIR 2009**: H. Palacios & H. Geffner, "Compiling Uncertainty Away in Conformant Planning Problems with Bounded Width", *JAIR* 35 (2009) 623–675, doi:10.1613/jair.2708; submetido em 10/2008, publicado em 08/2009. — [texto via alphaXiv/arXiv 1401.3468](https://www.alphaxiv.org/abs/1401.3468); [JAIR](https://jair.org/index.php/jair/article/view/10618); [mlanthology](https://mlanthology.org/jair/2009/palacios2009jair-compiling)
- O resumo do JAIR 2009 diz que a tradução "maps literals L and sets of assumptions t about the initial situation, into new literals KL/t that represent that L must be true if t is initially true", e que a tradução completa é "exponential in a parameter of the problem called the conformant width, which for most benchmarks is bounded". — [alphaXiv 1401.3468](https://www.alphaxiv.org/abs/1401.3468)
- **Esquema K_{T,M}**: tem dois parâmetros, um conjunto T de etiquetas e um conjunto M de *merges*. Um *merge* m é uma coleção de etiquetas que representa uma disjunção e é **válido** se uma das etiquetas é verdadeira em todo estado inicial possível. Ele se traduz em uma "merge action" que deriva KL a partir de KL/t para todo t ∈ m (Defs. 1 e 4). O Teorema 7 garante correção se todos os *merges* são válidos e todas as etiquetas consistentes. — [JAIR 2009](https://www.alphaxiv.org/abs/1401.3468)
- **K_S0**: as etiquetas são os estados iniciais possíveis e há um único *merge*. É completo (Teorema 9), mas exponencial. — [JAIR 2009](https://www.alphaxiv.org/abs/1401.3468)
- **Relevância conformante** (Def. 10) e **cláusulas relevantes** C_I(L) (Def. 11). **Merges de cobertura** (Def. 13) e **tradução de cobertura** (Def. 14). O Teorema 15 diz que traduções de cobertura são completas. **K_models** (Def. 16, Teorema 17) é correto e completo, mas exponencial no pior caso. — [JAIR 2009](https://www.alphaxiv.org/abs/1401.3468)
- **Largura conformante**: a cobertura c(C) é o conjunto de conjuntos mínimos de literais consistentes com I que contêm um literal de cada cláusula de C (Def. 18). w(L) é o tamanho do menor conjunto C ⊆ C*_I(L) cuja cobertura satisfaz C_I(L) (Def. 19). w(P) = max_L w(L) sobre os literais de precondição e de meta (Def. 20). w(P) é computável em tempo exponencial em w(P) (Prop. 21) e 0 ≤ w(P) ≤ n, com n o número de fluentes inicialmente desconhecidos (Prop. 22). A analogia explícita é com a largura (*treewidth*) de modelos gráficos (Dechter 2003). — [JAIR 2009](https://www.alphaxiv.org/abs/1401.3468)
- **K_i** (Def. 23, Teorema 24): para i fixo, K_i é correto e polinomial e, se w(P) ≤ i, é de cobertura e completo. Quase todos os benchmarks têm largura 1. — [JAIR 2009](https://www.alphaxiv.org/abs/1401.3468)
- **K0** é o caso sem etiquetas. A Proposição 3 (equivalência com a aproximação-0 de Son & Baral) está detalhada no relatório anterior. — [JAIR 2009](https://www.alphaxiv.org/abs/1401.3468)
- **Ações não determinísticas** (Seção 10): um efeito a: C → oneof(S1,…,Sm) vira as regras determinísticas a: C, h_i → S_i, com oneof(h1,…,hm) acrescentado ao estado inicial. A equivalência só vale para planos que executam cada ação não determinística **no máximo uma vez**. Os autores apresentam isso "as a matter of illustration only". — [JAIR 2009](https://www.alphaxiv.org/abs/1401.3468)
- **T0 nos IPCs**: T0 foi "the winning entry" na trilha conformante do IPC5 (2006). No IPC6 (2008), uma reimplementação usou uma combinação de K0 com K1, só para literais com w(L) = 1. Um bug impediu que Forest e Dispose fossem reconhecidos como largura 1, e o resultado foi recorrer a K_models. A versão do artigo usa K1 e recorre a K_models quando a busca falha. Os concorrentes no IPC6 foram CpA(H) e CpA(C) (Tran, Nguyen, Pontelli & Son). — [JAIR 2009, §8–9](https://www.alphaxiv.org/abs/1401.3468); [descrição do T0 no IPC6](https://ipc08.icaps-conference.org/probabilistic/wiki/images/7/7a/Team10-T0.pdf); [página UPF do T0](https://www.upf.edu/web/ai-ml/t0-conformant-planner)

### Inferences
- A cadeia de condições de completude é: K_S0 (sempre completo), traduções de cobertura (Teorema 15), K_models e K_i (completo se w(P) ≤ i). A largura é uma medida **sintática e estática** sobre I e a relação de relevância. Ela não depende de observações, o que explica por que os sucessores precisaram redefini-la para sensoriamento.

### Gaps
- Páginas de AAAI 2006 e ICAPS 2007 não verificadas (dblp inacessível).

## 2. Variantes conformantes e usos derivados (2009–2013)

### Takeaway
A família de traduções gerou K_Si/T1 (sempre completa, correta para largura limitada), controladores de estados finitos via tradução conformante (cp2fsc) e a compilação de planejamento conformante probabilístico.

### Cited Findings
- **Albore, Ramírez & Geffner**, "Effective Heuristics and Belief Tracking for Planning with Incomplete Information", ICAPS 2011, 21(1): 2–9, doi:10.1609/icaps.v21i1.13473. O artigo aponta três fraquezas da tradução: traduções completas são caras quando a largura é alta, traduções incompletas dão heurística infinita em problemas solúveis, e a cardinalidade da crença é ignorada. Propõe a tradução **K_Si**, "always complete and sound for problems of bounded width" (formulação do buscador). O planejador **T1** usa i = 1, com uma heurística extra baseada em invariantes one-of. — [OJS](https://ojs.aaai.org/index.php/ICAPS/article/view/13473)
- **Bonet, Palacios & Geffner**, "Automatic Derivation of Memoryless Policies and Finite-State Controllers Using Classical Planners", ICAPS 2009, pp. 34–41, doi:10.1609/icaps.v19i1.13379. A síntese de controladores é reformulada como problema conformante e resolvida por planejador clássico via a tradução completa de Palacios & Geffner. É a origem do repositório cp2fsc. — [OJS](https://ojs.aaai.org/index.php/ICAPS/article/view/13379); [PDF](https://www-i6.informatik.rwth-aachen.de/~hector.geffner/www.dtic.upf.edu/~hgeffner/blai-icaps09.pdf)
- Resumo Nectar: "Automatic Derivation of Finite-State Machines for Behavior Control", AAAI-10. — [OJS](https://ojs.aaai.org/index.php/AAAI/article/view/7706)
- **Taig & Brafman**, "Compiling Conformant Probabilistic Planning Problems into Classical Planning", ICAPS 2013, 23: 197–205, doi:10.1609/icaps.v23i1.13540. Estende a ideia de compilação para o caso probabilístico com limiar θ. — [referência em Azran et al. 2026](https://www.alphaxiv.org/abs/2609.16884)
- Tutorial: Palacios & Albore, "Translation-based approaches to conformant and contingent planning", ICAPS 2011. — [VideoLectures](https://videolectures.net/icaps2011_palacios_albore_planning)

### Gaps
- Não verifiquei o enunciado formal exato de K_Si (abstract apenas).

## 3. Planejamento contingente com sensoriamento: CLG, K-replanner, LW1, PO-PRP

### Takeaway
CLG (IJCAI 2009) leva o esquema para o caso contingente, com uma tradução X(P) para um problema não determinístico em espaço de estados e a **largura contingente** como condição de completude. O K-replanner (IJCAI 2011) troca a largura por uma condição estrutural ("problemas simples") e uma tradução **linear** K′(P). LW1 (AAAI 2014) recupera expressividade com uma tradução linear completa para **largura 1**.

### Cited Findings
- **Albore, Palacios & Geffner**, "A Translation-Based Approach to Contingent Planning", IJCAI 2009, pp. 1623–1628. Mapeia problemas contingentes em problemas não determinísticos em espaço de estados. Identifica a **"contingent width"**: para largura limitada, a tradução é "sound, polynomial, and complete". Resolve o problema traduzido via relaxação clássica (planejador CLG). — [IJCAI](https://www.ijcai.org/Abstract/09/271); [mlanthology](https://mlanthology.org/ijcai/2009/albore2009ijcai-translation); [PDF](https://www-i6.informatik.rwth-aachen.de/~hector.geffner/www.dtic.upf.edu/~hgeffner/alex-ijcai09.pdf)
- **Bonet & Geffner**, "Planning under Partial Observability by Classical Replanning: Theory and Experiments", IJCAI 2011, pp. 1936–1941 (Menlo Park: AAAI Press). É o **K-replanner**. Condição: problemas "simples", ou seja, cláusulas não unitárias de I invariantes e nenhum fluente oculto em condições de efeito. Nesse caso a tradução K′(P) é correta, completa e linear (detalhes no relatório anterior). — [repositório UPF](https://repositori.upf.edu/items/3cb5516d-a66f-40e7-8c82-af88db57376b/full); [IJCAI](https://www.ijcai.org/Abstract/11/324)
- **Bonet & Geffner**, "Flexible and Scalable Partially Observable Planning with Linear Translations", AAAI 2014, pp. 2235–2241. É o **LW1**: tradução linear completa para problemas de largura 1, combinando o escopo do CLG com a eficiência do K-replanner. — [OJS](https://ojs.aaai.org/index.php/AAAI/article/view/9047); [mlanthology](https://mlanthology.org/aaai/2014/bonet2014aaai-flexible); [PDF](https://www-i6.informatik.rwth-aachen.de/~hector.geffner/www.dtic.upf.edu/~hgeffner/blai-aaai-2014.pdf)
- **PO-PRP** (Muise, Belle & McIlraith, AAAI 2014) usa a compilação do K-replanner com o planejador FOND PRP para gerar políticas fortes cíclicas. Título exato não verificado nesta sessão; o relatório anterior cita OJS 9049. — [OJS](https://ojs.aaai.org/index.php/AAAI/article/view/9049)
- **Bonet & Geffner**, "Planning with Incomplete Information as Heuristic Search in Belief Space", AIPS 2000, pp. 52–61. É o antecessor de busca em espaço de crenças que as traduções procuram evitar. — [dblp (espelho)](https://dblp.uni-trier.de/rec/conf/aips/BonetG00.html); [PDF](https://bonetblai.github.io/reports/aips00-incomplete.pdf)

### Inferences
- O caminho CLG → K-replanner → LW1 troca completude geral por linearidade. A largura contingente (CLG) dá traduções polinomiais em i; o K-replanner é linear, mas exige "simplicidade"; LW1 é linear e completo para largura 1.

### Gaps
- Não obtive a definição formal da "contingent width" do CLG (só o resumo). Também não verifiquei a definição de largura usada no LW1, que provavelmente vem do JAIR 2014 de *belief tracking*.

## 4. Replanejamento por amostragem e regressão (linha Brafman–Shani)

### Takeaway
A linha de Ben-Gurion não usa etiquetas, mas usa determinização ou amostragem (SDR), compilação multi-caminho (MPSR) e regressão. A complexidade da regressão é limitada pela **largura de Bonet & Geffner**.

### Cited Findings
- **Shani & Brafman**, "Replanning in Domains with Partial Information and Sensing Actions", IJCAI 2011, pp. 2021–2026. — [referência em Azran et al.](https://www.alphaxiv.org/abs/2609.16884)
- **Brafman & Shani**, "Replanning in Domains with Partial Information and Sensing Actions", *JAIR* 45 (2012) 565–600, doi:10.1613/jair.3711. É o planejador **SDR** (Sample, Determinize, Replan). — [JAIR](https://jair.org/index.php/jair/article/view/10790); [CRIS](https://cris.iucc.ac.il/en/publications/replanning-in-domains-with-partial-information-and-sensing-action-2/)
- **Brafman & Shani**, "Online Belief Tracking using Regression for Contingent Planning", *Artificial Intelligence* 241 (2016) 131–152. A fórmula regredida envolve apenas variáveis relevantes à consulta, e a complexidade é limitada pelo parâmetro de largura do *factored belief tracking* (FBT) de Bonet & Geffner. Inclui representação por circuito polinomial que funciona com **ações não determinísticas**. Há uma versão anterior no ECAI 2014. — [CRIS](https://cris.iucc.ac.il/en/publications/online-belief-tracking-using-regression-for-contingent-planning/); [PDF ECAI 2014](https://tzin.bgu.ac.il/~shanigu/Publications/ECAI2014.Regression.pdf)
- **Komarnitsky & Shani**, "Computing Contingent Plans Using Online Replanning", AAAI 2016, pp. 3159–3165. Isso **corrige o enunciado da tarefa**: o venue é AAAI 2016, não ICAPS 2016. A versão em grafo com fusão de nós é Maliah, Komarnitski & Shani, "Computing Contingent Plan Graphs using Online Planning", ACM TAAS 2022 (CPOR). — [OJS](https://ojs.aaai.org/index.php/AAAI/article/view/10406); [mlanthology](https://mlanthology.org/aaai/2016/komarnitsky2016aaai-computing); [CRIS BGU (TAAS)](https://cris.bgu.ac.il/en/publications/computing-contingent-plan-graphs-using-online-planning-2/)
- HCP: Maliah, Brafman, Karpas & Shani, "Partially Observable Online Contingent Planning Using Landmark Heuristics", ICAPS 2014. — [PDF](https://tzin.bgu.ac.il/~shanigu/Publications/ICAPS2014.pdf)

### Gaps
- Não encontrei uma tradução no estilo K_{T,M} na linha SDR. A conexão é pelo parâmetro de largura, não por etiquetas.

## 5. Largura para rastreamento de crenças (belief tracking width, causal width)

### Takeaway
Bonet & Geffner deslocam a largura da tradução para o **rastreamento de crenças**. O FBT é exponencial na largura, que também cobre problemas não determinísticos; o *beam tracking* é correto, aproximado e exponencial apenas na **largura causal**, menor que a largura.

### Cited Findings
- **Bonet & Geffner**, "Causal Belief Decomposition for Planning with Sensing: Completeness Results and Practical Approximation", IJCAI 2013, pp. 2275–2281; arXiv 1909.13778 (2019). Testa *beam tracking* em Battleship, Minesweeper e Wumpus e relata desempenho de ponta em tempo real "in domains with high widths but low causal widths". — [IJCAI](https://ijcai.org/Abstract/13/336); [arXiv](https://arxiv.org/pdf/1909.13778)
- **Bonet & Geffner**, "Belief Tracking for Planning with Sensing: Width, Complexity and Approximations", *JAIR* 50 (2014) 923–970, doi:10.1613/jair.4475. Introduz uma noção de largura que **também cobre problemas não determinísticos**, o FBT exponencial nessa largura e o *beam tracking*, exponencial na largura causal. — [mlanthology](https://mlanthology.org/jair/2014/bonet2014jair-belief/); [bibbase](https://bibbase.org/network/publication/bonet-geffner-belieftrackingforplanningwithsensingwidthcomplexityandapproximations-2014)
- **Bonet & Geffner**, "Factored Probabilistic Belief Tracking", arXiv 1909.13779. Generaliza o *beam tracking* para ações e sensores ruidosos. O venue IJCAI 2016 não foi verificado. — [arXiv](https://arxiv.org/pdf/1909.13779)
- Livro: H. Geffner & B. Bonet, *A Concise Introduction to Models and Methods for Automated Planning*, Synthesis Lectures on AI and ML, Morgan & Claypool 2013, hoje na Springer (doi 10.1007/978-3-031-01564-9). Tem capítulos separados "Planning with Sensing: Logical Models" e "POMDP Planning". A série e o número exato não foram verificados nesta sessão. — [Springer](https://link.springer.com/10.1007/978-3-031-01564-9)

### Gaps
- Não obtive a definição formal de largura causal (os sites de Bonet e arXiv estão inacessíveis). Segundo a minha memória, ainda não verificada, a largura é o maior número de variáveis relevantes para uma precondição, meta ou observação, e a largura causal conta apenas os ancestrais causais imediatos.

## 6. Configurações epistêmicas e multiagente

### Takeaway
A linha K se estende a crenças aninhadas por duas rotas: Kominis & Geffner (ICAPS 2015/2017), que levam a compilação de crenças a múltiplos agentes, e Muise et al. (AAAI 2015), que compilam crenças aninhadas KD45n restritas (PDKB) para planejamento clássico.

### Cited Findings
- **Kominis & Geffner**, "Beliefs in Multiagent Planning: From One Agent to Many", ICAPS 2015, pp. 147–155. — [gravação ICAPS](https://www.icaps-conference.org/recording/beliefs-in-multiagent-planning-from-one-agent-to-many/)
- **Kominis & Geffner**, "Multiagent Online Planning with Nested Beliefs and Dialogue", ICAPS 2017, 27(1): 186–194. Estende a formulação ao caso online, em que os planos devem funcionar para o estado oculto verdadeiro revelado pelas observações. Isso **corrige o enunciado**: o *follow-up* é ICAPS 2017, não KR 2017. — [OJS](https://ojs.aaai.org/index.php/ICAPS/article/view/13826); [PDF](https://ml.rwth-aachen.de/~hector.geffner/www.dtic.upf.edu/~hgeffner/Filippos-ICAPS-2017.pdf)
- **Muise, Belle, Felli, McIlraith, Miller, Pearce & Sonenberg**, "Planning Over Multi-Agent Epistemic States: A Classical Planning Approach", AAAI 2015, pp. 3327–3334, doi:10.1609/aaai.v29i1.9665. Planeja da perspectiva de um agente com crenças aninhadas KD45n restritas, compiladas para planejamento clássico. A "Amended Version" saiu no workshop DMAP-2015 (arquivo "pdkbplanning"). — [OJS](https://ojs.aaai.org/index.php/AAAI/article/view/9665); [bibbase (versão revisada)](https://bibbase.org/network/publication/muise-belle-felli-mcilraith-miller-pearce-sonenberg-planningovermultiagentepistemicstatesaclassicalplanningapproachamendedversion-2015)
- Contexto recente: "The Epistemic Planning Domain Definition Language: Official Guideline" (arXiv 2601.20969, jan. 2026, Bolzano e Oxford). — [alphaXiv](https://www.alphaxiv.org/abs/2601.20969)
- Também: "A Dynamic Epistemic Framework for Conformant Planning" (arXiv 1606.07528), que reduz a verificação de planos e o planejamento conformante a *model checking* em DEL. — [alphaXiv](https://www.alphaxiv.org/abs/1606.07528)

### Gaps
- Não li o texto de Kominis & Geffner 2015. A afirmação de que eles compilam para planejamento clássico rastreando crenças com literais indexados por estado vem da minha memória e precisa ser confirmada. Também não verifiquei se usam explicitamente etiquetas, nem se Muise et al. empregam alguma noção de largura.

## 7. Trabalhos recentes (2017–2026) que usam etiquetas, largura ou K-traduções, incluindo aprendizado e LLMs

### Takeaway
O uso explícito mais forte de **etiquetas, largura e base** recente é o CPCES (Grastien & Scala, 2017–2020). Fadnis & Rintanen (PRICAI 2022) generalizam literais e etiquetas. Em 2025–2026 o vínculo com aprendizado ou LLMs é indireto: trabalhos com VLMs usam CPOR e citam Palacios & Geffner, mas não propõem novas traduções por largura. Não encontrei trabalho com LLM que use etiquetas ou largura diretamente.

### Cited Findings
- **Grastien & Scala**, "Intelligent belief state sampling for conformant planning", IJCAI 2017. Reduz o planejamento conformante sobre uma amostra a planejamento clássico e usa regressão para extrair contraexemplos. — [ANU](https://researchportalplus.anu.edu.au/en/publications/intelligent-belief-state-sampling-for-conformant-planning/)
- **Grastien & Scala**, "Sampling Strategies for Conformant Planning", ICAPS 2018, pp. 97–105. Analisa a fase de amostragem "using tags, width, and basis", limita o número máximo de iterações do CPCES e caracteriza estratégias que garantem encontrar subconjuntos de bases mínimas. — [OJS](https://ojs.aaai.org/index.php/ICAPS/article/view/13894)
- **Grastien & Scala**, "CPCES: A planning framework to solve conformant planning problems through a counterexample guided refinement", *Artificial Intelligence* 284 (2020) 103271, doi:10.1016/j.artint.2020.103271. Estuda correção, completude e otimalidade, relaciona o CPCES com etiquetas e largura e prova limites de pior caso. — [ANU](https://researchportalplus.anu.edu.au/en/publications/cpces-a-planning-framework-to-solve-conformant-planning-problems-/)
- **Fadnis & Rintanen**, PRICAI 2022: fórmulas de três valores generalizadas e redução automática de conformante para clássico (já no relatório anterior). — [PDF](https://users.aalto.fi/~rintanj1/papers/FadnisRintanen22.pdf)
- "On Conformant Planning and Model-Checking of ∃*∀* Hyperproperties" (arXiv 2512.23324, dez. 2025) relaciona planejamento conformante e verificação de hiperpropriedades. — [alphaXiv](https://www.alphaxiv.org/abs/2512.23324)
- **Azran, Navat & Keren** (Technion), "Bridging Learned Visual Perception and Symbolic Belief-Space Planning via Probabilistic Grounding", preprint aceito no NeuS 2026 (arXiv 2609.16884). Propõe o paradigma "VLM-as-probabilistic-grounder": o VLM produz uma distribuição sobre estados simbólicos, e planos robustos são obtidos com planejadores conformantes prontos, seguindo Taig & Brafman 2013 e chamando o **CPOR** via Unified Planning. Os autores relatam não terminação no backend CPOR do Unified Planning. Citam Palacios & Geffner 2009. — [alphaXiv](https://www.alphaxiv.org/abs/2609.16884)
- Outros recentes de crenças com VLM/LLM (TAMPAgent, arXiv 2611.03584; Active Epistemic Control, arXiv 2602.03974) apareceram na busca. Não verifiquei se usam traduções K. — [TAMPAgent](https://www.alphaxiv.org/abs/2611.03584); [AEC](https://www.alphaxiv.org/abs/2602.03974)

### Inferences
- A largura conformante sobreviveu como **ferramenta de análise**, por exemplo para limitar iterações de amostragem no CPCES, mais do que como base de novos compiladores. A integração com aprendizado e LLMs acontece no nível de *grounding*, e o planejamento continua delegado a planejadores da linhagem (CPOR, conformantes).

### Gaps
- Não encontrei trabalho de 2020–2026 que estenda K_i ou LW1 diretamente, nem planejador com LLM que use etiquetas ou largura. Isso vale com a ressalva de que as buscas foram limitadas, dblp estava inacessível e não verifiquei Google Scholar nem a lista de citações do JAIR 2009.

## 8. Complexidade (verificação de títulos e venues)

### Takeaway
Os resultados de complexidade conformante (EXPSPACE) e contingente (2-EXP) já estão documentados com fontes no relatório anterior. Esta seção apenas registra os dados bibliográficos.

### Cited Findings
- J. Rintanen, "Complexity of Planning with Partial Observability", ICAPS 2004. O resultado 2-EXP-completo e a Tabela 1 estão no relatório anterior. O título vem da minha memória; o PDF está no link. — [PDF](https://gki.informatik.uni-freiburg.de/papers/rintanen-icaps04.pdf)
- P. Haslum & P. Jonsson, "Some Results on the Complexity of Planning with Incomplete Information", ECP-99, publicado em LNAI 1809 (Springer, 2000). Prova que o conformante é EXPSPACE-completo. O título vem do nome do arquivo e da minha memória. — [PDF](https://www.ida.liu.se/divisions/aiics/publications/ECP-1999-Some-Results-Complexity.pdf)

### Gaps
- Páginas e títulos exatos de Rintanen 2004 e Haslum & Jonsson não foram confirmados por índice bibliográfico nesta sessão.
