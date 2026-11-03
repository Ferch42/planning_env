# Quatro linhas convergem no par (Π⁺, Π⁻)

A compilação de Palacios & Geffner (JAIR 2009) e a linha de Tran Cao Son tratam do mesmo objeto. Palacios & Geffner provam que **K0 equivale exatamente à 0-aproximação de Baral & Son** (Proposição 3). Também reconhecem que sua relação de relevância, com a cláusula 4 trocada por 4′, **é a relação de dependência de Son & Tu (KR 2006)**. A diferença entre as duas linhas está em como cada uma recupera a completude que a 0-aproximação perde. Son & Tu dividem o estado inicial em *contextos globais*: estados parciais formados por todas as combinações dos fluentes decisivos. Palacios & Geffner usam *contextos locais*, com uma tag por literal de precondição ou de meta. O custo da primeira abordagem é exponencial no número de fluentes decisivos; o da segunda, só na largura conformante ([Palacios & Geffner 2009](https://arxiv.org/abs/1401.3468); [Son & Tu 2006](https://www.cs.nmsu.edu/~tson/papers/KR0603SonT.pdf)).

As quatro linhas pedidas se organizam em torno dessa mesma representação de três valores (verdadeiro, falso, desconhecido):

- **Conhecimento (PKS):** o banco Kf do PKS é, na prática, o mesmo objeto escrito à mão, enriquecido com Kw/Kv/Kx.
- **Tradução:** as traduções (T0, CLG, K-replanner, LW1) automatizam esse nível de conhecimento e o levam ao planejamento contingente.
- **Suposições:** o planejamento baseado em suposições aparece como ações que "assumem" literais K. Exemplos: A0, que estende T0, e as ações de suposição do K-replanner.
- **Epistêmica:** a DEL monoagente é a semântica exata da qual essas representações são projeções corretas e incompletas.

A tese de Fernando Chiu (USP) fica no ponto mais baixo e mais barato desse mapa. Conforme a descrição recebida, seu estado (Π⁺, Π⁻) é o *a-state* ⟨T, F⟩ de Son & Baral. Esse estado é o K0 sem tags e o Kf do PKS. A tese herda a correção (*soundness*) e a progressão polinomial dessa representação, e também sua incompletude, que já tem contraexemplo canônico. **A tese em si não foi localizada nem lida nas notas**: tudo o que este relatório diz sobre ela é inferência a partir dessa descrição.

## K0 é a 0-aproximação, e a relevância de Palacios & Geffner é a dependência de Son & Tu

A 0-aproximação nasce em Baral & Son (ILPS 1997, pp. 387–401). Esse artigo não foi lido; a referência vem da bibliografia de Palacios & Geffner. A formalização completa está em Son & Baral, *Artificial Intelligence* 125(1–2):19–91, 2001 ([PDF](https://www.cs.nmsu.edu/~tson/papers/SonB01.pdf)). Nela, um **a-state é um par ⟨T, F⟩ de conjuntos disjuntos de fluentes conhecidos como verdadeiros e conhecidos como falsos**, e o espaço de estados tem 3ⁿ elementos, contra 2^(2ⁿ+n) c-states na semântica completa de A_K.

A transição sem sensoriamento é Res₀(a,⟨T,F⟩) = ⟨T ∪ e⁺ₐ \ F⁻ₐ, F ∪ e⁻ₐ \ F⁺ₐ⟩. Os efeitos com condição *conhecida* acrescentam literais. Os efeitos com condição apenas *possivelmente* verdadeira apagam o literal oposto, que volta a ser desconhecido. Os autores chamam isso de "raciocínio cético". A ação de sensoriamento gera um ramo por combinação de valores dos fluentes sensoreados ainda desconhecidos (Def. 7).

O artigo também define a 1-aproximação e a ω-aproximação. A primeira faz análise de casos a cada passo. A segunda faz análise de casos entre sensoriamentos. Vale a cadeia ⊨₀ ⊆ ⊨₁ ⊆ ⊨_ω ⊆ ⊨_{A_K} (Props. 4–6), e há contraexemplos de incompletude em todos os níveis ([Son & Baral 2001](https://www.cs.nmsu.edu/~tson/papers/SonB01.pdf)). Res₀ é polinomial e Res₁ é coNP-completo, resultado atribuído a Baral, Kreinovich & Trejo (AIJ 122, 2000), artigo não lido diretamente.

Palacios & Geffner reconstroem essa semântica como problema clássico. Em K0, os fluentes novos são KL e K¬L. Cada efeito a: C → L vira uma **regra de suporte** KC → KL e uma **regra de cancelamento** ¬K¬C → ¬K¬L. A Proposição 3 afirma que "π é plano clássico de K0(P) sse π é plano conformante para P segundo a 0-aproximação". O artigo nota que ambas "jogam fora a informação disjuntiva": com I = {p ∨ q}, nenhuma das duas aceita um plano conformante válido ([Palacios & Geffner 2009, §3](https://arxiv.org/abs/1401.3468)). As regras de suporte e de cancelamento são a tradução literal de Res₀: o suporte corresponde a e⁺ e o cancelamento corresponde a F⁻.

O segundo ponto de contato é mais fino e decide a questão de prioridade. A Definição 10 de Palacios & Geffner (relevância conformante) tem quatro cláusulas: L → L; L → L′ se a: C → L′ com L ∈ C; transitividade; e L → L′ se L → ¬L″ e L″ → ¬L′. Os próprios autores escrevem que, trocando a cláusula 4 por **4′ (L → L′ se ¬L → ¬L′)**, "equivalente a 4 no contexto de 1–3, a definição resultante é a de Son and Tu (2006)", usada ali "para gerar um conjunto limitado de estados iniciais 'parciais' sobre os quais a 0-aproximação é completa" ([Palacios & Geffner 2009, §6.1](https://arxiv.org/abs/1401.3468)). A Def. 2 de Son & Tu contém literalmente a cláusula "¬l₁ ⊲ ¬l₂" ([Son & Tu 2006](https://www.cs.nmsu.edu/~tson/papers/KR0603SonT.pdf)). A equivalência entre 4 e 4′ foi registrada como os autores a enunciam; as notas não a provaram de forma independente.

Son & Tu usam essa dependência para dar a **primeira condição suficiente de completude da 0-aproximação**. Um conjunto de estados é *aproximável* quando nenhum literal (ou ação) depende, em todos os seus elementos, de algum literal que distingue esses estados. O Teorema 1 diz que (D,{δ}) é completa se ext(δ) é aproximável. Um *conjunto decisivo* F reúne os fluentes desconhecidos dos quais algum literal depende tanto positiva quanto negativamente. Ele é calculado em tempo polinomial, sem garantia de minimalidade. Particionar δ por todas as interpretações de F produz Δ*, e sobre Δ* a 0-aproximação é completa (Teoremas 2–4). Daí sai o planejador **CPA⁺, "sound and complete"**; os CPA anteriores (AAAI-05, LPNMR-05) "são incompletos" ([Son & Tu 2006](https://www.cs.nmsu.edu/~tson/papers/KR0603SonT.pdf)).

Palacios & Geffner definem a largura w(L) como o tamanho da menor coleção de cláusulas cuja cobertura satisfaz as cláusulas iniciais relevantes a L. A largura w(P) é o máximo de w(L) sobre os literais de precondição e de meta. O Teorema 24 garante que **K_i é correto, polinomial para i fixo e completo se w(P) ≤ i**, e K_i com i = 0 coincide com K0 ([Palacios & Geffner 2009, §§6.4–6.5](https://arxiv.org/abs/1401.3468)).

Na §11, Palacios & Geffner fazem a comparação explícita com Son & Tu. Suponha n tautologias pᵢ ∨ ¬pᵢ, cada uma relevante a um único literal Lᵢ. Nesse caso, "o número de estados parciais necessários para a completude continua exponencial em n, enquanto o problema resultante tem largura 1". Os autores resumem: as tags codificam *contextos locais*, enquanto os estados parciais de Son & Tu codificam "suas combinações possíveis na forma de contextos globais" ([Palacios & Geffner 2009, §11](https://arxiv.org/abs/1401.3468)).

A questão de prioridade se resolve pelas datas, e isto é inferência das notas. Son & Tu saíram no KR 2006, pp. 481–491. Palacios & Geffner publicaram "Compiling uncertainty away" no AAAI-06, pp. 900–905, com K0 correto e incompleto, e "From conformant into classical planning" no ICAPS-07, pp. 264–271. Os dados das duas conferências vêm da lista de referências do JAIR 2009. A relação de dependência com a regra da negação e a ideia de garantir completude ramificando o estado inicial só nos fluentes relevantes aparecem primeiro em Son & Tu. A largura conformante, as tags locais, a família K_i e a base de estados críticos (Teoremas 30–31) são de Palacios & Geffner. As notas não verificaram se o AAAI-06 já trazia alguma noção de relevância, nem se as duas equipes chegaram a ela de forma independente.

O resto da linhagem de Son também se encaixa no mapa:

- **Tu, Son & Baral (TPLP 2007):** estendem a 0-aproximação a sensoriamento com **leis causais estáticas**. Um a-state passa a ser um conjunto consistente de literais *fechado* sob as leis (Cl_D). O conjunto pc(a,δ), que reúne o que "pode mudar", se propaga pelas leis estáticas. A transição é δ′ = Cl_D(e(a,δ) ∪ (δ \ ¬pc(a,δ))). O planejamento condicional na aproximação é **NP-completo**, contra Σ₂ᴾ na semântica completa ([Tu, Son & Baral, arXiv cs/0605017](https://arxiv.org/abs/cs/0605017)).
- **CpA(H) e CpA(C):** foram os rivais do T0 no IPC-6 (2008). São a versão de engenharia de Son & Tu: crenças em DNF de estados parciais, cada termo progredido pela 0-aproximação, e combinação de one-ofs independentes. Palacios & Geffner ligam essa técnica à sua noção de estados críticos ([Palacios & Geffner 2009, §§10–11](https://arxiv.org/abs/1401.3468); [PADL 2009](https://researchr.org/publication/TranNPS09)).
- **To, Son & Pontelli:** abandonam a aproximação pura e usam representações exatas de crença (DNF mínima, CNF mínima, implicados primos), em planejamento conformante e contingente ([AIJ 2015](https://dblp.org/rec/journals/ai/ToSP15); [ICAPS 2011](https://ojs.aaai.org/index.php/ICAPS/article/view/13477)).
- **EFP e mA\*:** depois de 2015, Son migra para o planejamento epistêmico multiagente com modelos de Kripke, que é representação exata e não aproximação ([EFP, ICAPS 2018](https://ojs.aaai.org/index.php/ICAPS/article/view/13881)).

A hierarquia abaixo junta as duas linhas. A ordenação entre os níveis 2–3 e os níveis 4–5 é inferência: as notas não encontraram nenhum artigo que prove inclusões como "ω-aproximação ⊆ K₁".

| Nível | Representação | Completude | Custo | Fonte |
|---|---|---|---|---|
| 1 | a-state ⟨T,F⟩ = K0 = Kf = (Π⁺,Π⁻) | só se w(P) = 0 ou ext(δ₀) aproximável | polinomial por passo; planejamento NP | Son & Baral 2001; P&G Prop. 3; Tu et al. 2007 |
| 2 | 1-aproximação | incompleta (contraexemplo D₄) | coNP por passo | Son & Baral 2001 |
| 3 | ω-aproximação | incompleta com sensoriamento (D₅) | exponencial entre sensoriamentos | Son & Baral 2001 |
| 4 | conjunto de a-states por conjunto decisivo | completa (conformante) | exponencial em nº de fluentes decisivos | Son & Tu 2006 |
| 5 | literais com tags KL/t + merges (K_i) | completa se w(P) ≤ i | exponencial só em i | P&G 2009, Teo. 24 |
| 6 | crença exata (DNF/CNF/PI, BDD, K_S0) | completa | exponencial no pior caso | To et al. 2015; P&G Teo. 9 |

## Mapa de relações entre as quatro linhas

O diagrama abaixo resume as relações. Uma seta "A → B" indica que B estende, compila ou aproxima A, conforme o rótulo.

```
                    Lógica epistêmica / cálculo de situações
        Moore 1985 ─► Scherl & Levesque 1993/2003 ─► Levesque 1996 (planos com sensoriamento)
                                   │                                │
                    (fragmento sintático)                  (condição de ramificação)
                                   ▼                                ▼
  NÍVEL DO CONHECIMENTO   Bacchus & Petrick KR'98 ─► PKS (AIPS-02, ICAPS-04): Kf, Kw, Kv, Kx
                                   │  Kf ≈ a-state (inferência)        │ "automatiza o nível do conhecimento"
                                   ▼                                    ▼
  APROXIMAÇÕES (Son)      Baral & Son 1997 ─► Son & Baral 2001 (0/1/ω-aprox.)
                                   │                 │ ≡ (Prop. 3)
                                   │                 ▼
  TRADUÇÕES               Son & Tu 2006 ══ relevância ══ P&G AAAI-06 (K0) ─► ICAPS-07 ─► JAIR 2009 (K_T,M, K_i, largura)
  (conformante)              │ (contexto global)              (contexto local)            │
                             ▼                                                             ├─► K_Si/T1 (2011), cp2fsc (2009)
                       CpA(H)/(C) IPC-6 ─► To/Son/Pontelli (crença exata)                 ├─► CPCES (amostragem; tags/largura como análise)
  Tu/Son/Baral 2007 (leis estáticas, sensoriamento)                                        │
                                                                                           ▼
  TRADUÇÕES (contingente)  CLG (IJCAI-09, largura contingente) ─► K-replanner (IJCAI-11, linear, "simples")
                                   │                                   │                    ─► LW1 (AAAI-14, linear, w=1)
                                   │  K_S0 inspira ─► SDR (IJCAI-11/JAIR-12) ─► CPOR (AAAI-16; TAAS-22)
                                   │                                   └─► PO-PRP (AAAI-14, FOND)
                                   └─► largura de rastreamento de crença (IJCAI-13, JAIR-14) ─► regressão (AIJ 2016)
  SUPOSIÇÕES               Nourbakhsh & Genesereth 1996 ; Albore & Bertoli 2004/06 (planos seguros)
                           T0 ─► A0 (Davis-Mendelow, Baier & McIlraith AAAI-13)
                           CLG/K-replanner: suposições sobre observações ; SDR: amostra de estado
  EPISTÊMICO (DEL)         Bolander & Andersen 2011 ; Löwe, Pacuit & Witzel 2011 ─► Andersen, Bolander & Jensen 2012 (DEL monoagente = contingente)
                           P&G/CLG ─► Kominis & Geffner ICAPS-15/17 (compilação multiagente)
                           PKS/literais ─► Muise et al. AAAI-15 (PDKB) ; Cooper et al. AIJ-21 (lógica leve)
                           Son & Baral 2001 ─(mudança de linha)─► mA* (AIJ 2022) ─► EFP (ICAPS-18) ─► EFP 2.0 ─► EPDDL / IPC 2026
```

A tabela registra cada aresta, o tipo de relação e a fonte que a sustenta.

| De | Relação | Para | Base |
|---|---|---|---|
| 0-aproximação (Son & Baral) | equivalente (compilação exata) | K0 (P&G) | Prop. 3 ([P&G 2009](https://arxiv.org/abs/1401.3468)) |
| Dependência (Son & Tu, Def. 2) | mesma relação | Relevância (P&G, Def. 10 com 4′) | declarado por P&G §6.1 |
| Estados parciais (Son & Tu) | contextos globais vs. locais; K_i domina em tamanho | Tags/merges (K_i) | P&G §11 |
| 0-aproximação | estende com leis estáticas e sensoriamento | Tu, Son & Baral 2007 | [arXiv cs/0605017](https://arxiv.org/abs/cs/0605017) |
| PKS | "definido sobre a 0-aproximação", segundo P&G; automatizado por K0 | K0 / T0 | P&G §11 |
| K_T,M / K_i | estende ao contingente (largura contingente) | CLG | [IJCAI 2009](https://www.ijcai.org/Abstract/09/271) |
| CLG | troca largura por "problemas simples"; tradução linear | K-replanner | [IJCAI 2011](https://www.ijcai.org/Abstract/11/324) |
| CLG + K-replanner | combina escopo e linearidade; completo para largura 1 | LW1 | [AAAI 2014](https://ojs.aaai.org/index.php/AAAI/article/view/9047) |
| K_S0 | inspira (proposições p/s) | SDR | [Brafman & Shani 2012](https://arxiv.org/abs/1401.6048) |
| Largura conformante | desloca-se para o rastreamento de crença | largura e largura causal (JAIR 2014) | [mlanthology](https://mlanthology.org/jair/2014/bonet2014jair-belief/) |
| T0 | estende com ações de suposição com custo | A0 | [AAAI 2013](https://ojs.aaai.org/index.php/AAAI/article/view/8687) |
| Traduções monoagente | eleva a crenças aninhadas | Kominis & Geffner | [ICAPS 2017](https://ojs.aaai.org/index.php/ICAPS/article/view/13826) |
| DEL S5 monoagente | semântica exata da qual crenças são projeções | contingente / K0 | [Bolander 2017](https://arxiv.org/abs/1703.02192); [JELIA 2012](https://link.springer.com/doi/10.1007/978-3-642-33353-8_8) |

## Cada linha tem uma ideia central e um custo próprio

**Planejamento baseado em suposições.** Nenhum artigo cunhou sozinho o termo, e ele designa pelo menos três coisas tecnicamente distintas. Nourbakhsh & Genesereth (*Autonomous Robots* 3(1):49–67, 1996) introduzem o "planejamento assumptivo": simplificar o mundo, executar e replanejar. A garantia de correção e completude depende de nunca dar passos que tornem o problema insolúvel ([CMU RI](https://ri.cmu.edu/publications/assumptive-planning-and-execution-a-simple-working-robot-architecture)). Albore & Bertoli (AAAI 2004, pp. 495–500) formalizam os planos **seguros**: atingem a meta quando a suposição vale, e um monitor detecta a tempo qualquer violação que ameace o sucesso ([ML Anthology](https://mlanthology.org/aaai/2004/albore2004aaai-generating)). Em 2006 estendem as suposições para LTL (ICAPS 2006, pp. 193–202) ([FBK](https://cris.fbk.eu/handle/11582/11488)). Davis-Mendelow, Baier & McIlraith (AAAI-13, pp. 209–216) fazem a escolha *offline* e *com custo* de suposições sobre o estado inicial, **estendendo a tradução T0** no planejador A0, e propõem aplicações em explicação e diagnóstico ([AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/8687)). Pellier & Fiorino (relatório de 2004, arXiv 1810.08431) usam suposições sobre precondições em HTN multiagente ([arXiv](https://arxiv.org/abs/1810.08431)). Do lado contingente, o padrão é otimismo seguido de replanejamento. CLG assume valores favoráveis para observações futuras. O K-replanner otimiza "o prefixo mais curto que atinge a meta ou produz uma observação que refuta uma suposição". SDR substitui o otimismo por um estado inicial amostrado ([Brafman & Shani 2012](https://arxiv.org/abs/1401.6048); [pôster IJCAI-11](https://bonetblai.github.io/reports/IJCAI11-replanning-poster.pdf)). A noção de "ação segura" de SDR é mais estreita que a de Albore & Bertoli: exige precondições conhecidas na crença verdadeira, e não a detectabilidade de qualquer suposição violada. As notas não encontraram nenhum artigo que combine escolha preferencial de suposições, à maneira de Davis-Mendelow, com garantia de monitorabilidade, à maneira de Albore & Bertoli.

**Planejamento no nível do conhecimento (PKS).** O PKS representa o *estado de conhecimento* do agente em quatro bancos:

- **Kf:** literais e valores de funções conhecidos, sem hipótese de mundo fechado;
- **Kw:** fórmulas cujo valor o agente saberá ("sabe se"), produzidas por sensoriamento binário;
- **Kv:** termos cujo valor o agente saberá em tempo de execução;
- **Kx:** disjunções exclusivas.

As ações atualizam esses bancos como no STRIPS ([Petrick & Bacchus, AIPS-02](https://cdn.aaai.org/AIPS/2002/AIPS02-022.pdf)). O algoritmo de inferência é correto e deliberadamente incompleto. O PKS **não pode representar K(P ∨ Q)** com disjunção não exclusiva, e a regra de consistência 3 apaga entradas de Kx depois de uma mudança ôntica. Por isso disjunções podem ser carregadas de uma crença para a seguinte, mas nunca criadas. As regras de atualização U são "invariantes de estado no nível do conhecimento", e a ramificação só ocorre sobre Kw, o que implementa uma condição de Levesque (1996). O ICAPS-04 acrescenta pós-dição, metas temporalmente estendidas e ramificação sobre Kv de funções de domínio finito ([Petrick & Bacchus, ICAPS-04](http://www.cs.toronto.edu/~fbacchus/Papers/PetrickBacchusICAPS2004.pdf)). Os fundamentos vêm de Moore, de Scherl & Levesque (AIJ 144, 2003) e de Levesque (AAAI-96). Palacios & Geffner apresentam K0 como solução a duas limitações do PKS: os domínios do PKS são escritos à mão e a busca é cega ([P&G 2009, §11](https://arxiv.org/abs/1401.3468)). Depois de 2013, o PKS sobrevive sobretudo em robótica e interação humano-robô, no grupo de Petrick ([ICAPS 2013](https://ojs.aaai.org/index.php/ICAPS/article/view/13589); [TraCE 2022](https://arxiv.org/abs/2207.09709)).

**Planejamento conformante e contingente por tradução.** O núcleo é o que se descreveu na seção anterior. O que muda entre os sucessores é a condição de completude:

| Planejador | Condição de completude | Tamanho da tradução | Observação | Fonte |
|---|---|---|---|---|
| CLG | largura contingente limitada | polinomial | — | [IJCAI 2009](https://www.ijcai.org/Abstract/09/271) |
| K-replanner | problemas "simples": cláusulas não unitárias invariantes, nenhum fluente oculto em condição de efeito | linear | — | [IJCAI 2011](https://www.ijcai.org/Abstract/11/324) |
| LW1 | largura 1 | linear | — | [AAAI 2014](https://ojs.aaai.org/index.php/AAAI/article/view/9047) |
| SDR | conectividade do domínio | — | não usa tags: amostra estados, determiniza e verifica a segurança por regressão sobre a crença verdadeira | [JAIR 2012](https://jair.org/index.php/jair/article/view/10790) |
| CPOR | — | — | constrói grafos contingentes chamando replanejamento online | [AAAI 2016](https://ojs.aaai.org/index.php/AAAI/article/view/10406); [TAAS 2022](https://cris.bgu.ac.il/en/publications/computing-contingent-plan-graphs-using-online-planning-2/) |

Bonet & Geffner deslocam a largura da tradução para o **rastreamento de crença**. O rastreamento fatorado é exponencial numa largura que também cobre ações não determinísticas, e o *beam tracking* é exponencial só na largura causal ([JAIR 50, 2014](https://mlanthology.org/jair/2014/bonet2014jair-belief/); [IJCAI 2013](https://ijcai.org/Abstract/13/336)). Brafman & Shani limitam a regressão por esse mesmo parâmetro ([AIJ 241, 2016](https://cris.iucc.ac.il/en/publications/online-belief-tracking-using-regression-for-contingent-planning/)). Depois de 2017, a largura sobrevive mais como **ferramenta de análise** do que como base de novos compiladores. No CPCES, por exemplo, tags, largura e base limitam as iterações de amostragem ([ICAPS 2018](https://ojs.aaai.org/index.php/ICAPS/article/view/13894); [AIJ 2020](https://researchportalplus.anu.edu.au/en/publications/cpces-a-planning-framework-to-solve-conformant-planning-problems-/)). As notas não acharam extensões diretas de K_i ou LW1 entre 2020 e 2026.

**Planejamento epistêmico (DEL).** Bolander & Andersen (JANCL 21(1):9–34, 2011) e Löwe, Pacuit & Witzel (LORI 2011) introduzem o planejamento DEL. Os estados são modelos epistêmicos, as ações são modelos de evento e a transição é o produto. O caso monoagente é decidível. O multiagente é indecidível já com dois agentes ([Aucher & Bolander, IJCAI 2013](https://ijcai.org/Abstract/13/015)) e já com precondições de profundidade modal 2 ([Charrier et al., IJCAI 2016](https://www.ijcai.org/Abstract/16/150)). Andersen, Bolander & Jensen (JELIA 2012) definem planos condicionais em DEL monoagente com observabilidade parcial, e isso é planejamento contingente reescrito em lógica ([Springer](https://link.springer.com/doi/10.1007/978-3-642-33353-8_8)). Bolander separa as abordagens **sintáticas** (PKS, Muise et al., Kominis & Geffner), que "modelam conhecimento", das **semânticas** (DEL), que "modelam ignorância", e as trata como duais ([Bolander 2017](https://arxiv.org/abs/1703.02192)). Os planejadores práticos se dividem em dois grupos:

- **Compilação para clássico com conhecimento literal restrito:** RP-MEP/PDKB de Muise et al. (AAAI 2015), Kominis & Geffner (ICAPS 2015/2017), Cooper et al. (AIJ 298, 2021) e as perspectivas de Hu, Miller & Lipovetzky (JAIR 75, 2022).
- **Busca nativa em Kripke ou *possibilities*:** EFP/PG-EFP (ICAPS 2018), EFP 2.0 (ICAPS 2020), MEPK (IJCAI 2017), DELPHIC (JELIA 2023) e DAEDALUS (KR 2025).

O primeiro grupo ganha escalabilidade pelo mesmo caminho que K0 e PKS: **renuncia ao conhecimento disjuntivo geral**. O EPDDL, de Burigana & Fabiano (arXiv 2601.20969, 2026), é a linguagem oficial da primeira trilha epistêmica do IPC 2026. Ele padroniza a semântica DEL completa e nomeia fragmentos por *requirements*; o fragmento base `:pal` contém só anúncios públicos ([guia oficial](https://www.alphaxiv.org/abs/2601.20969)).

## A tese de Chiu ocupa o nível w = 0 do mapa

A descrição recebida diz que a tese usa uma teoria (Π⁺, Π⁻) de literais sabidos verdadeiros e sabidos falsos, atualizada por regras de observação e com invariantes. **Nenhuma das notas localizou ou leu a tese.** O relatório anterior desta conversa chamou o mesmo objeto de (Γ⁺, Γ⁻). Tudo o que segue é inferência a partir das definições publicadas.

O par (Π⁺, Π⁻) é, inclusive na notação, o a-state ⟨T, F⟩ de Son & Baral. É também o conjunto {KL} ∪ {K¬L} de um estado K0 e a parte de literais do banco Kf do PKS. A exigência Π⁺ ∩ Π⁻ = ∅ corresponde à regra de consistência 1 do PKS. Daí decorrem três consequências, cada uma ancorada numa fonte primária.

A primeira é sobre **correção**. A tese só é correta no sentido de Son & Baral se um literal conhecido volta a ser desconhecido sempre que um efeito contrário *possivelmente* dispara (F⁻ₐ em Res₀, ou a regra de cancelamento ¬K¬C → ¬K¬L em K0). Remover o literal só quando a condição é *sabidamente* verdadeira quebra a correção ([Son & Baral 2001, §3.1](https://www.cs.nmsu.edu/~tson/papers/SonB01.pdf); [P&G 2009](https://arxiv.org/abs/1401.3468)).

A segunda é sobre **invariantes**. O tratamento correto na linhagem de Son é o par fechamento Cl_D mais propagação de pc pelas leis estáticas, de Tu, Son & Baral 2007. Fechar Π⁺/Π⁻ sob invariantes sem retirar os literais em ¬pc é incorreto. O descarte de ramos de observação inconsistentes com os invariantes (o exemplo da janela, com oneof{open, locked, closed}) corresponde à poda de observações impossíveis ([arXiv cs/0605017](https://arxiv.org/abs/cs/0605017)). No PKS, o análogo são as regras U, "invariantes no nível do conhecimento". No K-replanner, são as cláusulas invariantes que tornam um problema "simples".

A terceira é sobre **completude**. Ela só vale em classes restritas, e as caracterizações publicadas são estas:

- largura w(P) = 0 (Teorema 24 de P&G);
- ext(δ₀) aproximável (Teorema 1 de Son & Tu);
- problemas "simples" do K-replanner, desde que a tese mantenha também as cláusulas invariantes.

Que as duas primeiras coincidam é inferência das notas, não teorema publicado. O contraexemplo canônico para qualquer representação (Π⁺, Π⁻) com observações é o D₅ de Son & Baral (Ex. 11): {initially p; a causes ¬p if r; b determines r; c causes p if r}. A semântica completa conclui `Knows p after [a;b;c]`, mas a representação de três valores perde a correlação p ↔ ¬r depois de a, e a observação posterior de r não a recupera ([Son & Baral 2001](https://www.cs.nmsu.edu/~tson/papers/SonB01.pdf)). É o contraexemplo a citar na tese.

Quanto às regras de observação, se elas apenas acrescentam o literal observado a Π⁺ ou Π⁻, implementam a transição Φ₀ de sensoriamento, com a ramificação resolvida online pelo ambiente em vez de enumerada no plano. Em relação às quatro linhas, a posição da tese é esta:

| Linha | Correspondência na tese (inferência) | O que a tese não tem, se não acrescentar |
|---|---|---|
| Aproximações de Son | (Π⁺,Π⁻) = a-state; observação = Φ₀; invariantes ≈ leis estáticas (Tu et al. 2007) | conjunto decisivo / partição do estado inicial |
| Tradução | K0 = K_i com i = 0; replanejamento otimista ≈ K-replanner | tags KL/t e merges; cláusulas invariantes |
| Nível do conhecimento | Kf do PKS; regras de observação ≈ regras U / regras de suporte | Kw (raciocínio em tempo de plano sobre observações futuras), Kv, Kx, pós-dição |
| Suposições | literais desconhecidos tratados como valores assumidos; entrada em Π⁺/Π⁻ por observação = evento de monitoramento que refuta suposições | garantia de segurança à Albore & Bertoli: toda suposição observável antes do passo que depende dela |
| Epistêmico | projeção de um modelo S5 monoagente: p ∈ Π⁺ sse M ⊨ Kp | conhecimento disjuntivo, crenças aninhadas, multiagente |

Há dois caminhos publicados para dar completude à tese, e o argumento de Palacios & Geffner indica qual escala melhor. O caminho de Son & Tu mantém um *conjunto* de pares (Π⁺, Π⁻), um por interpretação dos fluentes decisivos, e é exponencial nesse número. O caminho de Palacios & Geffner usa pares condicionados por tags, KL/t, com merges, e é exponencial só na largura. Como "quase todos os benchmarks têm largura 1", a extensão mais barata é admitir entradas condicionais L/t ou, à moda do K-replanner, cláusulas invariantes. O argumento de terminação do replanejamento também tem modelo pronto. No K-replanner, cada refutação acrescenta pelo menos um literal conhecido. Em SDR, cada falha elimina pelo menos um estado inicial. Esses argumentos dependem de variáveis ocultas estáticas e de domínio conexo. As notas não verificaram se a tese assume essas hipóteses. Em relação ao EPDDL, a tese seria um fragmento monoagente (`:ontic-actions` mais `:partial-observability`) resolvido sob aproximação literal. O guia mostra informalmente que tarefas conformantes se codificam em EPDDL com o mesmo conjunto de soluções ([Burigana & Fabiano 2026](https://www.alphaxiv.org/abs/2601.20969)). Logo, uma codificação EPDDL dos domínios da tese teria os mesmos planos sempre que K0 fosse completo, e possivelmente mais planos nos demais casos.

## Artigos principais por linha, com dados bibliográficos

Na coluna de status, "verificado" significa que as notas confirmaram os dados numa fonte primária ou num índice bibliográfico. "Via referência" significa que os dados vêm da lista de referências de outro artigo. "Não verificado" significa que vêm de memória ou de fonte secundária.

### Aproximações de Son e coautores

| Artigo | Dados bibliográficos | Status |
|---|---|---|
| Baral & Son, "Approximate reasoning about actions in presence of sensing and incomplete information" | Proc. ILPS 1997, pp. 387–401 | via referência (P&G 2009); não lido |
| Son & Baral, "Formalizing sensing actions — A transition function based approach" | *Artificial Intelligence* 125(1–2):19–91, 2001; DOI 10.1016/S0004-3702(00)00080-1 | verificado ([PDF](https://www.cs.nmsu.edu/~tson/papers/SonB01.pdf)) |
| Baral, Kreinovich & Trejo (complexidade do planejamento com informação incompleta) | *Artificial Intelligence* 122:241–267, 2000 | via referência; não lido |
| Son, Tu, Gelfond & Morales, "Conformant planning for domains with constraints: A new approach" | Proc. AAAI-05, pp. 1211–1216 | via referência (P&G 2009) |
| Son, Tu, Gelfond & Morales, "An approximation of action theories of AL and its application to conformant planning" | LPNMR 2005, LNCS; DOI 10.1007/11546207_14 | verificado ([PDF](https://www.cs.nmsu.edu/~tson/papers/SonTGM05b.pdf)) |
| Son & Tu, "On the completeness of approximation based reasoning and planning in action theories with incomplete information" | Proc. KR 2006, pp. 481–491, AAAI Press | verificado ([PDF](https://www.cs.nmsu.edu/~tson/papers/KR0603SonT.pdf)) |
| Tu, Son & Baral, "Reasoning and planning with sensing actions, incomplete information, and static causal laws using answer set programming" | *TPLP* 7(4):377–450, 2007; arXiv cs/0605017 | título e periódico verificados; **volume e páginas não verificados** |
| Son, Pontelli & To, "Some results on the completeness of approximation based reasoning" | AAAI Spring Symposium 2008, SS-08-02 | verificado ([PDF](https://www.aaai.org/Papers/Symposia/Spring/2008/SS-08-02/SS08-02-011.pdf)) |
| Tran, Nguyen, Pontelli & Son, "Improving performance of conformant planners: Static analysis of declarative planning domain specifications" | PADL 2009, LNCS 5418, pp. 239–253 | verificado ([researchr](https://researchr.org/publication/TranNPS09)) |
| Tran, Nguyen, Pontelli & Son, "CPA(C)/(H): Two approximation-based conformant planners" | descrição de planejador, IPC-6 (2008) | texto não lido (host bloqueado) |
| Tu, Son, Gelfond & Morales, "Approximation of action theories and its application to conformant planning" | *Artificial Intelligence* 175(1):79–119, 2011; DOI 10.1016/j.artint.2010.04.007 | verificado por fonte secundária; texto não lido |
| Nguyen, Tran, Son & Pontelli, "On improving conformant planners by analyzing domain-structures" | AAAI 2011, pp. 998–1003; DOI 10.1609/aaai.v25i1.8005 | verificado ([OJS](https://ojs.aaai.org/index.php/AAAI/article/view/8005)) |
| To, Son & Pontelli, "Contingent planning as AND/OR forward search with disjunctive representation" | ICAPS 2011, pp. 258–265 | verificado ([OJS](https://ojs.aaai.org/index.php/ICAPS/article/view/13477)) |
| To, Son & Pontelli, "A generic approach to planning in the presence of incomplete information: Theory and implementation" | *Artificial Intelligence* 227:1–51, 2015; DOI 10.1016/j.artint.2015.06.001 | verificado ([DBLP](https://dblp.org/rec/journals/ai/ToSP15)) |

### Traduções conformantes e contingentes

| Artigo | Dados bibliográficos | Status |
|---|---|---|
| Palacios & Geffner, "Compiling uncertainty away: Solving conformant planning problems using a classical planner (sometimes)" | Proc. AAAI-06, pp. 900–905 | páginas via referência (P&G 2009) |
| Palacios & Geffner, "From conformant into classical planning: Efficient translations that may be complete too" | Proc. ICAPS 2007, pp. 264–271 | páginas via referência ([PDF](https://cdn.aaai.org/ICAPS/2007/ICAPS07-034.pdf)) |
| Palacios & Geffner, "Compiling uncertainty away in conformant planning problems with bounded width" | *JAIR* 35:623–675, 2009; DOI 10.1613/jair.2708; arXiv 1401.3468 | verificado ([JAIR](https://jair.org/index.php/jair/article/view/10618)) |
| Albore, Palacios & Geffner, "A translation-based approach to contingent planning" (CLG) | IJCAI 2009, pp. 1623–1628 | verificado ([IJCAI](https://www.ijcai.org/Abstract/09/271)) |
| Bonet, Palacios & Geffner, "Automatic derivation of memoryless policies and finite-state controllers using classical planners" | ICAPS 2009, pp. 34–41; DOI 10.1609/icaps.v19i1.13379 | verificado ([OJS](https://ojs.aaai.org/index.php/ICAPS/article/view/13379)) |
| Albore, Ramírez & Geffner, "Effective heuristics and belief tracking for planning with incomplete information" (K_Si/T1) | ICAPS 2011, pp. 2–9; DOI 10.1609/icaps.v21i1.13473 | verificado ([OJS](https://ojs.aaai.org/index.php/ICAPS/article/view/13473)) |
| Bonet & Geffner, "Planning under partial observability by classical replanning: Theory and experiments" (K-replanner) | IJCAI 2011, pp. 1936–1941; DOI 10.5591/978-1-57735-516-8/IJCAI11-324 | verificado ([IJCAI](https://ijcai.org/Abstract/11/324)) |
| Shani & Brafman, "Replanning in domains with partial information and sensing actions" (SDR) | IJCAI 2011, pp. 2021–2026 | verificado ([ML Anthology](https://mlanthology.org/ijcai/2011/shani2011ijcai-replanning)) |
| Brafman & Shani, mesmo título (SDR, versão de periódico) | *JAIR* 45:565–600, 2012; DOI 10.1613/jair.3711 | verificado ([JAIR](https://jair.org/index.php/jair/article/view/10790)) |
| Bonet & Geffner, "Causal belief decomposition for planning with sensing: Completeness results and practical approximation" | IJCAI 2013, pp. 2275–2281 | verificado ([IJCAI](https://ijcai.org/Abstract/13/336)) |
| Bonet & Geffner, "Belief tracking for planning with sensing: Width, complexity and approximations" | *JAIR* 50:923–970, 2014; DOI 10.1613/jair.4475 | verificado ([mlanthology](https://mlanthology.org/jair/2014/bonet2014jair-belief/)) |
| Bonet & Geffner, "Flexible and scalable partially observable planning with linear translations" (LW1) | AAAI 2014, pp. 2235–2241; DOI 10.1609/aaai.v28i1.9047 | verificado ([OJS](https://ojs.aaai.org/index.php/AAAI/article/view/9047)) |
| Muise, Belle & McIlraith (PO-PRP) | AAAI 2014 | **título não verificado** ([OJS](https://ojs.aaai.org/index.php/AAAI/article/view/9049)) |
| Maliah, Brafman, Karpas & Shani, "Partially observable online contingent planning using landmark heuristics" (HCP) | ICAPS 2014, pp. 163–171; DOI 10.1609/icaps.v24i1.13632 | verificado ([OJS](https://ojs.aaai.org/index.php/ICAPS/article/view/13632)) |
| Komarnitsky & Shani, "Computing contingent plans using online replanning" | **AAAI 2016**, pp. 3159–3165 | verificado ([OJS](https://ojs.aaai.org/index.php/AAAI/article/view/10406)) |
| Maliah, Komarnitski & Shani, "Computing contingent plan graphs using online planning" (CPOR) | *ACM TAAS*, 2022 | verificado ([CRIS BGU](https://cris.bgu.ac.il/en/publications/computing-contingent-plan-graphs-using-online-planning-2/)) |
| Brafman & Shani, "Online belief tracking using regression for contingent planning" | *Artificial Intelligence* 241:131–152, 2016 | verificado ([CRIS](https://cris.iucc.ac.il/en/publications/online-belief-tracking-using-regression-for-contingent-planning/)) |
| Taig & Brafman, "Compiling conformant probabilistic planning problems into classical planning" | ICAPS 2013, pp. 197–205; DOI 10.1609/icaps.v23i1.13540 | via referência |
| Grastien & Scala, "CPCES: A planning framework to solve conformant planning problems through a counterexample guided refinement" | *Artificial Intelligence* 284:103271, 2020 | verificado ([ANU](https://researchportalplus.anu.edu.au/en/publications/cpces-a-planning-framework-to-solve-conformant-planning-problems-/)) |

### Planejamento baseado em suposições

| Artigo | Dados bibliográficos | Status |
|---|---|---|
| Nourbakhsh & Genesereth, "Assumptive planning and execution: a simple, working robot architecture" | *Autonomous Robots* 3(1):49–67, 1996; DOI 10.1007/BF00162467 | verificado ([CMU RI](https://ri.cmu.edu/publications/assumptive-planning-and-execution-a-simple-working-robot-architecture)) |
| Koenig & Smirnov, "Sensor-based planning with the freespace assumption" | ICRA 1997 | verificado ([CMU RI](https://www.ri.cmu.edu/publications/sensor-based-planning-with-the-freespace-assumption)) |
| Albore & Bertoli, "Generating safe assumption-based plans for partially observable, nondeterministic domains" | AAAI 2004, pp. 495–500 | verificado ([ML Anthology](https://mlanthology.org/aaai/2004/albore2004aaai-generating)) |
| Albore & Bertoli, "Safe LTL assumption-based planning" | ICAPS 2006, pp. 193–202 | verificado ([FBK](https://cris.fbk.eu/handle/11582/11488)) |
| Pellier & Fiorino, "Assumption-based planning" | relatório técnico, Leibniz/IMAG, 2004; arXiv 1810.08431 | verificado ([arXiv](https://arxiv.org/abs/1810.08431)) |
| Yoon, Fern & Givan, "FF-Replan: A baseline for probabilistic planning" | ICAPS 2007, pp. 352–359 | página final não verificada ([DBLP](https://dblp.org/rec/conf/aips/YoonFG07)) |
| Davis-Mendelow, Baier & McIlraith, "Making reasonable assumptions to plan with incomplete information: Abridged report" | AAAI-12 Workshop, AAAI Tech. Report WS-12-12, pp. 16–22 | nome do workshop diverge entre listagens ([aitopics](https://aitopics.org/doc/conferences:D1B6392E)) |
| Davis-Mendelow, Baier & McIlraith, "Assumption-based planning: Generating plans and explanations under incomplete knowledge" (A0) | AAAI-13, pp. 209–216; DOI 10.1609/aaai.v27i1.8687 | verificado ([OJS](https://ojs.aaai.org/index.php/AAAI/article/view/8687)) |

### Planejamento no nível do conhecimento

| Artigo | Dados bibliográficos | Status |
|---|---|---|
| Levesque, "What is planning in the presence of sensing?" | AAAI-96, pp. 1139–1146 | via referência (AIPS-02) |
| Bacchus & Petrick, "Modeling an agent's incomplete knowledge during planning and during execution" | KR'98, pp. 432–443, Morgan Kaufmann | via referência; texto não lido |
| Petrick & Bacchus, "A knowledge-based approach to planning with incomplete information and sensing" | AIPS 2002, pp. 212–221, AAAI Press | verificado ([PDF](https://cdn.aaai.org/AIPS/2002/AIPS02-022.pdf)) |
| Petrick & Levesque, "Knowledge equivalence in combined action theories" | KR 2002, pp. 303–314 | via referência |
| Scherl & Levesque, "Knowledge, action, and the frame problem" | *Artificial Intelligence* 144(1–2):1–39, 2003; DOI 10.1016/S0004-3702(02)00365-X | verificado ([DBLP](https://dblp.org/rec/journals/ai/ScherlL03)) |
| Petrick & Bacchus, "Extending the knowledge-based approach to planning with incomplete information and sensing" | ICAPS 2004, pp. 2–11, AAAI Press | verificado ([PDF](http://www.cs.toronto.edu/~fbacchus/Papers/PetrickBacchusICAPS2004.pdf)) |
| Petrick & Foster, "Planning for social interaction in a robot bartender domain" | ICAPS 2013, pp. 389–397; DOI 10.1609/icaps.v23i1.13589 | verificado ([OJS](https://ojs.aaai.org/index.php/ICAPS/article/view/13589)) |

### Planejamento epistêmico

| Artigo | Dados bibliográficos | Status |
|---|---|---|
| Bolander & Andersen, "Epistemic planning for single- and multi-agent systems" | *JANCL* 21(1):9–34, 2011; DOI 10.3166/jancl.21.9-34 | verificado ([T&F](https://www.tandfonline.com/doi/abs/10.3166/jancl.21.9-34)) |
| Löwe, Pacuit & Witzel, "DEL planning and some tractable cases" | LORI 2011, LNCS 6953, pp. 179–192 | verificado ([Springer](https://link.springer.com/doi/10.1007/978-3-642-24130-7_13)) |
| Andersen, Bolander & Jensen, "Conditional epistemic planning" | JELIA 2012, LNCS 7519, pp. 94–106 | verificado ([Springer](https://link.springer.com/doi/10.1007/978-3-642-33353-8_8)) |
| Aucher & Bolander, "Undecidability in epistemic planning" | IJCAI 2013, pp. 27–33 | verificado (lista de referências do EPDDL) |
| Bolander, Jensen & Schwarzentruber, "Complexity results in epistemic planning" | IJCAI 2015 | páginas não verificadas ([PDF](https://people.compute.dtu.dk/tobo/ijcai2015.pdf)) |
| Kominis & Geffner, "Beliefs in multiagent planning: From one agent to many" | ICAPS 2015, pp. 147–155 | verificado (lista de referências do EPDDL) |
| Muise, Belle, Felli, McIlraith, Miller, Pearce & Sonenberg, "Planning over multi-agent epistemic states: A classical planning approach" | AAAI 2015, pp. 3327–3334; DOI 10.1609/aaai.v29i1.9665 | verificado ([OJS](https://ojs.aaai.org/index.php/AAAI/article/view/9665)) |
| Charrier, Maubert & Schwarzentruber, "On the impact of modal depth in epistemic planning" | **IJCAI 2016**, pp. 1030–1036 | verificado ([IJCAI](https://www.ijcai.org/Abstract/16/150)) |
| Bolander, "A gentle introduction to epistemic planning: The DEL approach" | M4M 2017, EPTCS 243, pp. 1–22; DOI 10.4204/EPTCS.243.1 | verificado ([arXiv 1703.02192](https://arxiv.org/abs/1703.02192)) |
| Kominis & Geffner, "Multiagent online planning with nested beliefs and dialogue" | **ICAPS 2017**, pp. 186–194 | verificado ([OJS](https://ojs.aaai.org/index.php/ICAPS/article/view/13826)) |
| Huang, Fang, Wan & Liu, "A general multi-agent epistemic planner based on higher-order belief change" (MEPK) | IJCAI 2017, pp. 1093–1101; DOI 10.24963/ijcai.2017/152 | verificado (lista de referências do EPDDL) |
| Le, Fabiano, Son & Pontelli, "EFP and PG-EFP: Epistemic forward search planners in multi-agent domains" | ICAPS 2018, pp. 161–170; DOI 10.1609/icaps.v28i1.13881 | verificado ([OJS](https://ojs.aaai.org/index.php/ICAPS/article/view/13881)) |
| Fabiano, Burigana, Dovier & Pontelli, "EFP 2.0: A multi-agent epistemic solver with multiple e-state representations" | ICAPS 2020, pp. 101–109 | verificado ([OJS](https://ojs.aaai.org/index.php/ICAPS/article/view/6650)) |
| Bolander, Charrier, Pinchinat & Schwarzentruber, "DEL-based epistemic planning: Decidability and complexity" | *Artificial Intelligence* 287:103304, 2020 | título, periódico e ano verificados; **autores e número não verificados** |
| Cooper, Herzig, Maffre, Maris, Perrotin & Régnier, "A lightweight epistemic logic and its application to planning" | *Artificial Intelligence* 298:103437, 2021 | verificado ([DBLP](https://dblp.org/rec/journals/ai/CooperHMMPR21)) |
| Baral, Gelfond, Pontelli & Son, "An action language for multi-agent domains" (mA\*) | *Artificial Intelligence* 302:103601, 2022; DOI 10.1016/j.artint.2021.103601 | verificado (lista de referências do EPDDL) |
| Hu, Miller & Lipovetzky, "Planning with perspectives – Decomposing epistemic planning using functional STRIPS" | *JAIR* 75:489–539, 2022 | verificado ([mlanthology](https://mlanthology.org/jair/2022/hu2022jair-planning)) |
| Burigana, Felli & Montali, "DELPHIC: Practical DEL planning via possibilities" | JELIA 2023, LNCS 14281, pp. 579–594 | verificado ([PDF](https://www.inf.unibz.it/~montali/papers/burigana-etal-JELIA23-delphic.pdf)) |
| Bolander, Burigana & Montali, "Depth-bounded epistemic planning" (DAEDALUS) | KR 2025 | verificado ([KR](https://proceedings.kr.org/2025/70/)) |
| Burigana & Fabiano, "The Epistemic Planning Domain Definition Language: Official Guideline" | arXiv 2601.20969, jan. 2026 | verificado ([alphaXiv](https://www.alphaxiv.org/abs/2601.20969)) |

## Correções de venue, de autoria e pontos não verificados

As notas registraram correções em relação a enunciados recebidos ou a listagens comuns. **Komarnitsky & Shani** saiu no **AAAI 2016**, não no ICAPS 2016 ([OJS](https://ojs.aaai.org/index.php/AAAI/article/view/10406)). O seguimento de **Kominis & Geffner** é **ICAPS 2017**, não KR 2017 ([OJS](https://ojs.aaai.org/index.php/ICAPS/article/view/13826)). **Charrier, Maubert & Schwarzentruber** é **IJCAI 2016**, não KR 2016 ([IJCAI](https://www.ijcai.org/Abstract/16/150)).

O "Cooper et al. ECAI 2016" é o artigo curto **"Simple epistemic planning: Generalised gossiping"**, ECAI 2016, pp. 1563–1564. A lógica epistêmica leve aplicada a planejamento está no **AIJ 298 (2021)**, e Vianey **não** é coautor desse artigo. Vianey aparece em "Lightweight parallel multi-agent epistemic planning", KR 2020, pp. 274–283 ([DBLP](https://dblp.org/pid/250/2827)).

O artigo AIJ 175 (2011) sobre aproximação de teorias de ação em AL é de **Tu, Son, Gelfond & Morales; Baral não é coautor**. Listagens que dão "AIPS 2006" para Albore & Bertoli usam a chave DBLP do ICAPS: trata-se do ICAPS 2006. O título oficial do PKS de 2004 é "Extending the knowledge-based approach…". Bolander 2017 o cita com o título variante "PKS: Knowledge-based planning…". A ordem de autores do SDR se inverte entre a versão IJCAI 2011 (Shani & Brafman) e a JAIR 2012 (Brafman & Shani). O relatório de 2012 de Davis-Mendelow et al. aparece com nomes de workshop divergentes. A lista de referências do AIPS-02 grafa o artigo do KR'98 como "Modeling and agent's…", um erro de digitação. Há ainda uma distinção de nomes: **EPDDL (2026) não é o E-PDDL (2021)**, e **Son não é autor do EFP 2.0**, embora seja coautor do EFP original.

Por fim, Palacios & Geffner descrevem a busca do PKS como "cega (aprofundamento iterativo)", mas os artigos do PKS falam em BFS e DFS. E as páginas do AIPS-02 aparecem como 212–222 num resumo, enquanto o PDF termina na página 221.

Ficaram **não verificados**:

- volume e páginas de Tu, Son & Baral na TPLP;
- o conteúdo da descrição CPA(C)/(H) do IPC-6;
- a definição exata da subclasse completa em AIJ 175 (2011);
- a definição formal da largura contingente do CLG e da largura causal;
- o título exato do PO-PRP;
- os detalhes do esquema K(P) do K-replanner, conhecidos só pelo pôster;
- o texto integral de Albore & Bertoli e de Davis-Mendelow et al.;
- Demolombe & Pozos Parra (2000);
- o "quinto banco" LCW do PKS;
- as páginas de Bolander, Jensen & Schwarzentruber (IJCAI 2015);
- autores e número do survey AIJ 2020;
- os resultados completos do IPC 2026 epistêmico. Uma notícia secundária da UVSQ cita o planejador DEPTH como premiado ([UVSQ](https://www.uvsq.fr/depth-recompense-lors-de-la-premiere-international-epistemic-planning-competition-iepc-2026?toPdf=true)).

Também são **inferências**, sem teorema publicado encontrado:

- a equivalência formal PKS(Kf) ≡ K0;
- a coincidência entre "w(P) = 0" e "ext(δ₀) aproximável";
- a equivalência "DEL S5 monoagente = planejamento contingente", folclórica e apoiada em Bolander 2017 e Andersen et al. 2012;
- toda afirmação sobre o conteúdo da tese de Chiu.

## Conclusão

O mapa mostra que as quatro linhas não competem: são fatias de uma mesma escada de representações do conhecimento. As fatias diferem em dois eixos: onde se paga pela disjunção (nunca, como em K0 e PKS; por tags, em K_i; por contextos globais, em Son & Tu; por modelos exatos, em DEL) e quem escolhe as suposições (o modelador, no PKS; o planejador com custo, no A0; o otimismo, no K-replanner e no CLG; a amostragem, no SDR). O resultado menos óbvio é histórico. A relevância que define a largura conformante foi publicada primeiro por Son & Tu, e a contribuição específica de Palacios & Geffner é *localizar* os contextos, o que transforma uma explosão exponencial no número de fluentes numa explosão exponencial só na largura.

Para a tese de Chiu, isso muda a pergunta útil. Já não se trata de saber se (Π⁺, Π⁻) é novo; a resposta é que não é. A pergunta passa a ser em que degrau da escada a tese se declara e com qual teorema emprestado ela prova correção e delimita completude. O degrau natural é w = 0. As provas a espelhar são a Proposição 3 de Palacios & Geffner e a transição com leis estáticas de Tu, Son & Baral. O contraexemplo a assumir é o D₅ de Son & Baral. O primeiro passo de extensão, barato e com garantias publicadas, são as tags ou as cláusulas invariantes do K-replanner.
