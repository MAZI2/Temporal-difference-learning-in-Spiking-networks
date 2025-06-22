# Reinforcement learning in spiking neural networks
    
    Spiking network
    Designing neurons and synapses 
        (properties like eligibility traces, membrain potential, conduction delay ..., other parameteres)
    Synaptic plasticity ... STDP
    
    Rward modulated spike timing depended plasticity R-STDP 

    Learning with R-STDP:
        Using R-STDP to learn a task 

        Distal reward problem
            demonstrate association and activations of groups
            shift to reward predicting signal

        TODO: Implementing R-STDP with distal reward modulation (not hard with one row?)
        TODO-extra: Implementing R-STDP with recurrent network

        Temporal difference learning
            Basic TD algorithm
            
            Actor - critic model example
            TODO-extra: Implementing A-C algorithm with distal reward (+ recurrent network)

            RESEARCH: recurrence (results as patterns, averages) and rewarding a state ... timeline, is it possible to apply this model?

-------
Current TODO:
    (in OneNote)

    examine the basic task that is solved with RSTDP (github)
    examine why the basic 2 neuron output did not work with nest AC ... does it work with github RSTDP (is it a valid test)?

    think about how to implement RSTDP with distal reward and recurrent network
        if possible, this is the main task of dipl

    else ...
    find a simplest test that works with nest TD.
        1. try to implement with distal reward
        2. try to implement with recurrent
    else ...
        implement single row (nest impl.) of AC to solve a problem ..., try to solve pong
        implement R-STDP to play pong

        compare the two, and tasks that one solves better than other
------
Spodbujevano učnje na impulznih nevronskih mrežah:
    Izdelava modela bioloških nevronov in sinaps in sinaptične plastičnosti. Evalvacija različnih modelov.
    Izdelava simulatorja in mehanizmov za vizualizacije delovanja impulznih nevronskih mrež.
    Učenje impulznih nevronskih mrež s pomočjo nagrade.
        Model sinaptične plastičnost odvisne od nagrajevanja in časovne razporeditvre impulzov (angl. R-STDP) VIR. 
        Uporaba R-STDP za učenje preproste naloge in klasično pogojevanje.

    Učenje kompleksne naloge - igranje igre Pong:
        Modeliranje človeškega dompaminskega sistema in nevronska vezja.
            TD VIR učenje na impulznih nevronskih mrežah.
            Actor-Critic VIR model

    Primerjava zunanjih orodij in implementacij spodbujevanega učenja na impulznih nevronskih mrežah z našo implementacijo.

Direct TODO:
    using simplest models, acchieve pong learning from scratch. A platform for testing, visualizing learning.
    Connect latest with diploma for easier
    Check the implementations of R-STDP, what is it exactly?  
Can it be implemented with recurrent izhikevich dopamine system? 

Test github implementation for the simple association test ... Why does the nest one not work? ... Debug connection wise 

What is the test that he used? Can it be a more achievable goal for diploma? 

R-STDP in recurrent 
    How does aversion arise? 
    <-  Unwinding as timeline 
Aversion DT w.r.t R-STDP 

------ LATEST1: 
    Write an excerpt from Dipl research notes. Create a diagram for better thinking. 
    Check the implementations 
    Create and integrate non clustered ideas (neuron to neuron mechanisms of state->action reward signals and promotion) to the higher level brain clusters.

Notes:
stdp learning - Izhkevich synapse? (step / standard?)
EXP. Izikhevich distal reward 
    do synapses really not strengthen with no DA present? 
    Decrease of control switch of rewarded output is very important to have noise? 

EXP. Actor critic implementation with izhikhevich + distal reward Izhikhevich
the connection between distal and TD model?
the difference between actor critic and basic TD
    The critic predicts if the new action will yield reward and applies reward. Also external direct pathway to reward exists(?).  
    When a new state arrives, the network predicts the reward alongside the action? (delay?) 

FORM. **rstdp** todos
FORM. **mechanic**
FORM. Why do we strongly learn negatives 

Bayesian parameter optimization?
software simulation without injected noise (see section 2.5) is unable to progress beyond the mean expected reward received

---- LATEST2:
Novel
- what do neurotransmitters represent ... activation of group emergent behaviours/group modes?
- hardwired modulez that are not modulated(learn) are task specific. In humans life delegates what needs to be hardwired
- does snn without any constraint converge to more reward? using noise
- hardwires represent restrictions on the  regular reward maximization
- as they are equivalent to dopamine, we are actually balancing multiple mechanisms. How to interpret these mechanisms and can new be created as new neurotransmitters? how to interpet this
- dopamine as one of such modes can be used for central reward narrative, but can also be used internally for some other goal to be LEARNED. Other goals can be aversion of bad response which is triggered with some hardwire
- since even these can be potentially learned from scratch ... some predictable patterns from life that we know we need are hardwired and the specifics only are learned.
- EXP. how to abstrahize this for artificial problems
- for example pong two goals are set one that rewards not dying and another that punishes death. the first is not so certain/strong but the other is hardwired as strong. these should compete
- write out this scenario. the dopamine is in both cases just the action competes. you drawn rhis already

- plasticity stabičity paradox?
the novel approach todo:
- write out posamezna pathwaye in funkcije s pomocjo blocks
- redraw izikhevich circuit in block structure. This makes a circuit that completes some task already
- the main block is basically the decision making cortex?
- delays in block structure?
- unlabel reward term. think of it as neuromodulatory block
- bulding blocks and intermediary neuron blocks bulding
- an exaple in notes of using separate neurotransmitters
- isolate parameters that d1, d2 and potentially some other neurotransmiters can regulate
- research functions of dopamine and mechanism
- the question regardin whether dopa works on neurons directly is in sync with idea of neurotransmitters triggering actions
- EXP. existing/evolutional vs on the spot. What are the capabilities/why it fails? Try with decomposing and interactions between evolved clusters instead

## Organised TODO
RESOURCES
- download sources
- collect sources and create a structured knowledge graph from **ResearchNotes**
- poglej clanke, ki si jih navedel v DS sem 2
- associative memory that the physicist John Hopfield discovered in 1982. The Hopfield network

- multiple neuron actor critic and imperfect error?
- is hebbian true in case of weight decrease biologically?

- Connect steps. Isolate also notes on paper
- Paste here the chapters and connect them with actual TODO


## Additional keywords/chapters perhaps
rekurenčnost
diskretizacija časa
inkrementalno učenje iz podatkovnih tokov


--------------------
## Brainstorm:
Association but for what cause? 
An action is associated with reward. Therefore a thought of that action (caused by some other activity) is associated with (probably lesser reward) and therefore releases dopamine and reinforces it/ENCOURAGES IT IN THE MOMENT? The action is then carried out. Beneficial for **living** organisms 

hardwiring through evolution? learn more about serotonin and aversive behaviour first. 

how does brain distinct good or bad behaviour. Error correction mechanism? 

locality of neuron clusters (axon growth limits) 

Tudi pomen lokalnosti nevronov **FIG1**

brain probably forces change in behaviour (aversion) with certain neurotransmitters. Not exactly paths that mean do opposite. 

LATEST: 
Trenutni razmislek je, vezan na sliko na tabli, da je del možganov in predvsem aversive behaviour evolucijski. Neurotransmiterji določajo spremembo v obnašanju in lahko je to "hardwired" recimo bremzanje motoričnega dela možganov ob bolečini. Ostale stvari in inteligenca pa so asociacije stvari s temi CORE funkcionalnostmi. Recimo reward, punishment, food ... **FIG2**

## Further
Similar mechanics (from article) but with just neuron excitement/diminishment (dopamine does not influence synapses but neuron firing) 
Algebraic modeling? 
in neuron to neuron dopamine impl. probabilistic analysis might actually be useful
Critical paths? … Tracking and describing behaviour
synchronized behaviour in snns?
