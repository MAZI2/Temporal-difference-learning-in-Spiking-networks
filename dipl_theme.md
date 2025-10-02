Find the nest association test ... redefine it and test. Make some visualizations to debug .. will come in handy later
Aversive behaviour ... later
----------
1. Modeliranje nevronov in sinaps
    - Predstavitev bioloških značilnosti nevronov in sinaps, ki bodo modelirane v nadaljevanju 0.5T
    - Hebbian rule, Izhikhevich, Hodgin-Hyxley
    - properties like eligibility traces, membrain potential, conduction delay ..., other parameteres
    - STDP
        stdp learning - Izhkevich synapse? (step / standard?)
    - redefinicija actor environment cikla ... vec o tem pri Nevronskih vezjih

2. Simulacija in vizualizacija
    - Mehanizmi znotraj
    - Diagrami
    - Framework za Pong **

3. RL on SNN
    - R-STDP 
        - implementation 1T
        - problem rešen z R-STDP 3T
            Github Primer
            Test github implementation for the simple association test ... Why does the nest one not work? ... Debug connection wise 

        FORM: R-STDP in recurrent 
            How does aversion arise? 
            <-  Unwinding as timeline 
        FORM: Aversion DT w.r.t R-STDP 
        FORM: software simulation without injected noise (see section 2.5) is unable to progress beyond the mean expected reward received

        Novelty TODO 1: Implementing R-STDP with distal reward modulation (not hard with one row?)
            - this should then work for TD. 
            - If not try to understand why not or what's different
        Novelty TODO 2: EXP. Implementing R-STDP with recurrent network

    - Distal reward problem
        - predstavitev 1T
            DONE (adapt on current implementation) demonstrate association and activations of groups
            DONE shift to reward predicting signal
        EXP. Izikhevich distal reward 
            do synapses really not strengthen with no DA present? 
            Decrease of control switch of rewarded output is very important to have noise? 

    - Credit assignment
        - DONE predstavitev 1T
    - TD
        - TODO 3: Na trenutni simulaciji: problem rešen s TD (pong?) single layer 1T
        - Novelty TODO 4: EXP. credit assignment on TD as izhikhevich .. compare the approach? (distal reward paper) 2T
            mejbi tukaj sploh neda ker rabis the pathways of actor-critic
    - Actor-Critic
        - TODO 5: Na trenutni simulaciji: implementacija 1 layer 1T
            examine why the basic 2 neuron output did not work with nest AC ... does it work with github RSTDP (is it a valid test)?
        - TODO 6: poskus/opazovanje expand to multilayer 1T
        - TODO 7: EXP. adaptacija to use credit assignment?
    
    - Nevronska vezja TODO: izraz? - Posplošitev actor-critic
        - human dopamine system
            - talk on behaviour
            - formulacija problema, identifikacija pathways
            - task, ki ga rešujemo

        - Novelty TODO 8: based on todo 7:
            - potentially adapt the rule / posplositev rules and identifikacija posameznih parametrov nevrona
            - introduce new neurotransmitters

        - Novel approach to solving this
            - TODO: 10.1. is hebbian true/suitable in case of weight decrease biologically?
                - compare the models of learning rules ... what do they actually differ in and what do I want to use
            FORM. **mechanic**
            FORM. Why do we strongly learn negatives 

            - what do neurotransmitters represent ... activation of group emergent behaviours/group modes?
            - hardwired modulez that are not modulated(learn) are task specific. In humans life delegates what needs to be hardwired
            - TODO 9: EXP. does snn without any constraint converge to more reward? using noise
            - hardwires represent restrictions on the regular reward maximization
                - existing/evolutional vs on the spot. What are the capabilities/why it fails? Try with decomposing and interactions between evolved clusters instead
            - as they are equivalent to dopamine, we are actually balancing multiple mechanisms. How to interpret these mechanisms and can new be created as new neurotransmitters? how to interpet this
            - dopamine as one of such modes can be used for central reward narrative, but can also be used internally for some other goal to be LEARNED. Other goals can be aversion of bad response which is triggered with some hardwire
            - since even these can be potentially learned from scratch ... some predictable patterns from life that we know we need are hardwired and the specifics only are learned.

            the novel approach TODO:
            - TODO 10: EXP. redraw izikhevich circuit in block structure. This makes a circuit that completes some task already
                write out posamezna pathwaye in funkcije s pomocjo blocks
                the main block is basically the decision making cortex?
                - delays in block structure? AKA does it work?

            - unlabel reward term. think of it as neuromodulatory block
            - bulding blocks and intermediary neuron blocks bulding
            - isolate parameters that d1, d2 and potentially some other neurotransmiters can regulate
            - research functions of dopamine and mechanism
            - the question regardin whether dopa works on neurons directly is in sync with idea of neurotransmitters triggering actions
        
    - Pong
        - poskus igranja z kakor daleč pridem z Actor-Critic in posplošitvijo. 
        - TODO 11: EXP. how to abstrahize this for artificial problems
            - for example pong two goals are set one that rewards not dying and another that punishes death. the first is not so certain/strong but the other is hardwired as strong. these should compete
            - write out this scenario. the dopamine is in both cases just the action competes. you drawn rhis already


## Organised TODO
RESOURCES
- Experiment tasks 
    If it doesn't work ... rethink and try to write a chapter on WHY it doesn't work in connection to models of synapse updates
    Read the problem with modelling article

- poglej clanke, ki si jih navedel v DipSem sem 2
- collect sources and create a structured knowledge graph from **ResearchNotes**
- associative memory that the physicist John Hopfield discovered in 1982. The Hopfield network

---------------------
## Additional keywords/chapters perhaps
plasticity stabičity paradox?
rekurenčnost
diskretizacija časa
inkrementalno učenje iz podatkovnih tokov
Bayesian parameter optimization?

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

The NMDA modelling and the nest approximation example
https://nest-simulator.readthedocs.io/en/v3.8/model_details/Brunel_Wang_2001_Model_Approximation.html