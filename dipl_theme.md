Reinforcement learning in spiking neural networks
    
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
Direct TODO:
    using simplest models, acchieve pong learning from scratch. A platform for testing, visualizing learning.
