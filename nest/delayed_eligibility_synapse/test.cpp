#include "connection.h"
#include "node.h"
#include "spike_event.h"
#include "kernel_manager.h"
#include "stdp_delayed_eligibility_synapse.h"
#include <iostream>

int main()
{
    nest::Kernel::get_instance(); // initialize the NEST kernel

    // Prepare common properties
    nest::STDPDelayedEligibilityCommonProperties cp;

    // Create a dummy synapse
    nest::stdp_delayed_eligibility_synapse<nest::TargetIdentifierIndex> syn;

    // Send a dummy spike
    nest::SpikeEvent e;
    syn.send(e, 0, cp);

    std::cout << "Test completed" << std::endl;
    return 0;
}
