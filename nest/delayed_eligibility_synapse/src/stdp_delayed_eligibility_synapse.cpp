#include "stdp_delayed_eligibility_synapse.h"

#include "common_synapse_properties.h"
#include "connector_model.h"
#include "event.h"
#include "kernel_manager.h"
#include "nest_impl.h"

#include "dictdatum.h"

namespace nest
{

// ------------------------------
// STDPDelayedEligibilityCommonProperties
// ------------------------------
STDPDelayedEligibilityCommonProperties::STDPDelayedEligibilityCommonProperties()
    : CommonSynapseProperties()
  , volume_transmitter_( nullptr )
  , A_plus_( 1.0 )
  , A_minus_( 1.5 )
  , tau_plus_( 20.0 )
  , tau_c_( 1000.0 )
  , tau_c_delay_(0.0)
  , tau_n_( 200.0 )
  , b_( 0.0 )
  , Wmin_( 0.0 )
  , Wmax_( 200.0 )
{
}

void STDPDelayedEligibilityCommonProperties::get_status(DictionaryDatum& d) const
{
  CommonSynapseProperties::get_status( d );

  const NodeCollectionDatum vt = NodeCollectionDatum( NodeCollection::create( volume_transmitter_ ) );
  def< NodeCollectionDatum >( d, names::volume_transmitter, vt );

  def< double >( d, names::A_plus, A_plus_ );
  def< double >( d, names::A_minus, A_minus_ );
  def< double >( d, names::tau_plus, tau_plus_ );
  def< double >( d, names::tau_c, tau_c_ );
  def< double >( d, names::tau_c, tau_c_delay_ );
  def< double >( d, names::tau_n, tau_n_ );
  def< double >( d, names::b, b_ );
  def< double >( d, names::Wmin, Wmin_ );
  def< double >( d, names::Wmax, Wmax_ );
}

void STDPDelayedEligibilityCommonProperties::set_status(const DictionaryDatum& d, ConnectorModel& cm)
{
  CommonSynapseProperties::set_status( d, cm );

  NodeCollectionDatum vt_datum;
  if ( updateValue< NodeCollectionDatum >( d, names::volume_transmitter, vt_datum ) )
  {
    if ( vt_datum->size() != 1 )
    {
      throw BadProperty( "Property volume_transmitter must be a single element NodeCollection" );
    }

    const size_t tid = kernel().vp_manager.get_thread_id();
    Node* vt_node = kernel().node_manager.get_node_or_proxy( ( *vt_datum )[ 0 ], tid );
    volume_transmitter* vt = dynamic_cast< volume_transmitter* >( vt_node );
    if ( not vt )
    {
      throw BadProperty( "Property volume_transmitter must be set to a node of type volume_transmitter" );
    }

    volume_transmitter_ = vt;
  }

  updateValue< double >( d, names::A_plus, A_plus_ );
  updateValue< double >( d, names::A_minus, A_minus_ );
  updateValue< double >( d, names::tau_plus, tau_plus_ );
  updateValue< double >( d, names::tau_c, tau_c_ );
  updateValue< double >( d, names::tau_c, tau_c_delay_ );
  updateValue< double >( d, names::tau_n, tau_n_ );
  updateValue< double >( d, names::b, b_ );
  updateValue< double >( d, names::Wmin, Wmin_ );
  updateValue< double >( d, names::Wmax, Wmax_ );
}

// ------------------------------
// Module registration
// ------------------------------
void register_stdp_delayed_eligibility_synapse(const std::string& name)
{
    register_connection_model<stdp_delayed_eligibility_synapse>(name);
}

} // namespace nest
