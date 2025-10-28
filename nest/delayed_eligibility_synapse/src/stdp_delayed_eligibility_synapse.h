/*
 *  stdp_delayed_eligibility_synapse.h
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 */

#ifndef STDP_DELAYED_ELIGIBILITY_SYNAPSE_H
#define STDP_DELAYED_ELIGIBILITY_SYNAPSE_H

#include "numerics.h"
#include "volume_transmitter.h"
#include "connection.h"
#include "spikecounter.h"
#include <deque>
#include <utility>

namespace nest
{

class STDPDelayedEligibilityCommonProperties : public CommonSynapseProperties
{
public:
  STDPDelayedEligibilityCommonProperties(); // fixed constructor name

  void get_status(DictionaryDatum& d) const;
  void set_status(const DictionaryDatum& d, ConnectorModel& cm);
  long get_vt_node_id() const;

  volume_transmitter* volume_transmitter_;
  double A_plus_;
  double A_minus_;
  double tau_plus_;
  double tau_c_;
  double tau_c_delay_;
  double tau_n_;
  double b_;
  double Wmin_;
  double Wmax_;
};

inline long STDPDelayedEligibilityCommonProperties::get_vt_node_id() const
{
  return volume_transmitter_ ? volume_transmitter_->get_node_id() : -1;
}

void register_stdp_delayed_eligibility_synapse(const std::string& name);

template <typename targetidentifierT>
class stdp_delayed_eligibility_synapse : public Connection<targetidentifierT>
{
public:
  typedef STDPDelayedEligibilityCommonProperties CommonPropertiesType;
  typedef Connection<targetidentifierT> ConnectionBase;

  static constexpr ConnectionModelProperties properties =
      ConnectionModelProperties::HAS_DELAY |
      ConnectionModelProperties::IS_PRIMARY |
      ConnectionModelProperties::SUPPORTS_HPC |
      ConnectionModelProperties::SUPPORTS_LBL;

  stdp_delayed_eligibility_synapse();

  stdp_delayed_eligibility_synapse(const stdp_delayed_eligibility_synapse&) = default;
  stdp_delayed_eligibility_synapse& operator=(const stdp_delayed_eligibility_synapse&) = default;

  using ConnectionBase::get_delay;
  using ConnectionBase::get_delay_steps;
  using ConnectionBase::get_rport;
  using ConnectionBase::get_target;

  void get_status(DictionaryDatum& d) const;
  void set_status(const DictionaryDatum& d, ConnectorModel& cm);
  void check_synapse_params(const DictionaryDatum& d) const;
  void check_connection(Node& src, Node& tgt, size_t receptor_type, const CommonPropertiesType& cp);


  class ConnTestDummyNode : public nest::ConnTestDummyNodeBase
  {
  public:
      using nest::ConnTestDummyNodeBase::handles_test_event;
      size_t handles_test_event(nest::SpikeEvent&, size_t) override { return nest::invalid_port; }
      size_t handles_test_event(nest::DSSpikeEvent&, size_t) override { return nest::invalid_port; }
  };

  bool send(Event& e, size_t t, const STDPDelayedEligibilityCommonProperties& cp);
  void trigger_update_weight(size_t t,
                             const std::vector<spikecounter>& modulator_spikes,
                             double t_trig,
                             const STDPDelayedEligibilityCommonProperties& cp);

  void set_weight(double w) { weight_ = w; }

private:
//  void process_delayed_c_(double t_current, const STDPDelayedEligibilityCommonProperties& cp);
  void update_modulator_(const std::vector<spikecounter>& modulator_spikes,
                         const STDPDelayedEligibilityCommonProperties& cp);

  void update_weight_(double c0, double n0, double minus_dt,
                      const STDPDelayedEligibilityCommonProperties& cp);

  void process_modulator_spikes_(const std::vector<spikecounter>& modulator_spikes,
                                 double t0, double t1,
                                 const STDPDelayedEligibilityCommonProperties& cp);

  void facilitate_(double kplus, const STDPDelayedEligibilityCommonProperties& cp);
  void depress_(double kminus, const STDPDelayedEligibilityCommonProperties& cp);
  double get_c_delayed_(double t_current, const STDPDelayedEligibilityCommonProperties& cp);

  double weight_;
  double Kplus_;
  double c_;
  // c history
  std::deque<std::pair<double, double>> c_history_;
  double n_;
  size_t modulator_spike_idx_;
  double t_last_update_;
  double t_lastspike_;
//  std::deque<std::pair<double, double>> delayed_c_;
};

// ========================================================
// Implementation
// ========================================================

// Add a helper to process all delayed eligibility traces due at current time
/*
template <typename targetidentifierT>
inline void
stdp_delayed_eligibility_synapse<targetidentifierT>::process_delayed_c_(
    double t_current,
    const STDPDelayedEligibilityCommonProperties& cp)
{
    std::cout << "[DEBUG] process_delayed_c_= " << c_ << "  at t_current=" << t_current << std::endl;
    // Apply all delayed eligibility traces that are due
    while (!delayed_c_.empty() && delayed_c_.front().first <= t_current)
    {
        double t_delayed = delayed_c_.front().first;
        double c_delayed = delayed_c_.front().second;

        // propagate weight from the time of the delayed eligibility trace to now
        double minus_dt = t_delayed - t_last_update_;

        std::cout << "[DEBUG] " << t_current << " applying delayed c: " << c_delayed << ", minus_dt=" << minus_dt << std::endl;
        update_weight_(c_delayed, n_, minus_dt, cp);

        // Remove the processed trace
        delayed_c_.pop_front();
    }

    // Update last update time to current time
    // t_last_update_ = t_current;
}
*/

template <typename targetidentifierT>
double
stdp_delayed_eligibility_synapse<targetidentifierT>::get_c_delayed_(
    double t_current,
    const STDPDelayedEligibilityCommonProperties& cp)
{
    double t_target = t_current - cp.tau_c_delay_;

    if (c_history_.empty())
        return c_; // fallback if no history

    // too early — use earliest known
    if (t_target <= c_history_.front().first)
        return c_history_.front().second;

    // find most recent entry before t_target
    for (auto it = c_history_.rbegin(); it != c_history_.rend(); ++it)
    {
        if (it->first <= t_target)
            return it->second;
    }

    return c_; // fallback (shouldn't happen)
}


template <typename targetidentifierT>
stdp_delayed_eligibility_synapse<targetidentifierT>::stdp_delayed_eligibility_synapse()
    : ConnectionBase(),
      weight_(1.0),
      Kplus_(0.0),
      c_(0.0),
      n_(0.0),
      modulator_spike_idx_(0),
      t_last_update_(0.0),
      t_lastspike_(0.0)
{
  //std::cout << "[DEBUG] stdp_delayed_eligibility_synapse constructed" << std::endl;
}

template <typename targetidentifierT>
void stdp_delayed_eligibility_synapse<targetidentifierT>::get_status(DictionaryDatum& d) const
{
  ConnectionBase::get_status(d);
  def<double>(d, names::weight, weight_);
  def<double>(d, names::Kplus, Kplus_);
  def<double>(d, names::c, c_);
  def<double>(d, names::n, n_);
}

template <typename targetidentifierT>
void stdp_delayed_eligibility_synapse<targetidentifierT>::set_status(const DictionaryDatum& d, ConnectorModel& cm)
{
  ConnectionBase::set_status(d, cm);
  updateValue<double>(d, names::weight, weight_);
  updateValue<double>(d, names::Kplus, Kplus_);
  updateValue<double>(d, names::c, c_);
  updateValue<double>(d, names::n, n_);

  if (Kplus_ < 0)
      throw BadProperty("Kplus must be non-negative.");
}

template < typename targetidentifierT >
void
stdp_delayed_eligibility_synapse< targetidentifierT >::check_synapse_params( const DictionaryDatum& syn_spec ) const
{
  // Setting of parameter c and n not thread safe.
  if ( kernel().vp_manager.get_num_threads() > 1 )
  {
    if ( syn_spec->known( names::c ) )
    {
      throw NotImplemented(
        "For multi-threading Connect doesn't support the setting "
        "of parameter c in stdp_delayed_eligibility_synapse. "
        "Use SetDefaults() or CopyModel()." );
    }
    if ( syn_spec->known( names::n ) )
    {
      throw NotImplemented(
        "For multi-threading Connect doesn't support the setting "
        "of parameter n in stdp_delayed_eligibility_synapse. "
        "Use SetDefaults() or CopyModel()." );
    }
  }
}

template < typename targetidentifierT >
inline void
stdp_delayed_eligibility_synapse< targetidentifierT >::update_modulator_( const std::vector< spikecounter >& modulator_spikes,
  const STDPDelayedEligibilityCommonProperties& cp )
{
  double minus_dt = modulator_spikes[ modulator_spike_idx_ ].spike_time_ - modulator_spikes[ modulator_spike_idx_ + 1 ].spike_time_;
  ++modulator_spike_idx_;
  n_ = n_ * std::exp( minus_dt / cp.tau_n_ ) + modulator_spikes[ modulator_spike_idx_ ].multiplicity_ / cp.tau_n_;
}

template < typename targetidentifierT >
inline void
stdp_delayed_eligibility_synapse< targetidentifierT >::update_weight_( double c0,
  double n0,
  double minus_dt,
  const STDPDelayedEligibilityCommonProperties& cp )
{
  const double taus_ = ( cp.tau_c_ + cp.tau_n_ ) / ( cp.tau_c_ * cp.tau_n_ );
  weight_ = weight_
    - c0
      * ( n0 / taus_ * numerics::expm1( taus_ * minus_dt )
        - cp.b_ * cp.tau_c_ * numerics::expm1( minus_dt / cp.tau_c_ ) );

  if ( weight_ < cp.Wmin_ )
  {
    weight_ = cp.Wmin_;
  }
  if ( weight_ > cp.Wmax_ )
  {
    weight_ = cp.Wmax_;
  }
}

template < typename targetidentifierT >
inline void
stdp_delayed_eligibility_synapse< targetidentifierT >::process_modulator_spikes_( const std::vector< spikecounter >& modulator_spikes,
  double t0,
  double t1,
  const STDPDelayedEligibilityCommonProperties& cp )
{
  // process dopa spikes in (t0, t1]
  // propagate weight from t0 to t1
  if ( ( modulator_spikes.size() > modulator_spike_idx_ + 1 )
    and ( t1 - modulator_spikes[ modulator_spike_idx_ + 1 ].spike_time_ > -1.0 * kernel().connection_manager.get_stdp_eps() ) )
  {
    // there is at least 1 dopa spike in (t0, t1]
    // propagate weight up to first dopa spike and update dopamine trace
    // weight and eligibility c are at time t0 but dopamine trace n is at time
    // of last dopa spike
    double n0 =
      n_ * std::exp( ( modulator_spikes[ modulator_spike_idx_ ].spike_time_ - t0 ) / cp.tau_n_ ); // dopamine trace n at time t0

//    update_weight_( c_, n0, t0 - modulator_spikes[ modulator_spike_idx_ + 1 ].spike_time_, cp );
    double c_delayed = get_c_delayed_(t0, cp);
    update_weight_(c_delayed, n0,
                   t0 - modulator_spikes[modulator_spike_idx_ + 1].spike_time_,
                   cp);
    update_modulator_( modulator_spikes, cp );

    // process remaining dopa spikes in (t0, t1]
    //double cd;
    while ( ( modulator_spikes.size() > modulator_spike_idx_ + 1 )
      and ( t1 - modulator_spikes[ modulator_spike_idx_ + 1 ].spike_time_ > -1.0 * kernel().connection_manager.get_stdp_eps() ) )
    {
      // propagate weight up to next dopa spike and update dopamine trace
      // weight and dopamine trace n are at time of last dopa spike td but
      // eligibility c is at time
      // t0

      //cd = c_
      //  * std::exp( ( t0 - modulator_spikes[ modulator_spike_idx_ ].spike_time_ ) / cp.tau_c_ ); // eligibility c at time of td
      //update_weight_(
      //  cd, n_, modulator_spikes[ modulator_spike_idx_ ].spike_time_ - modulator_spikes[ modulator_spike_idx_ + 1 ].spike_time_, cp );
      double spike_time_prev = modulator_spikes[modulator_spike_idx_].spike_time_;
      double spike_time_next = modulator_spikes[modulator_spike_idx_ + 1].spike_time_;

      // --- ✅ get delayed c at the time of the previous dopa spike ---
      c_delayed = get_c_delayed_(spike_time_prev, cp);
      update_weight_(c_delayed, n_,
                     spike_time_prev - spike_time_next,
                     cp);
      update_modulator_( modulator_spikes, cp );
    }

    // propagate weight up to t1
    // weight and dopamine trace n are at time of last dopa spike td but
    // eligibility c is at time t0
    //cd = c_ * std::exp( ( t0 - modulator_spikes[ modulator_spike_idx_ ].spike_time_ ) / cp.tau_c_ ); // eligibility c at time td
    //update_weight_( cd, n_, modulator_spikes[ modulator_spike_idx_ ].spike_time_ - t1, cp );
    double spike_time_last = modulator_spikes[modulator_spike_idx_].spike_time_;
    c_delayed = get_c_delayed_(spike_time_last, cp);
    update_weight_(c_delayed, n_,
                   spike_time_last - t1,
                   cp);
  }
  else
  {
    // no dopamine spikes in (t0, t1]
    // weight and eligibility c are at time t0 but dopamine trace n is at time
    // of last dopa spike
    double n0 =
      n_ * std::exp( ( modulator_spikes[ modulator_spike_idx_ ].spike_time_ - t0 ) / cp.tau_n_ ); // dopamine trace n at time t0
    //update_weight_( c_, n0, t0 - t1, cp );
    double c_delayed = get_c_delayed_(t0, cp);

    update_weight_(c_delayed, n0, t0 - t1, cp);
  }

  // update eligibility trace c for interval (t0, t1]
  c_ = c_ * std::exp( ( t0 - t1 ) / cp.tau_c_ );
  c_history_.emplace_back(t1, c_);

  // trim entries older than ~1.5× delay window
  while (!c_history_.empty() &&
         (t1 - c_history_.front().first > cp.tau_c_delay_ * 1.5))
  {
    c_history_.pop_front();
  }
}

template < typename targetidentifierT >
inline void
stdp_delayed_eligibility_synapse< targetidentifierT >::facilitate_( double kplus, const STDPDelayedEligibilityCommonProperties& cp )
{
  c_ += cp.A_plus_ * kplus;

  c_history_.emplace_back(t_last_update_, c_);
  while (!c_history_.empty() &&
         t_last_update_ - c_history_.front().first > cp.tau_c_delay_ * 1.5)
    c_history_.pop_front();

  // Schedule delayed application
  //delayed_c_.emplace_back(t_last_update_ + cp.tau_c_delay_, c_);
  //std::cout << "[DEBUG] facilitate_: c=" << c_ << std::endl;
}

template < typename targetidentifierT >
inline void
stdp_delayed_eligibility_synapse< targetidentifierT >::depress_( double kminus, const STDPDelayedEligibilityCommonProperties& cp )
{
  c_ -= cp.A_minus_ * kminus;

  c_history_.emplace_back(t_last_update_, c_);
    while (!c_history_.empty() &&
           t_last_update_ - c_history_.front().first > cp.tau_c_delay_ * 1.5)
      c_history_.pop_front();

  // Schedule delayed application
  //delayed_c_.emplace_back(t_last_update_ + cp.tau_c_delay_, c_);
  //std::cout << "[DEBUG] depress_: c=" << c_ << std::endl;
}

/**
 * Send an event to the receiver of this connection.
 * \param e The event to send
 * \param p The port under which this connection is stored in the Connector.
 */
template < typename targetidentifierT >
inline bool
stdp_delayed_eligibility_synapse< targetidentifierT >::send( Event& e, size_t t, const STDPDelayedEligibilityCommonProperties& cp )
{
  Node* target = get_target( t );
  if (!target)
  {
      std::cerr << "[ERROR] send(): target node is nullptr!" << std::endl;
      return false;
  }

  if (!cp.volume_transmitter_)
  {
      std::cerr << "[ERROR] send(): volume_transmitter_ is nullptr!" << std::endl;
      return false;
  }

  // purely dendritic delay
  double dendritic_delay = get_delay();

  double t_spike = e.get_stamp().get_ms();
  //std::cout << "[DEBUG] spike time: " << t_spike << ", weight: " << weight_ << std::endl;
  // first, apply all due delayed eligibility traces
//  this->process_delayed_c_(t_spike, cp);

  // get history of dopamine spikes
  const std::vector< spikecounter >& modulator_spikes = cp.volume_transmitter_->deliver_spikes();
  if (modulator_spike_idx_ >= modulator_spikes.size())
  {
      std::cerr << "[ERROR] modulator_spike_idx_ out of range!" << std::endl;
      modulator_spike_idx_ = 0; // reset safely
  }

  // get spike history in relevant range (t_last_update, t_spike] from
  // postsynaptic neuron
  std::deque< histentry >::iterator start;
  std::deque< histentry >::iterator finish;
  target->get_history( t_last_update_ - dendritic_delay, t_spike - dendritic_delay, &start, &finish );

  // facilitation due to postsynaptic spikes since last update
  double t0 = t_last_update_;
  double minus_dt;

  while ( start != finish )
  {
    process_modulator_spikes_( modulator_spikes, t0, start->t_ + dendritic_delay, cp );
    t0 = start->t_ + dendritic_delay;
    minus_dt = t_last_update_ - t0;
    // facilitate only in case of post- after presyn. spike
    // skip facilitation if pre- and postsyn. spike occur at the same time
    if ( t_spike - start->t_ > kernel().connection_manager.get_stdp_eps() )
    {
      facilitate_( Kplus_ * std::exp( minus_dt / cp.tau_plus_ ), cp );
    }
    ++start;
  }

  // depression due to new pre-synaptic spike
  process_modulator_spikes_( modulator_spikes, t0, t_spike, cp );
  depress_( target->get_K_value( t_spike - dendritic_delay ), cp );

  e.set_receiver( *target );
  e.set_weight( weight_ );
  e.set_delay_steps( get_delay_steps() );
  e.set_rport( get_rport() );
  e();

  Kplus_ = Kplus_ * std::exp( ( t_last_update_ - t_spike ) / cp.tau_plus_ ) + 1.0;
  t_last_update_ = t_spike;
  t_lastspike_ = t_spike;

   std::deque<histentry>::iterator hist_start, hist_end;
  target->get_history(0.0, e.get_stamp().get_ms(), &hist_start, &hist_end);

  /*
  std::cout << "[DEBUG] full history:" << std::endl;
  for (auto it = hist_start; it != hist_end; ++it)
  {
      std::cout << "  spike at t = " << it->t_ << std::endl;
  }
  */

  //std::cout << "[DEBUG] send() finished, new weight: " << weight_ << std::endl;
  return true;
}

template < typename targetidentifierT >
inline void
stdp_delayed_eligibility_synapse< targetidentifierT >::trigger_update_weight( size_t t,
  const std::vector< spikecounter >& modulator_spikes,
  const double t_trig,
  const STDPDelayedEligibilityCommonProperties& cp )
{
  /*
  if (kernel().vp_manager.get_thread_id() == 0) // print only from thread 0 to avoid spam
    {
        std::cout << "[DEBUG synapse] time=" << t_trig
                  << " c=" << c_
                  << " n=" << n_
                  << " weight=" << weight_
                  << std::endl;
        std::fflush(stdout); // ensure immediate output
    }
    */
//  this->process_delayed_c_(t_trig, cp);
  // propagate all state variables to time t_trig
  // this does not include the depression trace K_minus, which is updated in the
  // postsyn. neuron

  // purely dendritic delay
  double dendritic_delay = get_delay();

  // get spike history in relevant range (t_last_update, t_trig] from postsyn.
  // neuron
  std::deque< histentry >::iterator start;
  std::deque< histentry >::iterator finish;
  get_target( t )->get_history( t_last_update_ - dendritic_delay, t_trig - dendritic_delay, &start, &finish );

  // facilitation due to postsyn. spikes since last update
  double t0 = t_last_update_;
  double minus_dt;

  while ( start != finish )
  {
    process_modulator_spikes_( modulator_spikes, t0, start->t_ + dendritic_delay, cp );
    t0 = start->t_ + dendritic_delay;
    minus_dt = t_last_update_ - t0;
    facilitate_( Kplus_ * std::exp( minus_dt / cp.tau_plus_ ), cp );
    ++start;
  }

  // propagate weight, eligibility trace c, dopamine trace n and facilitation
  // trace K_plus to time t_trig but do not increment/decrement as there are no
  // spikes to be handled at t_trig
  process_modulator_spikes_( modulator_spikes, t0, t_trig, cp );
  n_ = n_ * std::exp( ( modulator_spikes[ modulator_spike_idx_ ].spike_time_ - t_trig ) / cp.tau_n_ );
  Kplus_ = Kplus_ * std::exp( ( t_last_update_ - t_trig ) / cp.tau_plus_ );

  t_last_update_ = t_trig;
  modulator_spike_idx_ = 0;


  double t_past = t_trig - cp.tau_c_delay_;

  // Retrieve the delayed eligibility trace
  double c_delayed = get_c_delayed_(t_past, cp);

  // Print it for debugging

  Node* post = get_target(t);

  if (post && post->get_node_id() == 22) {
  std::cout << "[DEBUG trigger_update_weight] "
            << " | post_node_id=" << (post ? post->get_node_id() : -1)
            << " | t_trig=" << t_trig
            << " | tau_c_delay_=" << cp.tau_c_delay_
            << " | t_past=" << t_past
            << " | c_current=" << c_
            << " | c_delayed=" << c_delayed
            << " | n=" << n_
            << std::endl;
  }

}

template <typename targetidentifierT>
void stdp_delayed_eligibility_synapse<targetidentifierT>::check_connection(
    Node& src,
    Node& tgt,
    size_t receptor_type,
    const CommonPropertiesType& cp)
{
    // Step 1: base class check
    ConnTestDummyNode dummy_target;
    this->Connection<targetidentifierT>::check_connection_(dummy_target, src, tgt, receptor_type);


    if ( not cp.volume_transmitter_ )
    {
      throw BadProperty( "No volume transmitter has been assigned to the dopamine synapse." );
    }

    tgt.register_stdp_connection( t_lastspike_ - get_delay(), get_delay() );

    // Add any other synapse-specific rules here
}

} // of namespace nest

#endif // of #ifndef stdp_delayed_eligibility_synapse_H
