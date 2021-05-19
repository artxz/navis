#    This script is part of navis (http://www.github.com/schlegelp/navis).
#    Copyright (C) 2018 Philipp Schlegel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
"""Module containing functions and classes to built `NEURON` simulator models.

Useful resources
----------------
- http://www.inf.ed.ac.uk/teaching/courses/nc/NClab1.pdf

ToDo
----
- connect neurons
- use neuron ID as GID

Examples
--------
Initialize and run a simple model. For debugging/testing only

>>> import navis
>>> from navis.interfaces import nrn
>>> import neuron

>>> # Set finer time steps
>>> neuron.h.dt = 0.025  # .01 ms

>>> # Set the temperature - how much does this matter?
>>> # Default is 6.3 (from HH model)
>>> # neuron.h.celsius = 24

>>> # This is a DA1 PN from the hemibrain dataset
>>> n = navis.example_neurons(1) / 125
>>> n.reroot(n.soma, inplace=True)

>>> # Get dendritic postsynapses
>>> post = n.connectors[n.connectors.type == 'post']
>>> post = post[post.y >= 250]

>>> # Initialize as a DrosophilaPN which uses a bunch of parameter
>>> cmp = nrn.DrosophilaPN(n, res=1)

>>> # Simulate some synaptic inputs on the first 10 input synapse
>>> cmp.add_synaptic_current(post.node_id.unique()[0:10], max_syn_cond=.1,
                             rev_pot=-10)

>>> # Add voltage recording at the soma and one of the synapses
>>> cmp.add_voltage_record(n.soma, label='soma')
>>> cmp.add_voltage_record(post.node_id.unique()[0:10])

>>> # Initialize Run for 200ms
>>> print('Running model')
>>> cmp.run_simulation(200, v_init=-60)
>>> print('Done')

>>> # Plot
>>> cmp.plot_results()

Simulate some presynaptic spikes

>>> cmp = nrn.DrosophilaPN(n, res=1)
>>> cmp.add_voltage_record(n.soma, label='soma')
>>> cmp.add_voltage_record(post.node_id.unique()[0:10])
>>> cmp.add_synaptic_input(post.node_id.unique()[0:10], syn_curr=.1, spike_no=5,
                           spike_int=50, spike_noise=1, syn_tau2=1.1,
                           syn_rev_pot=-10, cn_weight=0.04)
>>> cmp.run_simulation(200, v_init=-60)
>>> cmp.plot_results()

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config, core, utils, graph

# We will belay any import error
try:
    import neuron
except ImportError:
    raise ImportError('This interface requires the `neuron` libary to be '
                      'installed:\n pip3 install neuron\n'
                      'See also https://neuron.yale.edu/neuron/')

from neuron.units import ms, mV
neuron.h.load_file('stdrun.hoc')

# Set up logging
logger = config.logger

__all__ = []

# It looks like there can only ever be one reference to the time
# If we have multiple models, we will each reference them to this variable
main_t = None


class NeuronCompartmentModel:
    """Compartment model representing a single neuron in NEURON.

    Parameters
    ----------
    x :         navis.TreeNeuron
                Neuron to generate model for. Has to be in microns!
    res :       int
                Approximate length [um] of segments. This guarantees that
                no section has any segment that is longer than `res` but for
                small branches (i.e. "sections") the segments might be smaller.
                Lower ``res`` = more detailed simulation.

    """

    def __init__(self, x: 'core.TreeNeuron', res=1):
        """Initialize Neuron."""
        utils.eval_param(x, name='x', allowed_types=(core.TreeNeuron, ))

        # Note that we make a copy to make sure that the data underlying the
        # model will not accidentally be changed
        self.skeleton = x.copy()

        # Max section resolution per segment
        self.res = res

        # Some placeholders
        self._sections = []
        self._stimuli = {}
        self._records = {}
        self._synapses = {}

        # Generate the actual model
        self._validate_skeleton()
        self._generate_sections()

        # Add recording of time
        global main_t
        if isinstance(main_t, type(None)):
            main_t = neuron.h.Vector().record(neuron.h._ref_t)

    def __repr__(self):
        s = (f'CompartmentModel<id={self.skeleton.label},'
             f'sections={len(self.sections)};'
             f'stimuli={len(self.stimuli)};'
             f'records={len(self.records)}>'
             )
        return s

    @property
    def label(self):
        return f'CompartmentModel[{self.skeleton.label}]'

    @property
    def nodes(self) -> pd.DataFrame:
        return self.skeleton.nodes

    @property
    def cm(self) -> float:
        """Membran capacity [micro Farads / cm^2] of all sections."""
        return np.array([s.cm for s in self.sections])

    @cm.setter
    def cm(self, value: float):
        """Membran capacity [micro Farads / cm^2] for all sections."""
        for s in self.sections:
            s.cm = value

    @property
    def Ra(self) -> float:
        """Axial resistance [Ohm * cm] of all sections."""
        return np.array([s.Ra for s in self.sections])

    @Ra.setter
    def Ra(self, value: float):
        """Set axial resistance [Ohm * cm] for all sections."""
        for s in self.sections:
            s.Ra = value

    @property
    def sections(self) -> np.ndarray:
        """List of sections making up this model."""
        return self._sections

    @property
    def stimuli(self) -> dict:
        """Return mapping of node ID(s) to stimuli."""
        return self._stimuli

    @property
    def synapses(self) -> dict:
        """Return mapping of node ID(s) to synapses."""
        return self._synapses

    @property
    def records(self) -> dict:
        """Return mapping of node ID(s) to recordings."""
        return self._records

    @property
    def t(self) -> np.ndarray:
        """The global time. Should be the same for all neurons."""
        return main_t

    def _generate_sections(self):
        """Generate sections from the neuron.

        This should not be called multiple times!

        """
        # For each node find the relative (0-1) position within its section
        G = self.skeleton.graph
        node2sec = {}
        node2pos = {}
        roots = self.skeleton.root
        for i, seg in enumerate(self.skeleton.small_segments):
            # Get child->parent distances in this segment
            dist = np.array([G.edges[(c, p)]['weight']
                             for c, p in zip(seg[:-1], seg[1:])])
            # Invert so that segment/distances are parent->child
            dist = dist[::-1]
            seg = seg[::-1]
            # Insert 0 as the first distance
            dist = np.insert(dist, 0, 0)
            # Get normalized position within segment
            norm_pos = dist.cumsum() / dist.sum()

            # Drop the last point which is a branch point - unless it's the root
            if seg[0] not in roots:
                seg = seg[1:]
                norm_pos = norm_pos[1:]

            # Update dictionaries
            node2pos.update(dict(zip(seg, norm_pos)))
            node2sec.update(dict(zip(seg, [i] * len(seg))))

        self.skeleton.nodes['sec_ix'] = self.skeleton.nodes.node_id.map(node2sec)
        self.skeleton.nodes['sec_pos'] = self.skeleton.nodes.node_id.map(node2pos)

        # First generate sections
        nodes = self.skeleton.nodes.set_index('node_id')
        self._sections = []
        for i, seg in enumerate(self.skeleton.small_segments):
            # Generate segment
            sec = neuron.h.Section(name=f'segment_{i}')
            # Set length
            sec.L = graph.segment_length(self.skeleton, seg)
            # Set mean diameter
            sec.diam = nodes.loc[seg, 'radius'].mean() * 2
            # Set number of segments for this section
            # We also will make sure that each section has an odd
            # number of segments
            sec.nseg = 1 + 2 * int(sec.L / (self.res * 2))
            # Keep track of section
            self.sections.append(sec)

        self._sections = np.array(self.sections)

        # Connect segments
        for i, seg in enumerate(self.skeleton.small_segments):
            # Root does not need to be connected
            if seg[-1] in roots:
                continue
            parent = nodes.loc[seg[-1]]
            parent_sec = self.sections[parent.sec_ix]
            self.sections[i].connect(parent_sec)

    def _validate_skeleton(self):
        """Validate skeleton."""
        if self.skeleton.units and not self.skeleton.units.dimensionless:
            if self.skeleton.units.units != config.ureg.Unit('um'):
                logger.warning('Model expects coordinates in microns but '
                               f'neuron has units "{self.skeleton.units}"!')

        if len(self.skeleton.root) > 1:
            logger.warning('Neuron has multiple roots and hence consists of '
                           'multiple disconnected fragments!')

        if 'radius' not in self.skeleton.nodes.columns:
            raise ValueError('Neuron node table must have `radius` column')

        if np.any(self.skeleton.nodes.radius.values <= 0):
            raise ValueError('Neuron node table contains radii <= 0.')

    def add_synaptic_input(self, where, start=5 * ms,
                           spike_no=1, spike_int=10 * ms, spike_noise=0,
                           syn_tau1=.1 * ms, syn_tau2=10 * ms, syn_rev_pot=0,
                           syn_curr=0.1,
                           cn_thresh=10, cn_delay=1 * ms, cn_weight=0):
        """Add synaptic input to model.

        This uses the Exp2Syn synapse. All targets in `where` are triggered
        by the same NetStim - i.e. they will all receive their spike(s) at the
        same time.

        Parameters
        ----------
        where :         int | list of int
                        Node IDs at which to simulate synaptic input.

        Properties for presynaptic spikes:

        start :         int
                        Onset [ms] of first spike from beginning of simulation.
        spike_no :      int
                        Number of presynaptic spikes to produce.
        spike_int :     int
                        Interval [ms] between consecutive spikes.
        spike_noise :   float [0-1]
                        Fractional randomness in spike timing.

        Synapse properties:

        syn_tau1 :      int
                        Rise time constant [ms].
        syn_tau2 :      int
                        Decay time constant [ms].
        syn_rev_pot :   int
                        Reversal potential (e) [mV].
        syn_curr :      int
                        Synaptic current (i) [nA].

        Connection properties:

        cn_thresh :     int
                        Presynaptic membrane potential [mV] at which synaptic
                        event is triggered.
        cn_delay :      int
                        Delay [ms] between presynaptic trigger and postsynaptic
                        event.
        cn_weight :     int
                        Weight variable. This bundles a couple of synaptic
                        properties such as e.g. how much transmitter is released
                        or binding affinity at postsynaptic receptors.

        """
        where = utils.make_iterable(where)

        # Make a new stimulator
        stim = neuron.h.NetStim()
        stim.number = spike_no
        stim.start = start
        stim.noise = spike_noise
        stim.interval = spike_int

        # Go over the nodes
        nodes = self.nodes.set_index('node_id')
        for node in nodes.loc[where].itertuples():
            # Generate synapses for the nodes in question
            # Note that we are not reusing existing synapses
            # in case the properties are different
            sec = self.sections[node.sec_ix](node.sec_pos)
            syn = neuron.h.Exp2Syn(sec)
            syn.tau1 = syn_tau1
            syn.tau2 = syn_tau2
            syn.e = syn_rev_pot
            syn.i = syn_curr

            self.synapses[node.Index] = self.synapses.get(node.Index, []) + [syn]

            # Connect spike stimulus and synapse
            ncstim = neuron.h.NetCon(stim, syn)
            ncstim.threshold = cn_thresh
            ncstim.delay = cn_delay
            ncstim.weight[0] = cn_weight

            self.stimuli[node.Index] = self.stimuli.get(node.Index, []) + [ncstim, stim]

    def inject_current_pulse(self, where, start=5,
                             duration=1, current=0.1):
        """Add current injection (IClamp) stimulation to model.

        Parameters
        ----------
        where :     int | list of int
                    Node IDs at which to stimulate.
        start :     int
                    Onset (delay) [ms] from beginning of simulation.
        duration :  int
                    Duration (dur) [ms] of injection.
        current :   float
                    Amount (i) [nA] of injected current.

        """
        self._add_stimulus('IClamp', where=where, delay=start,
                           dur=duration, i=current)

    def add_synaptic_current(self, where, start=5, tau=0.1, rev_pot=0,
                             max_syn_cond=0.1):
        """Add synaptic current(s) (AlphaSynapse) to model.

        Parameters
        ----------
        where :         int | list of int
                        Node IDs at which to stimulate.
        start :         int
                        Onset [ms] from beginning of simulation.
        tau :           int
                        Decay time constant [ms].
        rev_pot :       int
                        Reverse potential (e) [mV].
        max_syn_cond :  float
                        Max synaptic conductance (i) [uS].

        """
        self._add_stimulus('AlphaSynapse', where=where, onset=start,
                           tau=tau, e=rev_pot, gmax=max_syn_cond)

    def _add_stimulus(self, stimulus, where, **kwargs):
        """Add generic stimulus."""
        if not callable(stimulus):
            stimulus = getattr(neuron.h, stimulus)

        where = utils.make_iterable(where)

        nodes = self.nodes.set_index('node_id')
        for node in nodes.loc[where].itertuples():
            sec = self.sections[node.sec_ix](node.sec_pos)
            stim = stimulus(sec)

            for k, v in kwargs.items():
                setattr(stim, k, v)

            self.stimuli[node.Index] = self.stimuli.get(node.Index, []) + [stim]

    def add_voltage_record(self, where, label=None):
        """Add voltage recording to model.

        Parameters
        ----------
        where :     int | list of int
                    Node IDs at which to record.
        label :     str, optional
                    If label is given, this recording will be added as
                    ``self.records[label]`` else  ``self.records[node_id]``.

        """
        self._add_record(where, what='v', label=label)

    def add_current_record(self, where, label=None):
        """Add voltage recording to model.

        Parameters
        ----------
        where :     int | list of int
                    Node IDs at which to record.
        label :     str, optional
                    If label is given, this recording will be added as
                    ``self.records[label]`` else  ``self.records[node_id]``.

        """
        self._add_record(where, what='i', label=label)

    def _add_record(self, where, what, label=None):
        """Add a recording to given node.

        Parameters
        ----------
        where :     int | list of int
                    Node IDs at which to record.
        what :      str
                    What to record. Can be e.g. `v` or `_ref_v` for Voltage.
        label :     str, optional
                    If label is given, this recording will be added as
                    ``self.records[label]`` else  ``self.records[node_id]``.

        """
        where = utils.make_iterable(where)

        if not isinstance(what, str):
            raise TypeError(f'Required str e.g. "v", got {type(what)}')

        if not what.startswith('_ref_'):
            what = f'_ref_{what}'

        nodes = self.nodes.set_index('node_id')
        for node in nodes.loc[where].itertuples():
            sec = self.sections[node.sec_ix](node.sec_pos)
            rec = neuron.h.Vector().record(getattr(sec, what))

            if label:
                self.records[label] = rec
            else:
                self.records[node.Index] = rec

    def clear_records(self):
        """Clear records."""
        self._records = {}

    def clear_stimuli(self):
        """Clear records."""
        self._stimuli = {}

    def clear_synapses(self):
        """Clear records."""
        self._synapses = {}

    def clear(self):
        """Attempt to remove model from NEURON space.

        This is not guaranteed to work. Check `neuron.h.topology()` to inspect.

        """
        # Basically we have to bring the reference count to zero
        self.clear_records()
        self.clear_stimuli()
        self.clear_synapses()
        for s in self._sections:
            del s
        self._sections = []

    def insert(self, mechanism, subset=None, **kwargs):
        """Insert biophysical mechanism for model.

        Parameters
        ----------
        mechanism : str
                    Mechanism to insert - e.g. "hh" for Hodgkin-Huxley kinetics.
        subset :    list of int
                    Indices of segments to set mechanism for. If ``None`` will
                    add mechanism to all segments.
        **kwargs
                    Use to set properties for mechanism.

        """
        if not subset:
            sections = self.sections
        else:
            sections = self.sections[subset]

        for sec in utils.make_iterable(sections):
            _ = sec.insert(mechanism)
            for seg in sec:
                mech = getattr(seg, mechanism)
                for k, v in kwargs.items():
                    setattr(mech, k, v)

    def uninsert(self, mechanism, subset=None):
        """Remove biophysical mechanism from model.

        Parameters
        ----------
        mechanism : str
                    Mechanism to remove - e.g. "hh" for Hodgkin-Huxley kinetics.
        subset :    list of int
                    Indices of segments to remove the mechanism from. If
                    ``None`` will try to remove it from all segments.

        """
        if not subset:
            sections = self.sections
        else:
            sections = self.sections[subset]

        for sec in utils.make_iterable(sections):
            if hasattr(sec, mechanism):
                _ = sec.uninsert(mechanism)

    def plot_structure(self):
        """Visualize structure in 3D using matplotlib."""
        _ = neuron.h.PlotShape().plot(plt)

    def run_simulation(self, duration=25 * ms, v_init=-65 * mV):
        """Run the simulation."""
        # This resets the entire model space not just this neuron!
        neuron.h.finitialize(v_init)
        neuron.h.continuerun(duration)

    def plot_results(self, ax=None):
        """Plot results."""
        if not len(self.t):
            logger.warning('Looks like the simulation has not yet been run.')
            return
        if not self.records:
            logger.warning('Nothing to plot: no recordings found.')
            return

        if not ax:
            fig, ax = plt.subplots()

        for k, v in self.records.items():
            ax.plot(self.t, v, label=k)

        ax.set_xlabel('time [ms]')
        ax.set_ylabel('voltage [mV]')

        ax.legend()
        return ax


class DrosophilaPN(NeuronCompartmentModel):
    """Compartment model of a olfactory projection neuron in Drosophila.

    Uses biophysical properties from Tobin et al. (2017).

    Parameters
    ----------
    x :         navis.TreeNeuron
                Neuron to generate model for. Has to be in microns!
    res :       int
                Approximate length [um] of segments. This guarantees that
                no section has any segment that is longer than `res` but for
                small branches (i.e. "sections") the segments might be smaller.
                Lower ``res`` = more detailed simulation.
    passive :   bool
                If True, will insert passive membrane properties.
    active :    bool
                If True, will insert active (spiking) membrane properties.

    """

    def __init__(self, x, res=1, passive=True, active=False):
        super().__init__(x, res=res)

        self.Ra = 266.1  # specific axial resistivity in Ohm cm
        self.cm = 0.8    # specific membrane capacitance in mF / cm**2

        # Add passive membran properties
        self.insert('pas',
                    g=1/20800,  # specific leakage conductance = 1/Rm; Rm = specific membran resistance in Ohm cm**2
                    e=-60,      # leakage reverse potential
                    )
