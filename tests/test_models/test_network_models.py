import pytest
import networkx as nx
import numpy as np
import pandas as pd

from navis.models.network_models import (
    TraversalModel,
    BayesianTraversalModel,
    linear_activation_p,
    linear_activation_with_neg,
)

def test_traversal_models():
    models = (TraversalModel, BayesianTraversalModel)

    G = nx.path_graph(10, create_using=nx.DiGraph)
    G.add_edge(0, 9)
    G.add_node(10)
    edges = nx.to_pandas_edgelist(G)
    edges['weight'] = np.ones(edges.shape[0])

    results = {}
    for m in models:
        model = m(edges, seeds=[1], max_steps=8)
        model.run(iterations=1)
        res = model.summary
        if res.index.name != 'node':  # issue here with older pandas versions
            res.set_index('node', inplace=True)
        assert 0 not in res.index
        assert 9 not in res.index
        assert 10 not in res.index
        for i in range(1, 9):
            row = res.loc[i]
            assert row.layer_min == row.layer_max
        results[m] = res

    for m in models:
        pd.testing.assert_frame_equal(results[TraversalModel], results[m], check_dtype=False)


def test_bayesian_batched_path_dag():
    """On a path graph the batched BTM should hit each node exactly at
    the corresponding step. cmf entries before that step must be 0,
    entries at-or-after must be 1 (weight=1 => certain traversal)."""
    G = nx.path_graph(6, create_using=nx.DiGraph)
    edges = nx.to_pandas_edgelist(G)
    edges['weight'] = 1.0

    model = BayesianTraversalModel(edges, seeds=[0], max_steps=6)
    res = model.run()
    cmf_by_node = dict(zip(res['node'], res['cmf']))

    # Node i is reached at step i (seed at step 0).
    for i in range(6):
        cmf = np.asarray(cmf_by_node[i])
        assert np.all(cmf[:i] == 0.0), (i, cmf)
        assert np.all(cmf[i:] == 1.0), (i, cmf)


def test_bayesian_batched_inhibitory():
    """Inhibitory (negative-weight) edges multiplicatively suppress the
    activation probability accumulated from excitatory inputs.
    With one excitatory and one fully-inhibitory parent both at cmf=1,
    the target's cmf should stay at 0."""
    edges = pd.DataFrame(
        [[0, 2, 0.9], [1, 2, -0.9]],
        columns=['source', 'target', 'weight'],
    ).astype({'source': 'int64', 'target': 'int64', 'weight': 'float64'})

    def act(w):
        return linear_activation_with_neg(w, neg_w=-1.0, pos_w=1.0)

    model = BayesianTraversalModel(
        edges, seeds=[0, 1], max_steps=4, traversal_func=act
    )
    res = model.run()
    cmf_by_node = dict(zip(res['node'], res['cmf']))
    cmf2 = np.asarray(cmf_by_node[2])
    # new_pmf = (1 - (1 - 0.9)) * (1 - 0.9) = 0.9 * 0.1 = 0.09 per step
    assert cmf2[0] == 0.0
    assert np.isclose(cmf2[1], 0.09)
    # If both parents were excitatory (no inhibition), cmf2[1] would be 0.99.
    assert cmf2[1] < 0.5


def test_bayesian_batched_summary_columns():
    """The summary DataFrame must expose the documented layer columns."""
    G = nx.path_graph(5, create_using=nx.DiGraph)
    edges = nx.to_pandas_edgelist(G)
    edges['weight'] = 1.0
    model = BayesianTraversalModel(edges, seeds=[0], max_steps=5)
    model.run()
    s = model.summary
    for col in ('layer_min', 'layer_max', 'layer_mean', 'layer_median'):
        assert col in s.columns
