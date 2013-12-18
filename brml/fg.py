#factor graph

#from abc import ABCMeta, abstractmethod
import networkx as nx

from .potential import Potential

class FactorGraph(nx.Graph):
    """
    FactorGraph represents a factor graph. This class is a subclass of the NetworkX Graph class.
    """

    def __init__(self, potentials=[]):
        """Create a factor graph from a list of potentials."""
        super(FactorGraph, self).__init__()
        for pot in potentials:
            self.add_node(pot, bipartite=0)
            self.add_nodes_from(pot.variables, bipartite=1)
            self.add_star([pot,]+list(pot.variables))
        # for each potential, create node and connect edges to all variables
        # done


    def is_fg(self):
        """Verify that this graph's types actually represent a factor graph."""
        from networkx.algorithms import bipartite
        if not bipartite.is_bipartite(self): return False
        for node in nx.nodes_iter(self):
            ispot=isinstance(node, Potential)
            for neighb in nx.all_neighbors(self, node):
                if ispot==isinstance(neighb, Potential):
                    print("Not FG: ",node, " with ", neighb)
                    return False
        return True

    def is_tree(self):
        """Returns True if this factor graph has no cycles."""
        return len(nx.cycle_basis(self))==0

    # Request a message from node to be sent back to parent (ie. exclude parent from neighbourhood).
    def _sumprod_step(self, node, parent):
        children = list(nx.all_neighbors(self, node))
        children.remove(parent)
        is_pot=isinstance(node, Potential)

        if is_pot:
            msg=node
            for child in children:
                msg=msg*self._sumprod_step(child, node)
            msg=msg.marginalize(children)
        else: #variable
            msg=parent.unity()
            for child in children:
                msg=msg*self._sumprod_step(child, node)
        return msg

    # compute marginal potential using the sum-product algorithm
    def sumprod(self, variable):
        """Execute the sum-product algorithm using belief propagation
        on this factor graph to compute the marginal in one variable.
        Efficiently computes

        .. math::
            p(x) = \sum_y p(x,y)

        where :math:`x` is one variable over which :math:`p` is a distribution
        defined by the potential factors,
        and :math:`y` are the remaining variables.

        .. note:: Only implemented for tree-type factor graphs."""
        if not self.is_fg(): raise "Invalid factor graph!"
        if not self.is_tree(): raise "Non-tree factor graph belief propagation not implemented!"
        if variable not in self: raise "Supplied variable not in factor graph!"

        # broadcast outward
        children = list(nx.all_neighbors(self, variable))
        if len(children)==0: raise "Isolated variable, cannot compute marginal!"

        marg = self._sumprod_step(children[0], variable)

        for child in children[1:]:
            marg=marg*self._sumprod_step(child, variable)
        return marg

    def pretty_draw(self, *args, **kwargs):
        """Make a nice plot of this factor graph.

        If :samp:`pots` is a list of potentials:

        >>> graph = FactorGraph(pots)
        >>> graph.pretty_draw()
        >>> import matplotlib.pyplot as plt
        >>> plt.show()"""
        pos=nx.spring_layout(self)
        nx.draw_networkx_nodes(self, pos, nodelist=[node for node in self.nodes() if isinstance(node, Potential)], node_shape='s')
        nx.draw_networkx_nodes(self, pos, nodelist=[node for node in self.nodes() if not isinstance(node, Potential)])
        nx.draw_networkx_edges(self, pos, *args, **kwargs)
        nx.draw_networkx_labels(self, pos, *args, **kwargs)
