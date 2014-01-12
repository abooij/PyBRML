#factor graph

#from abc import ABCMeta, abstractmethod
import networkx as nx
from networkx.algorithms import bipartite

from .potential import Potential

class FactorGraph(nx.Graph):
    """
    FactorGraph represents a factor graph, which is a graphical representation of how the factors of a distribution depend on variables.

    Concretely, if :math:`p(x,y)=f(x,y)g(y)`, for example by defining :math:`f(x,y)=p(x|y)` and :math:`g` the marginalization of :math:`p`,
    then the factor graph of :math:`p`, constructed from the list of :math:`f` and :math:`g`,
    is a graph with 2 variable nodes and 2 potential nodes, where the :math:`f` node is connected to both variables, and :math:`g` just
    to :math:`y`.

    Because factor nodes are always connected to variable nodes and vice versa, this graph is always a bipartite graph.
    This can be checked at runtime using :py:func:`is_fg`.

    This class is a subclass of the NetworkX Graph class.
    """

    def __init__(self, potentials=[]):
        """
        Create a factor graph from a list of potentials.

        :param potentials: factors of the distribution
        :type potentials: list of :py:class:`Potential`\ s
        """

        #cannot really do anything if no potentials are given
        assert(len(potentials)>0)

        super(FactorGraph, self).__init__()

        # save list of potentials and variables for later reference
        self.potentials=potentials
        self.variables=[]
        for pot in potentials:
            self.add_node(pot, bipartite=0)
            self.add_nodes_from(pot.variables, bipartite=1)
            self.add_star([pot,]+list(pot.variables))
            for variable in pot.variables:
                if variable not in self.variables:
                    self.variables.append(variable)
        # for each potential, create node and connect edges to all variables



    def is_fg(self):
        """Verify that this graph's types actually represent a factor graph.

        :returns: True if it is, False if not."""
        if not bipartite.is_bipartite(self): return False
        for node in self.nodes_iter():
            ispot=isinstance(node, Potential)
            for neighb in nx.all_neighbors(self, node):
                if ispot==isinstance(neighb, Potential):
                    print("Not FG: ",node, " with ", neighb)
                    return False
        return True

    def is_tree(self):
        """Check that this factor graph has no cycles.

        :returns: True if it doesn't have cycles (ie. it is a tree)."""
        return len(nx.cycle_basis(self))==0

    def sumprod(self):
        messages={} # dict: key is target, value is dict: key is source, value is message from source to target

        unity=None #unity potential (to be found below)

        schedule={}
        for node in self:
            messages[node]={} # again, this is a 2-d dict. first key is target, second key is source, value is message.

            if unity==None and isinstance(node, Potential):
                unity=node.unity() #save unity

            nbhd=self.neighbors(node)
            schedule[node]=([], nbhd, nbhd[:]) #first is neighbors we have the message from, second is the neighbors we have to send it to, third is complete neighborhood

        #nodes that are ready to send messages... initially just add the leaf nodes
        ready_nodes=[node for (node, (_, nbhd, _)) in schedule.items() if len(nbhd) == 1]

        #as long as we can process nodes
        while len(ready_nodes):
            #pop a node from the stack (DFS) and send its messages
            msg_source = ready_nodes.pop()
            children = schedule[msg_source][0] #nodes we got a message from. should be equal to messages[msg_source].keys()
            if len(schedule[msg_source][2])-len(schedule[msg_source][0]) == 1:
                msg_target = list(set(schedule[msg_source][1]) - set(schedule[msg_source][0]))[0]
            else: #received all messages, send to whoever we haven't sent to
                if len(schedule[msg_source][1]) == 0:
                    continue
                msg_target = schedule[msg_source][1][0]

            #print("Sending from [ "+str(msg_source)+" ] to [ "+str(msg_target)+" ]")

            is_pot=isinstance(msg_source, Potential)

            #incoming messages that do NOT originate from the target node
            incoming=messages[msg_source].copy()
            incoming.pop(msg_target, None)
            if is_pot:
                msg=msg_source
                # multiply this potentials by all incoming messages
                for msg_from in incoming.values():
                    msg=msg*msg_from
                msg=msg.marginalize(list(incoming.keys()))
            else: #variable
                msg=unity
                for msg_from in incoming.values():
                    msg=msg*msg_from
            messages[msg_target][msg_source]=msg

            schedule[msg_target][0].append(msg_source)
            schedule[msg_source][1].remove(msg_target)
            target_nbhd_size = len(schedule[msg_target][2])
            if target_nbhd_size-1 <= len(schedule[msg_target][0]):
                for next_target in schedule[msg_target][1]:
                    if len(set(schedule[msg_target][0]) | set([next_target])) == target_nbhd_size:
                        ready_nodes.append(msg_target)
        #print(schedule)

        #now check that all messages have been sent
        for node, (_, sent, _) in schedule.items():
            if len(sent):
                print("Did not send ",sent," for node ",node)
        return messages

    #calculate all single-variable marginals using the sum-product algorithm
    def marginals(self):
        msgs=self.sumprod()
        marginals = {}
        unity=self.potentials[0].unity()
        for var in self.variables:
            marginals[var]=unity
            for msg in msgs[var].values():
                marginals[var]=marginals[var]*msg
        return marginals

    def pretty_draw(self, *args, **kwargs):
        """
        Make a nice plot of this factor graph.

        If :samp:`pots` is a list of potentials:

        >>> graph = FactorGraph(pots)
        >>> graph.pretty_draw()
        >>> import matplotlib.pyplot as plt
        >>> plt.show()

        :param: Parameters are forwarded to the respective NetworkX calls.
        """
        pos=nx.spring_layout(self)
        nx.draw_networkx_nodes(self, pos, nodelist=[node for node in self.nodes() if isinstance(node, Potential)], node_shape='s')
        nx.draw_networkx_nodes(self, pos, nodelist=[node for node in self.nodes() if not isinstance(node, Potential)])
        nx.draw_networkx_edges(self, pos, *args, **kwargs)
        nx.draw_networkx_labels(self, pos, *args, **kwargs)
