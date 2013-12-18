from abc import ABCMeta, abstractmethod

class Potential(metaclass=ABCMeta):
    """
    Potential represents a (factor of a) probability function. This is an abstract base class instantiated by e.g. :py:class:`TablePotential`

    .. note::
        Should we have some smart way to handle __truediv__?
    """

    variables=[] # variables over which the probability function varies. should be hashable.
    priori_variables=[] # variables over which this potential is prior

    @abstractmethod
    def multiply(self, other):
        """
        Multiply a Potential pointwise (over its variables' values) with another. Returns the new potential.

        .. math:: f_1\\cdot f_2
        """
        pass

    @abstractmethod
    def make_priori(self, variables=[]):
        """ Make certain variables priori variables, ie. compute

        .. math::
            f(x,variables):=f(x,variables)/f(variables)

        where self is :math:`f(x,variables)`, and :math:`f(variables)` is computed using :py:func:`marginalize`. """
        pass

    @abstractmethod
    def evaluate(self, values):
        """ Compute the probability function when a certain variable's value is given. Argument is zipped list of variable values.

        .. note::
            Maybe return numerical value sometimes? """
        pass

    @abstractmethod
    def marginalize(self, variables):
        """ Marginalize the distribution, ie. compute

        .. math::
            f(x) = \sum_y f(x,y)

        Argument is list of variables to *forget* (corresponding to :math:`y`). """
        pass

    @abstractmethod
    def __str__(self):
        pass

    def __mul__(self, other):
        """alias of :py:func:`multiply`"""
        return self.multiply(other)

    @abstractmethod
    def __hash__(self):
        pass
    @abstractmethod
    def __eq__(self, other):
        pass
    #def __ne__(self, other):
    #    if type(self)!=type(other): return False
    #    return not(self==other)
    __ne__ = lambda x,y: not(x==y)

    @abstractmethod
    def unity(self):
        """Generate a unity potential which must have the property that

        >>> pot.unity() * pot == pot
        True
        """
        pass


import numpy
class TablePotential(Potential):
    """
    TablePotential represents a discrete distribution.
    """
    def __init__(self, variables, table):
        """Create a table potential. :ref:`table` should be a numpy table whose nth dimension represents the nth variable in the list variables."""
        self.table=table
        self.variables=tuple(variables)

    def __str__(self):
        return "Variables: "+str(self.variables)+"\nTable:\n"+str(self.table)

    def __hash__(self):
        return hash(self.variables) ^ hash(str(self.table))

    def __eq__(self, other):
        if type(self) != type(other): return False
        return numpy.all(self.table==other.table) and self.variables==other.variables

    def unity(self):
        return TablePotential([], numpy.array(1))

    def _expand_table(self, new_variables): # adds axes to numpy arrays for purposes of additional variables and brings them in the right order
        old_variables=list(self.variables)
        new_variables=list(new_variables)
        missing_variables=[var for var in new_variables if var not in old_variables]
        expanded_table=self.table

        for variable in missing_variables:
            n = len(expanded_table.shape) # insert idx
            expanded_table = numpy.expand_dims(expanded_table, n)
            old_variables.insert(n, variable)
            #print("expanding ",variable[1],n)
            expanded_table = numpy.repeat(expanded_table, variable[1], axis=n)
        for n, variable in enumerate(new_variables):
            old_idx=old_variables.index(variable)
            if old_idx==n: continue # do we need to transpose the array (ie. swap axes)?

            expanded_table=numpy.swapaxes(expanded_table, old_idx, n)
            swap_save=old_variables[n]
            old_variables[n]=old_variables[old_idx]
            old_variables[old_idx]=swap_save
        return expanded_table

    def multiply(self, other):
        self_variables=list(self.variables)
        other_variables=list(other.variables)
        new_variables=list(set(self_variables+other_variables)) # combine variable lists
        new_variables.sort()

        new_table=numpy.multiply(self._expand_table(new_variables), other._expand_table(new_variables))

        new_potential=TablePotential(new_variables, new_table)
        return new_potential
        # FIXME compute new list of priori variables!

    def divide(self, other):
        """Computes

        .. math::
            f(x)/g(y)

        where :math:`f` and :math:`g` are potentials which may or may not depend on the same variables.
        """
        self_variables=list(self.variables)
        other_variables=list(other.variables)
        new_variables=list(set(self_variables+other_variables)) # combine variable lists
        new_variables.sort()

        new_table=numpy.divide(self._expand_table(new_variables), other._expand_table(new_variables))

        new_potential=TablePotential(new_variables, new_table)
        return new_potential

    def __truediv__(self, other):
        """alias of :py:func:`divide`"""
        return self.divide(other)

    def make_priori(self, variables):
        forget_vars=set(self.variables) - set(variables)
        return self.divide(self.marginalize(forget_vars))

    def marginalize(self, variables):
        #print("marg ",variables, self.variables)
        indices=[self.variables.index(x) for x in variables]
        indices.sort()
        indices.reverse()

        new_table=self.table

        for idx in indices:
            new_table=numpy.sum(new_table, idx)

        new_potential=TablePotential([x for x in self.variables if x not in variables], new_table)

        return new_potential

    def evaluate(self, values):
        import copy
        eval_pot = copy.copy(self) #evaluated potential
        for variable, value in values:
            #print("Evaluating",variable,value)
            if variable not in self.priori_variables:
                eval_pot = eval_pot.make_priori([variable])
            #print(eval_pot.table, value, eval_pot.variables.index(variable))
            var_axis=eval_pot.variables.index(variable)
            new_shape=list(eval_pot.table.shape)
            new_shape.pop(var_axis)
            eval_pot.table = numpy.take(eval_pot.table, [value], axis=var_axis)
            eval_pot.table=numpy.reshape(eval_pot.table, new_shape)
            variables_list=list(eval_pot.variables)
            variables_list.pop(var_axis)
            eval_pot.variables=tuple(variables_list)

        return eval_pot

    def normalize(self):
        """Normalize a potential such that

        .. math::
            \sum_{x} f(x)=1
        """
        return self/self.marginalize(self.variables)
