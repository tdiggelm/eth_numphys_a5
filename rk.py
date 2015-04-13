#!/usr/bin/env python
#############################################################################
# course:   Numerische Methoden D-PHYS
# exercise: assignment 5
# author:   Thomas Diggelmann <thomas.diggelmann@student.ethz.ch>
# date:     13.04.2015
#############################################################################
from numpy import zeros, tile, array, shape, hstack
from scipy.optimize import fsolve

class RungeKutta(object):

    def __init__(self, A, b, c):
        """Build a new Runge-Kutta instance given a Butcher Scheme

        Input:
        A:  Butcher matrix A of shape (s,s)
        b:  Butcher vector b of shape (s,)
        c:  Butcher vector c of shape (s,)
        """
        self._A = A.copy()
        self._b = b.copy()
        self._c = c.copy()
        self._s = A.shape[0]


    def set_rhs(self, f):
        """Set the right-hand-side

        Input:
        f:  The right-hand-side f(t, y(t))
        """
        self._f = f


    def set_iv(self, T0, IV):
        """Set the initial values at time T0

        Input:
        T0:  The initial time T0
        IV:  The initial value y(T0)
        """
        self._T0 = T0
        self._IV = IV.copy()
        self._d = IV.shape[0]


    def integrate(self, Tend, steps):
        r"""
        Integrate ODE with Runge-Kutta Method

        Input:
        Tend:   Endzeit
        steps:  Anzahl Schritte

        Output:
        t, u:  Zeit und Loesung
        """
        u = zeros((self._d+1, steps))
        #################################################################################
        #                                                                               #
        # TODO: Implementieren Sie hier die Zeitevolution mittels Runge-Kutta Verfahren #
        #                                                                               #
        # Hinweis: Rufen Sie die Klassen-Methode self._step geeignet auf.               #
        #                                                                               #
        #################################################################################
        dt = float(Tend - self._T0)/steps
        u[:-1, 0] = self._IV
        u[-1, 0] = self._T0
        for i in xrange(0, steps-1):
            u[:,i+1] = self._step(u[:,i], dt)
        return u[-1,:], u[:-1,:]


    def _step(self, u, dt):
        r"""
        Makes a single Runge-Kutta step of size dt, starting from current solution u(t).

        Input:
        u:     Current solution u(t) := [y(t), t]
        dt:    Timestep size

        Output:
        unew:  New solution u(t+dt) := [y(t+dt), t+dt]
        """
        d = self._d
        s = self._s
        unew = zeros((d+1,))
        ##################################################################################
        #                                                                                #
        # TODO: Implementieren Sie hier einen einzelnen Schritt eines                    #
        #       allgemeinen impliziten Runge-Kutta Verfahrens                            #
        #                                                                                #
        # Hinweis: Das Butcher Schema ist in den Klassen-Variablen self._A, self._b      #
        #          und self._c gespeichert.                                              #
        #                                                                                #
        ##################################################################################
        f = self._f
        b = self._b
        c = self._c
        A = self._A
        tn, yn = u[-1], u[:-1]
        def F(k0):
            k0 = k0.reshape((s,d))
            k = hstack([k0[i]-f(tn+c[i]*dt, yn+dt*sum(A[i,j]*k0[j] for j in xrange(s))) for i in xrange(s)])
            return k
        k = fsolve(F, zeros(s*d)).reshape((s,d))
        ynew = yn + dt*sum(b[i]*k[i] for i in xrange(s))
        tnew = tn + dt
        return hstack([ynew, tnew])
