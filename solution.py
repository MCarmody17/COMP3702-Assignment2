from operator import truediv
import sys
import time
from constants import *
from environment import *
from state import State
import heapq
"""
solution.py

This file is a template you should use to implement your solution.

You should implement each section below which contains a TODO comment.

COMP3702 2022 Assignment 2 Support Code

Last updated by njc 08/09/22
"""


class Solver:

    def __init__(self, environment: Environment):
        self.environment = environment
        self.EXIT_STATE = environment.is_solved
        #
        # TODO: Define any class instance variables you require (e.g. dictionary mapping state to VI value) here.
        #

    # === Value Iteration ==============================================================================================

    def vi_initialise(self):
        """
        Initialise any variables required before the start of Value Iteration.
        """
        #
        # TODO: Implement any initialisation for Value Iteration (e.g. building a list of states) here. You should not
        #  perform value iteration in this method.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #

        state = self.environment.get_init_state()
        env = self.environment
        self.states = []
        states = [state]
        heapq.heapify(states)
        self.differences = []
        self.converged = False

        # dict: state --> path_cost
        visited = {state: 0}
        n_expanded = 0

        while len(states) > 0:
            # self.loop_counter.inc()
            n_expanded += 1
            node = heapq.heappop(states)

            # check if this state is the goal
            # if env.is_solved(node):
            # print("solved")
            #    print("Container", len(states))
            #    print("Visited:", len(visited))
            #    return visited

            # add unvisited successors to states
            successors = node.get_successors()
            for s in successors:
                if s not in visited.keys():
                    visited[s] = s.path_cost
                    heapq.heappush(states, s)

        print("Visited:", len(visited))

        self.states = visited

        print("states:", len(states))
        self.values = {state: (69 if self.environment.is_solved(state) else 0)
                       for state in self.states}
        #self.values = {state: 0 for state in self.states}
        self.policy = {state: FORWARD for state in self.states}

        return visited

    def vi_is_converged(self):
        """
        Check if Value Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        #
        # TODO: Implement code to check if Value Iteration has reached convergence here.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        # for s in self.states:
        #   print("values: ")
        # print(self.values[s])z
        # print("new values: ")
        # print(self.new_values[s])

        if(self.converged is True):
            return True
        else:
            return False

    def stoch_action(self, a):
        """ Returns the probabilities with which each action will actually occur,
            given that action a was requested.

        Parameters:
            a: The action requested by the agent.

        Returns:
            The probability distribution over actual actions that may occur.
        """
        PCW = self.environment.drift_cw_probs
        PCCW = self.environment.drift_ccw_probs
        PDM = self.environment.double_move_probs

        # return{[0.05*(1-0.25)], [0.05*(1-0.25)], [0.05*0.25], [0.05*0.25], [0.25]}
        # print((0.05*0.75)+(0.05*0.75)+(0.05*0.25)+(0.05*0.25)+0.25)
        # if a == FORWARD:
        p_normal_move = 0.675
        p_double_move = 0.25
        p_drift_left = 0.05
        p_drift_right = 0.05
        p_drift_left_double_move = 0.05 * (0.675*0.675)
        p_drift_right_double_move = 0.05 * (0.675*0.675)

        # ----> ADD ONE FOR EACH ACTION
        return{(FORWARD): 0.675, (FORWARD, FORWARD): p_double_move, (SPIN_LEFT): p_drift_left, (SPIN_RIGHT): p_drift_right, (SPIN_LEFT, FORWARD, FORWARD): p_drift_left_double_move, (SPIN_RIGHT, FORWARD, FORWARD): p_drift_right_double_move}

    def vi_iteration(self):
        """
        Perform a single iteration of Value Iteration (i.e. loop over the state space once).
        """
        #
        # TODO: Implement code to perform a single iteration of Value Iteration here.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        new_values = {}
        new_policy = {}

        for s in self.states:
            action_values = {}
            if self.environment.is_solved(s):
                new_values[s] = 0.0
                self.converged = True

            for a in ROBOT_ACTIONS:
                total = 0
                for stoch_action, p in self.stoch_action(a).items():
                    reward, s_next = self.environment.perform_action(  # Perform action VS dynamics??
                        s, a)
                    total += p * (reward + (self.environment.gamma *
                                            self.values[s_next]))
                action_values[a] = total
            # Update state value with best action
            new_values[s] = max(action_values.values())
            new_policy[s] = dict_argmax(action_values)

        differences = [abs(self.values[s] - new_values[s])
                       for s in self.states]

        max_diff = max(differences)

        self.differences.append(max_diff)

        if max_diff < self.environment.epsilon:
            self.converged = True

        # Update values
        self.values = new_values
        self.policy = new_policy

    def vi_plan_offline(self):  # DONT CHANGE THIS
        """
        Plan using Value Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.vi_initialise()
        while not self.vi_is_converged():
            self.vi_iteration()

    def vi_get_state_value(self, state: State):
        """
        Retrieve V(s) for the given state.
        :param state: the current state
        :return: V(s)
        """
        #
        # TODO: Implement code to return the value V(s) for the given state (based on your stored VI values) here. If a
        #  value for V(s) has not yet been computed, this function should return 0.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        s = state
        return s.path_cost

    def vi_select_action(self, state: State):
        """
        Retrieve the optimal action for the given state (based on values computed by Value Iteration).
        :param state: the current state
        :return: optimal action for the given state (element of ROBOT_ACTIONS)
        """
        #
        # TODO: Implement code to return the optimal action for the given state (based on your stored VI values) here.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        return self.policy[state]

    # === Policy Iteration =============================================================================================

    def pi_initialise(self):
        """
        Initialise any variables required before the start of Policy Iteration.
        """
        #
        # TODO: Implement any initialisation for Policy Iteration (e.g. building a list of states) here. You should not
        #  perform policy iteration in this method. You should assume an initial policy of always move FORWARDS.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        pass

    def pi_is_converged(self):
        """
        Check if Policy Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        #
        # TODO: Implement code to check if Policy Iteration has reached convergence here.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        pass

    def pi_iteration(self):
        """
        Perform a single iteration of Policy Iteration (i.e. perform one step of policy evaluation and one step of
        policy improvement).
        """
        #
        # TODO: Implement code to perform a single iteration of Policy Iteration (evaluation + improvement) here.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        pass

    def pi_plan_offline(self):
        """
        Plan using Policy Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.pi_initialise()
        while not self.pi_is_converged():
            self.pi_iteration()

    def pi_select_action(self, state: State):
        """
        Retrieve the optimal action for the given state (based on values computed by Value Iteration).
        :param state: the current state
        :return: optimal action for the given state (element of ROBOT_ACTIONS)
        """
        #
        # TODO: Implement code to return the optimal action for the given state (based on your stored PI policy) here.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        pass

    # === Helper Methods ===============================================================================================
    #
    #
    # TODO: Add any additional methods here
    #
    #


def dict_argmax(d):
    max_value = max(d.values())
    for k, v in d.items():
        if v == max_value:
            return k
