from operator import truediv
import sys
import time
from constants import *
from environment import *
from state import State
import heapq
import numpy as np
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

        # ----> ADD ONE FOR EACH ACTION

        if a == FORWARD:
            p_normal_move = 0.675
            p_double_move = 0.25
            p_drift_left = 0.05
            p_drift_right = 0.05
            p_drift_left_double_move = 0.05 * (0.675*0.675)
            p_drift_right_double_move = 0.05 * (0.675*0.675)
            return{(FORWARD, FORWARD): p_double_move, (SPIN_LEFT): p_drift_left, (SPIN_RIGHT): p_drift_right, (SPIN_LEFT, FORWARD, FORWARD): p_drift_left_double_move, (SPIN_RIGHT, FORWARD, FORWARD): p_drift_right_double_move, (FORWARD): 0.675}
        elif a == REVERSE:
            p_normal_move = 0.855
            p_double_move = 0.1
            p_drift_left = 0.025
            p_drift_right = 0.025
            p_drift_left_double_move = 0.0025
            p_drift_right_double_move = 0.0025
            return{(FORWARD, FORWARD): p_double_move, (SPIN_LEFT): p_drift_left, (SPIN_RIGHT): p_drift_right, (SPIN_LEFT, FORWARD, FORWARD): p_drift_left_double_move, (SPIN_RIGHT, FORWARD, FORWARD): p_drift_right_double_move, (FORWARD): 0.855}
        return{(FORWARD, FORWARD): 0.25, (SPIN_LEFT): 0.05, (SPIN_RIGHT): 0.05, (SPIN_LEFT, FORWARD, FORWARD): 0.02278, (SPIN_RIGHT, FORWARD, FORWARD): 0.02778, (FORWARD): 0.675}

    def get_transition_probabilities(self, state, action):
        probabilities = dict()
        for action_combinations, prob in self.stoch_action(action).items():
            actions = action_combinations
            next_state = state

            # for a in actions:
            _, next_state = self.environment.apply_dynamics(
                next_state, action)

            probabilities[next_state] = probabilities.get(next_state, 0) + prob
        return probabilities

    def get_reward(self, s):
        """ Returns the reward for being in state s. """
        if s == self.environment.is_solved(s):
            return 0
        return self.rewards.get(s, 0)

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
                    reward, s_next = self.environment.apply_dynamics(  # Perform action VS dynamics??
                        s, a)
                    print(reward)
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

        state = self.environment.get_init_state()

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

            successors = node.get_successors()
            for s in successors:
                if s not in visited.keys():
                    visited[s] = s.path_cost
                    heapq.heappush(states, s)

        print("Visited:", len(visited))

        self.states = visited
        self.statesL = list(self.states)

        self.values = {state: 0 for state in self.states}
        self.policy = {pi: FORWARD for pi in self.states}
        self.r = [0 for s in self.states]
        self.r = [0 for state in self.states]
        self.USE_LIN_ALG = False

        # Transition Matrix
        self.t_model = np.zeros(
            [len(self.states), len(ROBOT_ACTIONS), len(self.states)])
        for index, state in enumerate(self.states):
            for j, a in enumerate(ROBOT_ACTIONS):
                transitions = self.get_transition_probabilities(state, a)
                for next_state, prob in transitions.items():
                    # This may not be correct?
                    self.states = list(self.states)
                    self.t_model[index][j][self.states.index(
                        next_state)] = prob

        # reward vector
        r_model = np.zeros([len(self.states)])
        for index, state in enumerate(self.states):
            rewards = []
            for action in ROBOT_ACTIONS:
                reward, _ = self.environment.apply_dynamics(state, action)
                rewards.append(reward)
            r_model[index] = min(rewards)
        self.r_model = r_model

        print(self.r_model)

        # lin alg policy
        la_policy = np.zeros([len(self.states)], dtype=np.int64)
        for index, state in enumerate(self.states):
            la_policy[index] = 1
        self.la_policy = la_policy

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
        if(self.converged is True):
            print("ITS TRUE?")
            return True
        else:
            print("Its false?")
            return False

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
        print("Do we even get here?")
        if self.USE_LIN_ALG:
            print("we get new pol?")
            new_policy = {s: ROBOT_ACTIONS[self.la_policy[i]]
                          for i, s in enumerate(self.states)}
        else:
            new_policy = {}
            for s in self.states:
                action_values = {}

                for a in ROBOT_ACTIONS:
                    total = 0
                    for stoch_action, p in self.stoch_action(a).items():
                        reward, s_next = self.environment.perform_action(  # Perform action VS dynamics??
                            s, a)

                        total += p * (reward + (self.environment.gamma *
                                                self.values[s_next]))
                    action_values[a] = total
                # Update state value with best action

                new_policy[s] = dict_argmax(action_values)

        if new_policy == self.policy:
            self.converged = True

        self.policy = new_policy
        if self.USE_LIN_ALG:
            for i, s in enumerate(self.grid.states):
                self.la_policy[i] = self.policy[s]
        return new_policy

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
        return self.policy[state]

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
