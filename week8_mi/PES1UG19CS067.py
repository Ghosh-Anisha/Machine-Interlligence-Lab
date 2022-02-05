import numpy as np


class HMM:
    """
    HMM model class
    Args:
        A: State transition matrix
        states: list of states
        emissions: list of observations
        B: Emmision probabilites
    """

    def __init__(self, A, states, emissions, pi, B):
        self.A = A
        self.B = B
        self.states = states
        self.emissions = emissions
        self.pi = pi
        self.N = len(states)
        self.M = len(emissions)
        self.make_states_dict()

    def make_states_dict(self):
        """
        Make dictionary mapping between states and indexes
        """
        self.states_dict = dict(zip(self.states, list(range(self.N))))
        self.emissions_dict = dict(zip(self.emissions, list(range(self.M))))

    def viterbi_algorithm(self, seq):
        """
        Function implementing the Viterbi algorithm
        Args:
            seq: Observation sequence (list of observations. must be in the emmissions dict)
        Returns:
            nu: Porbability of the hidden state at time t given an obeservation sequence
            hidden_state_sequence: Most likely state sequence 
        """
        # TODO
        emission_seq_067 = []
        for i in seq:
            emission_seq_067.append(self.emissions_dict[i])

        prob_067 = [] 
        prob_067.append(tuple(self.pi[i]*self.B[i, emission_seq_067[0]] for i in range(self.N)))

        for i in range(1, len(emission_seq_067)):
            curr_067 = []
            prev_067 = prob_067[len(prob_067)-1]
            
            for j in range(len(self.A[0,:])):
                prob_value=0
                for k in range(self.N):
                    prob_value=max(prob_value,prev_067[k]*self.A[k,j]*self.B[j,emission_seq_067[i]])
                curr_067.append(prob_value)
            prob_067.append(tuple(curr_067))

        ans_067 = []
        for i in prob_067:
            index = self.states[np.argmax(i)]
            ans_067.append(index)
            
        return ans_067
        