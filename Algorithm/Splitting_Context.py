import numpy as np
from Environments.Users import UserC1,UserC2,UserC3
from Environments.Context_environment import ContextEnvironment

class Context():

    def __init__(self):
        self._possible_splits = [[0,0],[1,0],[0,1],[1,1]]
        self._current_split = [0,0,0,0]
        self._all_split = False  #boolean to check if all the possible splits have been done
        self_first_split = True
        self._confidence_interval = 1e-5

    def split_context_tree(self):
        """
        Method used to split the context in two parts
        """
        #TO BE IMPLEMENTED
        feature_1_split = self._current_split[:2]
        feature_2_split = self._current_split[2:]
        #Complex split
        feature_2_1_split = self._current_split[2:3]
        feature_2_2_split = self._current_split[3:]
        feature_1_1_split = self._current_split[:1]
        feature_1_2_split = self._current_split[1:2]


        pass
    def evaluate_splitting(self,rewards,users_collected):
        """
        Method used to evaluate the splitting of the users
        """
        n_c1 = users_collected.count(1)
        n_c2 = users_collected.count(2)
        n_c3 = users_collected.count(3)

        total_user = len(users_collected)

        for split in self._possible_splits:
            if self._first_split:
                # Divides C3 from the rest
                feature_1_split = self._current_split[:2]
                rewards_C3 = [value for value, num in zip(rewards, users_collected) if num == 3]
                other_rew = [value for value, num in zip(rewards, users_collected) if num != 3]
                prob_c3 = n_c3/total_user
                prob_other = (total_user-n_c3)/total_user
                if self.calculate_splitting_condition(prob_c3,prob_other,rewards_C3,other_rew,rewards):
                    context = ContextEnvironment()

                # Divides C3+C1 and C3+C2
                feature_2_split = self._current_split[2:]
                rewards_C3_C1 = [value for value, num in zip(rewards, users_collected) if num == 3 or num == 1]
                prob_C3_C1 = (n_c3+n_c1)/total_user
                rewards_C3_C2 = [value for value, num in zip(rewards, users_collected) if num == 3 or num == 2]
                prob_C3_C2 = (n_c3+n_c2)/total_user
                self.calculate_splitting_condition(prob_C3_C1, prob_C3_C1, rewards_C3_C2, prob_C3_C2, rewards)

            #Splitting on a sub tree
            else:
                splits =

    def lower_bound(self,rewards):
        """
        Method to calculate Hoeffding lower bound μ
        """
        return np.mean(rewards) - np.sqrt(-np.log(self._confidence_interval)/(2*len(rewards)))

    def calculate_splitting_condition(self,p1,p2,rewards1,rewards2,rewards_old):
        valid = False
        if p1*self.lower_bound(rewards1) + p2*self.lower_bound(rewards2) > self.lower_bound(rewards_old):
            valid = True
        return valid

    def fit_learner(self,reward):