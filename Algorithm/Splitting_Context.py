import numpy as np
from Environments.Users import UserC1,UserC2,UserC3
from Environments.Context_environment import ContextEnvironment
from Learners.GPTS_Contextual import Learner

class Context():

    def __init__(self):
        self._possible_splits = [[0,0],[1,0],[0,1],[1,1]]
        self._current_split = [False,False]   #Chech whether split on condition is happened or not
        self._all_split = False  #boolean to check if all the possible splits have been done
        self_first_split = True
        self._confidence_interval = 1e-5

    def split_context(self,context,learner):
        """
        Method used to split the context in two parts
        """
        #Create a map of the features of the context, maximum 3 splits
        features_map = []
        for splits in context.keys():
            tmp = []
            for user in context[splits]:
                tmp.append(user.get_features)
            features_map.append(tmp)    #created a map of feature for each context

        if self._current_split[0] and self._current_split[1]:
            return False
        feature_subset_1 = []
        feature_subset_2 = []
        new_context = []
        for i in [0,1]:
            splitting_feature = i
            if not (self._current_split[splitting_feature]):
                if len(new_context)!=0:
                    features_map = new_context
                for subcontext in features_map:
                    for feature in subcontext:
                        if feature[splitting_feature] == 0:
                            feature_subset_1.append(feature)
                        else:
                            feature_subset_2.append(feature)


                #Created the feature subset to split on
                result = self.evaluate_splitting(learner,feature_subset_1,feature_subset_2)

                if result:
                    new_context = [feature_subset_1,feature_subset_2]
                else:
                    break

        final_context = {}
        for i,subcontext in enumerate(new_context):
            key = "Split" + str(i)
            final_context[key] = []
            for features in subcontext:
                user = self.assess_user_type(features[0],feature[1])
                final_context[key] = final_context[key].append(user)

        return final_context
    def evaluate_splitting(self,learner: Learner,feature1,feature2):
        """
        Method used to evaluate the splitting of the users
        """
        total_reward = learner.collected_rewards
        pulled_features = learner.features
        #Get the indices of the splitting features
        indices_f1 = [index for index, value in enumerate(pulled_features) if any(value == feature for feature in feature1)]
        indices_f2 = [index for index, value in enumerate(pulled_features) if any(value == feature for feature in feature2)]

        reward_f1 = total_reward[indices_f1]
        reward_f2 = total_reward[indices_f2]

        prob_1 = len(reward_f1)/len(total_reward)
        prob_2 = len(reward_f2)/len(total_reward)

        return self.calculate_splitting_condition(prob_1,prob_2,reward_f1,reward_f2,total_reward)


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

    def assess_user_type(self,f1_value,f2_value):
        """
        Method used to assess the type of user and return the right means depending on the relative demand curve
        """
        if f1_value:
            if f2_value:
                return UserC1()
            else:
                return UserC2()
        else:
            return UserC3()