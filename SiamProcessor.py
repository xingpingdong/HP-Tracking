from rl.core import Processor

class SiamProcessor(Processor):
    def process_reward(self, reward):
        # The magnitude of the reward can be important. Since each step yields a relatively
        # high reward, we reduce the magnitude by two orders.
        return reward #/ 10.

    def process_observation(self, observation):
        return observation / 10
