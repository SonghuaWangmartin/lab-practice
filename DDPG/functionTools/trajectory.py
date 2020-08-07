
class SampleTrajectory:
    def __init__(self, maxRunningSteps, transit, isTerminal, rewardFunc, reset):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.rewardFunc = rewardFunc
        self.reset = reset

    def __call__(self, getaction):
        state = self.reset()
        while self.isTerminal(state):
            state = self.reset()

        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                break
            action = getaction(state)
            nextState = self.transit(state, action)
            reward = self.rewardFunc(self.isTerminal(state))
            trajectory.append((state, action, reward, nextState))

            state = nextState

        return trajectory

