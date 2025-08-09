class BinarySuccessReward:
    """
    A simple reward function that returns a binary success signal.
    This is used in the RIPT framework where only sparse rewards are available.
    """
    def compute_reward(self, idx, episode, batch) -> float:
        """
        Computes the reward for a given episode.

        Args:
            idx: The index of the episode in the batch.
            episode: A dictionary containing the trajectory data. It must
                     contain a 'success' key with a boolean value.
            batch: The original batch data (not used here).

        Returns:
            1.0 if the episode was successful, 0.0 otherwise.
        """
        return 1.0 if episode.get('success', False) else 0.0 