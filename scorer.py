class RecommendationScorer:
    """
    Scores and ranks recommendation candidates
    using multiple weighted scoring functions.
    """

    def __init__(self):

        # Example:
        #
        # {
        #     "relevance": {
        #         "function": relevance_function,
        #         "weight": 0.5
        #     }
        # }

        self.scorers = {}

    # --------------------------------------------------
    # Add Scorer
    # --------------------------------------------------

    def add_scorer(self, name, function, weight):
        """
        Register a scoring function.

        Args:
            name: Name of scoring factor.
            function: Function that returns a score.
            weight: Importance of the scorer.
        """

        if weight < 0:
            raise ValueError(
                "Weight cannot be negative."
            )

        self.scorers[name] = {
            "function": function,
            "weight": weight
        }

    # --------------------------------------------------
    # Calculate Score
    # --------------------------------------------------

    def calculate_score(
        self,
        user_id,
        item_id,
        context
    ):
        """
        Calculate weighted score for one item.

        Returns:
            tuple:
                final_score,
                explanations
        """

        if not self.scorers:
            return 0.0, []

        total_score = 0.0
        total_weight = 0.0

        explanations = []

        for name, scorer in self.scorers.items():

            function = scorer["function"]
            weight = scorer["weight"]

            # Execute scoring function
            score = function(
                user_id,
                item_id,
                context
            )

            # Ensure score is between 0 and 1
            score = max(
                0.0,
                min(1.0, float(score))
            )

            total_score += score * weight
            total_weight += weight

            explanations.append(
                f"{name}: {score:.2f}"
            )

        # Prevent division by zero
        if total_weight == 0:
            return 0.0, explanations

        final_score = (
            total_score / total_weight
        )

        return final_score, explanations

    # --------------------------------------------------
    # Rank Candidates
    # --------------------------------------------------

    def rank_candidates(
        self,
        user_id,
        candidates,
        limit=10,
        context=None
    ):
        """
        Score all candidates and return top N items.
        """

        if context is None:
            context = {}

        ranked = []

        for item_id in candidates:

            score, explanation = self.calculate_score(
                user_id,
                item_id,
                context
            )

            ranked.append({
                "item_id": item_id,
                "score": round(score, 4),
                "explanation": explanation
            })

        # Highest score first
        ranked.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return ranked[:limit]