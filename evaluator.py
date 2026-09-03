import math


class RecommendationEvaluator:
    """
    Evaluates recommendation quality using
    Precision@K, Recall@K, and NDCG@K.
    """

    # --------------------------------------------------
    # Precision@K
    # --------------------------------------------------

    @staticmethod
    def precision_at_k(
        recommendations,
        relevant_items,
        k
    ):
        """
        Precision@K:

        Number of relevant recommendations
        divided by number of recommendations.
        """

        if k <= 0:
            return 0.0

        top_k = recommendations[:k]

        if not top_k:
            return 0.0

        relevant_items = set(relevant_items)

        hits = sum(
            1
            for item in top_k
            if item in relevant_items
        )

        return hits / len(top_k)

    # --------------------------------------------------
    # Recall@K
    # --------------------------------------------------

    @staticmethod
    def recall_at_k(
        recommendations,
        relevant_items,
        k
    ):
        """
        Recall@K:

        Number of relevant items found in top-K
        divided by total relevant items.
        """

        relevant_items = set(relevant_items)

        if not relevant_items:
            return 0.0

        top_k = recommendations[:k]

        hits = sum(
            1
            for item in top_k
            if item in relevant_items
        )

        return hits / len(relevant_items)

    # --------------------------------------------------
    # NDCG@K
    # --------------------------------------------------

    @staticmethod
    def ndcg_at_k(
        recommendations,
        relevant_items,
        k
    ):
        """
        Calculate Normalized Discounted Cumulative Gain.

        Relevant items appearing earlier receive
        higher scores.
        """

        if k <= 0:
            return 0.0

        relevant_items = set(relevant_items)

        if not relevant_items:
            return 0.0

        top_k = recommendations[:k]

        dcg = 0.0

        for position, item in enumerate(top_k):

            if item in relevant_items:

                # Position is zero-based
                rank = position + 1

                dcg += (
                    1 / math.log2(rank + 1)
                )

        # Ideal ranking
        ideal_hits = min(
            len(relevant_items),
            k
        )

        idcg = 0.0

        for rank in range(
            1,
            ideal_hits + 1
        ):

            idcg += (
                1 / math.log2(rank + 1)
            )

        if idcg == 0:
            return 0.0

        return dcg / idcg

    # --------------------------------------------------
    # Evaluate All
    # --------------------------------------------------

    def evaluate_all(
        self,
        recommendations_dict,
        ground_truth_dict,
        k
    ):
        """
        Calculate average Precision@K,
        Recall@K, and NDCG@K across users.
        """

        precision_scores = []
        recall_scores = []
        ndcg_scores = []

        for user_id, recommendations in (
            recommendations_dict.items()
        ):

            # Missing ground truth
            relevant_items = ground_truth_dict.get(
                user_id,
                []
            )

            # Skip users without ground truth
            if not relevant_items:
                continue

            precision_scores.append(
                self.precision_at_k(
                    recommendations,
                    relevant_items,
                    k
                )
            )

            recall_scores.append(
                self.recall_at_k(
                    recommendations,
                    relevant_items,
                    k
                )
            )

            ndcg_scores.append(
                self.ndcg_at_k(
                    recommendations,
                    relevant_items,
                    k
                )
            )

        # No evaluable users
        if not precision_scores:

            return {
                "precision@k": 0.0,
                "recall@k": 0.0,
                "ndcg@k": 0.0
            }

        return {
            "precision@k": (
                sum(precision_scores)
                / len(precision_scores)
            ),

            "recall@k": (
                sum(recall_scores)
                / len(recall_scores)
            ),

            "ndcg@k": (
                sum(ndcg_scores)
                / len(ndcg_scores)
            )
        }