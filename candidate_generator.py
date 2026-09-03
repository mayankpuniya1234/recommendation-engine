class CandidateGenerator:
    """
    Generates recommendation candidates using
    collaborative, content-based, popularity,
    and hybrid strategies.
    """

    def __init__(
        self,
        user_history,
        user_similarities,
        item_similarity,
        popularity,
        limit=20
    ):

        self.user_history = user_history
        self.user_similarities = user_similarities
        self.item_similarity = item_similarity
        self.popularity = popularity

        # Maximum candidates to return
        self.limit = limit

    # --------------------------------------------------
    # Collaborative Filtering
    # --------------------------------------------------

    def collaborative_candidates(self, user_id):
        """
        Recommend items liked by users similar to the
        given user.
        """

        # Cold-start user
        if user_id not in self.user_history:
            return self.popularity_candidates()

        user_items = set(
            self.user_history[user_id]
        )

        candidates = []

        similar_users = self.user_similarities.get(
            user_id,
            {}
        )

        # Most similar users first
        sorted_users = sorted(
            similar_users.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for similar_user, similarity in sorted_users:

            items = self.user_history.get(
                similar_user,
                []
            )

            for item in items:

                # Don't recommend already seen items
                if item not in user_items and item not in candidates:
                    candidates.append(item)

                if len(candidates) >= self.limit:
                    return candidates

        return candidates

    # --------------------------------------------------
    # Content Based
    # --------------------------------------------------

    def content_based_candidates(self, user_id):
        """
        Recommend items similar to items the user
        has already interacted with.
        """

        # Cold-start user
        if user_id not in self.user_history:
            return self.popularity_candidates()

        history = self.user_history[user_id]

        candidates = []

        for item in history:

            similar_items = self.item_similarity.get(
                item,
                []
            )

            for similar_item in similar_items:

                # Don't recommend already seen items
                if (
                    similar_item not in history
                    and similar_item not in candidates
                ):
                    candidates.append(similar_item)

                if len(candidates) >= self.limit:
                    return candidates

        return candidates

    # --------------------------------------------------
    # Popularity Based
    # --------------------------------------------------

    def popularity_candidates(self):
        """
        Return the most popular items.
        """

        sorted_items = sorted(
            self.popularity.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            item
            for item, score in sorted_items[:self.limit]
        ]

    # --------------------------------------------------
    # Hybrid
    # --------------------------------------------------

    def hybrid_candidates(self, user_id):
        """
        Combine collaborative filtering,
        content-based filtering, and popularity.
        """

        collaborative = self.collaborative_candidates(
            user_id
        )

        content_based = self.content_based_candidates(
            user_id
        )

        popularity = self.popularity_candidates()

        candidates = []

        # Give priority to personalized strategies
        combined = (
            collaborative
            + content_based
            + popularity
        )

        for item in combined:

            if item not in candidates:
                candidates.append(item)

            if len(candidates) >= self.limit:
                break

        return candidates