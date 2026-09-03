import math


class SimilarityCalculator:
    """
    Calculates similarity between users, items, or skills.
    """

    @staticmethod
    def cosine_similarity(vec1, vec2):
        """
        Calculate cosine similarity between two vectors.

        Returns:
            float: Value between 0 and 1.
        """

        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same length.")

        if not vec1:
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))

        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        # Zero-vector edge case
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        similarity = dot_product / (magnitude1 * magnitude2)

        # Keep result between 0 and 1
        return max(0.0, min(1.0, similarity))

    @staticmethod
    def jaccard_similarity(set1, set2):
        """
        Calculate Jaccard similarity between two sets.

        Formula:
            Intersection / Union

        Returns:
            float: Value between 0 and 1.
        """

        set1 = set(set1)
        set2 = set(set2)

        # Both sets are empty
        if not set1 and not set2:
            return 1.0

        # One set is empty
        if not set1 or not set2:
            return 0.0

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union

    @staticmethod
    def pearson_correlation(ratings1, ratings2):
        """
        Calculate Pearson correlation between two rating lists.

        Returns:
            float: Value between -1 and 1.
        """

        if len(ratings1) != len(ratings2):
            raise ValueError(
                "Rating lists must have the same length."
            )

        if len(ratings1) < 2:
            return 0.0

        mean1 = sum(ratings1) / len(ratings1)
        mean2 = sum(ratings2) / len(ratings2)

        numerator = sum(
            (a - mean1) * (b - mean2)
            for a, b in zip(ratings1, ratings2)
        )

        denominator1 = math.sqrt(
            sum((a - mean1) ** 2 for a in ratings1)
        )

        denominator2 = math.sqrt(
            sum((b - mean2) ** 2 for b in ratings2)
        )

        # Constant ratings edge case
        if denominator1 == 0 or denominator2 == 0:
            return 0.0

        correlation = numerator / (denominator1 * denominator2)

        # Pearson naturally ranges from -1 to 1
        return max(-1.0, min(1.0, correlation))


# --------------------------------------------------
# Simple Tests
# --------------------------------------------------

if __name__ == "__main__":

    print("=== Similarity Calculator Tests ===")

    # Cosine
    cosine = SimilarityCalculator.cosine_similarity(
        [1, 2, 3],
        [1, 2, 3]
    )

    print("Cosine Similarity:", round(cosine, 3))

    # Jaccard
    jaccard = SimilarityCalculator.jaccard_similarity(
        {"Python", "SQL", "ML"},
        {"Python", "SQL", "Java"}
    )

    print("Jaccard Similarity:", round(jaccard, 3))

    # Pearson
    pearson = SimilarityCalculator.pearson_correlation(
        [1, 2, 3, 4, 5],
        [2, 4, 6, 8, 10]
    )

    print("Pearson Correlation:", round(pearson, 3))