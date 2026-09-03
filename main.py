from similarity import SimilarityCalculator
from candidate_generator import CandidateGenerator
from scorer import RecommendationScorer
from evaluator import RecommendationEvaluator


# ============================================================
#                 1. SIMILARITY CALCULATOR
# ============================================================

print("\n" + "=" * 60)
print("1. SIMILARITY CALCULATOR")
print("=" * 60)


# Cosine Similarity
cosine = SimilarityCalculator.cosine_similarity(
    [1, 2, 3],
    [1, 2, 3]
)

print(
    "Cosine Similarity:",
    round(cosine, 3)
)


# Jaccard Similarity
jaccard = SimilarityCalculator.jaccard_similarity(
    {"Python", "SQL", "ML"},
    {"Python", "SQL", "Java"}
)

print(
    "Jaccard Similarity:",
    round(jaccard, 3)
)


# Pearson Correlation
pearson = SimilarityCalculator.pearson_correlation(
    [1, 2, 3, 4, 5],
    [2, 4, 6, 8, 10]
)

print(
    "Pearson Correlation:",
    round(pearson, 3)
)


# ============================================================
#                 2. CANDIDATE GENERATOR
# ============================================================

print("\n" + "=" * 60)
print("2. CANDIDATE GENERATOR")
print("=" * 60)


# User's interaction history
user_history = {

    "user1": [
        "item1",
        "item2"
    ],

    "user2": [
        "item1",
        "item3",
        "item4"
    ],

    "user3": [
        "item2",
        "item4",
        "item5"
    ]
}


# Similar users
user_similarities = {

    "user1": {
        "user2": 0.90,
        "user3": 0.70
    },

    "user2": {
        "user1": 0.90
    },

    "user3": {
        "user1": 0.70
    }
}


# Similar items
item_similarity = {

    "item1": [
        "item3",
        "item4"
    ],

    "item2": [
        "item5",
        "item6"
    ],

    "item3": [
        "item1",
        "item4"
    ],

    "item4": [
        "item3",
        "item5"
    ],

    "item5": [
        "item2",
        "item6"
    ]
}


# Popularity score
popularity = {

    "item1": 1.00,
    "item2": 0.90,
    "item3": 0.80,
    "item4": 0.70,
    "item5": 0.60,
    "item6": 0.50
}


# Create Candidate Generator
generator = CandidateGenerator(
    user_history=user_history,
    user_similarities=user_similarities,
    item_similarity=item_similarity,
    popularity=popularity,
    limit=20
)


# Collaborative
collaborative = generator.collaborative_candidates(
    "user1"
)

print(
    "\nCollaborative Candidates:",
    collaborative
)


# Content Based
content = generator.content_based_candidates(
    "user1"
)

print(
    "Content-Based Candidates:",
    content
)


# Popularity
popular = generator.popularity_candidates()

print(
    "Popularity Candidates:",
    popular
)


# Hybrid
hybrid = generator.hybrid_candidates(
    "user1"
)

print(
    "Hybrid Candidates:",
    hybrid
)


# ============================================================
#                 3. SCORER & RANKER
# ============================================================

print("\n" + "=" * 60)
print("3. SCORER & RANKER")
print("=" * 60)


scorer = RecommendationScorer()


# ------------------------------------------------------------
# Relevance Scorer
# ------------------------------------------------------------

def relevance_score(
    user_id,
    item_id,
    context
):

    relevance = context.get(
        "relevance",
        {}
    )

    return relevance.get(
        item_id,
        0.0
    )


# ------------------------------------------------------------
# Popularity Scorer
# ------------------------------------------------------------

def popularity_score(
    user_id,
    item_id,
    context
):

    popularity_data = context.get(
        "popularity",
        {}
    )

    return popularity_data.get(
        item_id,
        0.0
    )


# ------------------------------------------------------------
# Recency Scorer
# ------------------------------------------------------------

def recency_score(
    user_id,
    item_id,
    context
):

    recency = context.get(
        "recency",
        {}
    )

    return recency.get(
        item_id,
        0.0
    )


# Register scoring functions
scorer.add_scorer(
    name="relevance",
    function=relevance_score,
    weight=0.50
)

scorer.add_scorer(
    name="popularity",
    function=popularity_score,
    weight=0.30
)

scorer.add_scorer(
    name="recency",
    function=recency_score,
    weight=0.20
)


# Context used by scorers
context = {

    "relevance": {

        "item3": 0.90,
        "item4": 0.80,
        "item5": 0.70,
        "item6": 0.50
    },

    "popularity": {

        "item1": 1.00,
        "item2": 0.90,
        "item3": 0.80,
        "item4": 0.70,
        "item5": 0.60,
        "item6": 0.50
    },

    "recency": {

        "item3": 0.80,
        "item4": 0.60,
        "item5": 0.90,
        "item6": 0.40
    }
}


# Rank hybrid candidates
ranked = scorer.rank_candidates(
    user_id="user1",
    candidates=hybrid,
    limit=5,
    context=context
)


print("\nFinal Recommendations:")

for recommendation in ranked:

    print(
        f"\nItem: {recommendation['item_id']}"
    )

    print(
        f"Score: {recommendation['score']}"
    )

    print(
        "Explanation:",
        ", ".join(
            recommendation["explanation"]
        )
    )


# ============================================================
#                 4. EVALUATOR
# ============================================================

print("\n" + "=" * 60)
print("4. RECOMMENDATION EVALUATOR")
print("=" * 60)


# Recommendations generated by system
recommendations = {

    "user1": [
        "item3",
        "item5",
        "item4",
        "item6"
    ],

    "user2": [
        "item5",
        "item6",
        "item2"
    ]
}


# Items that users actually liked
ground_truth = {

    "user1": [
        "item3",
        "item5"
    ],

    "user2": [
        "item5"
    ]
}


# Create evaluator
evaluator = RecommendationEvaluator()


# Evaluate
metrics = evaluator.evaluate_all(
    recommendations_dict=recommendations,
    ground_truth_dict=ground_truth,
    k=3
)


print("\nEvaluation Results:")

for metric, value in metrics.items():

    print(
        f"{metric}: {value:.3f}"
    )


# ============================================================
#                 INDIVIDUAL METRIC TEST
# ============================================================

print("\n" + "=" * 60)
print("INDIVIDUAL METRIC TEST")
print("=" * 60)


test_recommendations = [
    "item3",
    "item5",
    "item8"
]

test_relevant = [
    "item3",
    "item5"
]


precision = evaluator.precision_at_k(
    test_recommendations,
    test_relevant,
    3
)

recall = evaluator.recall_at_k(
    test_recommendations,
    test_relevant,
    3
)

ndcg = evaluator.ndcg_at_k(
    test_recommendations,
    test_relevant,
    3
)


print(
    "Precision@3:",
    round(precision, 3)
)

print(
    "Recall@3:",
    round(recall, 3)
)

print(
    "NDCG@3:",
    round(ndcg, 3)
)


print("\n" + "=" * 60)
print("RECOMMENDATION ENGINE DAY 1 COMPLETE")
print("=" * 60)