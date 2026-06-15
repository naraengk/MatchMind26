from wc26_predictor.api.main import load_context, monte_carlo_tournament


def test_monte_carlo_probabilities_are_consistent() -> None:
    context = load_context()
    result = monte_carlo_tournament(
        context["fixtures"],
        context["artifact"],
        simulations=200,
        seed=2026,
    )

    assert result["simulations"] == 200
    assert len(result["probabilities"]) == 48
    assert abs(sum(row["Champion"] for row in result["probabilities"]) - 100) < 0.2
    assert result["most_common_final"]["Probability"] > 0

    for row in result["probabilities"]:
        stages = [
            row["Round of 32"],
            row["Round of 16"],
            row["Quarterfinals"],
            row["Semifinals"],
            row["Final"],
            row["Champion"],
        ]
        assert stages == sorted(stages, reverse=True)
        assert abs(sum(row[f"Group Rank {rank}"] for rank in range(1, 5)) - 100) < 0.2
