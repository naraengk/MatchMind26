from wc26_predictor.features.elo import EloRatings, margin_multiplier, match_weight


def test_match_weight_orders_competitions_by_importance() -> None:
    assert match_weight("FIFA World Cup") > match_weight("FIFA World Cup qualification")
    assert match_weight("FIFA World Cup qualification") > match_weight("Friendly")
    assert match_weight("UEFA Euro") == 50.0
    assert match_weight("Some Regional Cup") == 30.0


def test_margin_multiplier_grows_with_goal_difference() -> None:
    assert margin_multiplier(0, 0) == 1.0
    assert margin_multiplier(1, 0) == 1.0
    assert margin_multiplier(3, 1) == 1.5
    assert margin_multiplier(3, 0) == 1.75
    assert margin_multiplier(5, 0) > margin_multiplier(4, 0)


def test_ratings_are_zero_sum_and_favor_the_winner() -> None:
    elo = EloRatings()
    elo.update("A", "B", 2, 0, neutral=True, tournament="Friendly")
    assert elo.get("A") > 1500 > elo.get("B")
    assert abs((elo.get("A") - 1500) + (elo.get("B") - 1500)) < 1e-9


def test_home_advantage_reduces_credit_for_home_wins() -> None:
    neutral = EloRatings()
    neutral.update("A", "B", 1, 0, neutral=True, tournament="Friendly")
    home = EloRatings()
    home.update("A", "B", 1, 0, neutral=False, tournament="Friendly")
    assert home.get("A") < neutral.get("A")


def test_big_competitive_wins_move_ratings_more_than_friendlies() -> None:
    friendly = EloRatings()
    friendly.update("A", "B", 1, 0, neutral=True, tournament="Friendly")
    world_cup = EloRatings()
    world_cup.update("A", "B", 4, 0, neutral=True, tournament="FIFA World Cup")
    assert world_cup.get("A") > friendly.get("A")
