from megasena import calculate_frequency, generate_games, _max_consecutive_run, _is_valid_game

def test_calculate_frequency_returns_60():
    draws = [
        [1, 2, 3, 4, 5, 6],
        [10, 20, 30, 40, 50, 60],
        [1, 20, 30, 40, 50, 60],
    ]
    freq = calculate_frequency(draws)
    assert isinstance(freq, dict)
    assert len(freq) == 60
    assert sum(freq.values()) == 18
    assert freq[1] == 2
    assert freq[60] == 2


def test_generate_games_structure_uniqueness_and_rules():
    freq = {i: (60 - i) for i in range(1, 61)}
    history = {tuple(sorted([1, 2, 3, 4, 5, 6]))}

    games = generate_games(
        freq,
        history,
        n_games=3,
        seed=42,
        mix=(3,2,1),
        max_run=2,
        target_even=3,
        tolerance=0,
        max_same_ending=2,
    )

    assert isinstance(games, list)
    assert len(games) == 3
    seen = set()
    for g in games:
        assert len(g) == 6
        assert len(set(g)) == 6
        assert all(1 <= n <= 60 for n in g)
        assert tuple(sorted(g)) not in history
        assert _max_consecutive_run(g) <= 2
        even = sum(1 for x in g if x % 2 == 0)
        assert even == 3
        # Mesmo final: no máximo 2
        endings = [n % 10 for n in g]
        assert max(endings.count(d) for d in range(10)) <= 2
        seen.add(tuple(g))
    assert len(seen) == 3


def test_same_ending_filter():
    # Mais de 2 com mesmo final (0): inválido
    game_bad = [10, 20, 30, 31, 42, 53]
    assert not _is_valid_game(game_bad, history_set=set(), max_same_ending=2)
    # No máximo 2 com mesmo final: válido
    game_ok = [10, 21, 32, 43, 54, 5]
    assert _is_valid_game(game_ok, history_set=set(), max_same_ending=2)
