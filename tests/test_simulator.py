from loop.schema import ALL_ARMS
from loop.simulate.campaigns import compute_historical_arm_propensities
from loop.simulate.world import generate_world


def test_simulator_seed_determinism() -> None:
    world_a = generate_world(seed=42, mc_leads=2000)
    world_b = generate_world(seed=42, mc_leads=2000)

    assert len(world_a.customers) == 40
    assert len(world_b.customers) == 40

    for cid in world_a.customers:
        ca = world_a.customers[cid]
        cb = world_b.customers[cid]
        assert ca.alpha_c == cb.alpha_c
        assert ca.meeting_logit == cb.meeting_logit
        assert ca.n_sends == cb.n_sends
        for sig in ca.beta_cs:
            assert ca.beta_cs[sig] == cb.beta_cs[sig]

def test_propensities_positive_and_sum_to_one() -> None:
    world = generate_world(seed=10, mc_leads=1000)
    for profile in world.customers.values():
        props = compute_historical_arm_propensities(profile)
        assert len(props) == 27
        total = 0.0
        for arm in ALL_ARMS:
            val = props[arm]
            assert val > 0.0
            total += val
        assert abs(total - 1.0) < 1e-6

def test_ground_truth_arm_values_reproducible() -> None:
    world1 = generate_world(seed=100, mc_leads=2000)
    world2 = generate_world(seed=100, mc_leads=2000)

    for cid in world1.customers:
        c1 = world1.customers[cid]
        c2 = world2.customers[cid]
        assert len(c1.ground_truth_arm_values) == 27
        for arm in ALL_ARMS:
            val1 = c1.ground_truth_arm_values[arm]
            val2 = c2.ground_truth_arm_values[arm]
            assert 0.0 <= val1 <= 1.0
            assert val1 == val2
