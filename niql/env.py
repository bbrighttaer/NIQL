from marllib import marl


def make_mpe_env(**kwargs):
    # choose environment + scenario
    env = marl.make_env(
        environment_name="mpe",
        map_name="simple_spread",
        force_coop=True,
        max_cycles=25,
        **kwargs,
    )
    return env
