import random  # noqa

from .runners import algos

# seed = 118359435
seed = random.randrange(0, 2 ** 32 - 1)
print(f"Using seed: {seed}")
