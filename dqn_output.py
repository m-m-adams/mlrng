import numpy as np
import nistrng
import numpy
import random
import matplotlib.pyplot as plt
from sympy import python

dqn_out = np.loadtxt("dqn_output.txt", dtype=np.uint8, delimiter=",")
xp = np.packbits(dqn_out)
numpy_out = np.random.randint(0, 1, size=100_000)
np_bits = nistrng.pack_sequence(numpy_out)
python_rand = np.array([random.randint(0, 1) for _ in range(100_000)])
py_bits = nistrng.pack_sequence(python_rand)
plt.xlabel("Output value")
plt.ylabel("Frequency")
plt.title('DQN Output frequencies as 8 bit integers')
plt.hist(xp, bins=64, density=True)
plt.savefig("DQNOutputDistribution.png")


def run_tests(seq: np.ndarray):
    eligible_battery: dict = nistrng.check_eligibility_all_battery(
        seq, nistrng.SP800_22R1A_BATTERY)

    for name in eligible_battery.keys():
        print("-" + name)

    results = nistrng.run_all_battery(seq, eligible_battery, False)
    # Print results one by one
    print("Test results:")
    for result, elapsed_time in results:
        if result.passed:
            print(f"\item PASSED {result.name}")
        else:
            print(f"\item FAILED {result.name}")


run_tests(dqn_out)
# Test results:
# - FAILED - score: 0.0 - Monobit - elapsed time: 0 ms
# - PASSED - score: 0.987 - Frequency Within Block - elapsed time: 1 ms
# - FAILED - score: 0.0 - Longest Run Ones In A Block - elapsed time: 6 ms
# - PASSED - score: 0.678 - Binary Matrix Rank - elapsed time: 203 ms
# - FAILED - score: 0.0 - Discrete Fourier Transform - elapsed time: 3 ms
# - PASSED - score: 0.346 - Non Overlapping Template Matching - elapsed time: 100 ms
# - FAILED - score: 0.0 - Serial - elapsed time: 1388 ms
# - FAILED - score: 0.0 - Approximate Entropy - elapsed time: 1651 ms
# - PASSED - score: 1.0 - Cumulative Sums - elapsed time: 40 ms
# - PASSED - score: 0.971 - Random Excursion - elapsed time: 109 ms
# - FAILED - score: 0.0 - Random Excursion Variant - elapsed time: 0 ms
