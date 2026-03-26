from pathlib import Path
import matplotlib.pyplot as plt

base_dir = Path('C:/', 'SGL_DATA', 'joplin_20240208')
drift_test_file = base_dir / "drift_test.txt"

with open(drift_test_file, 'r') as f:
	drift_amounts = []
	decoder_accs = []
	for line in f:
		tokens = line.split(',')
		tokens = [ token.strip() for token in tokens ]
		drift_amounts.append(int(tokens[0]))
		decoder_accs.append(float(tokens[1]))
	
	plt.plot(drift_amounts, decoder_accs)
	plt.title("Decoder accuracy with incorrect drift")
	plt.xlabel("Drift amount (um)")
	plt.ylabel("Decoder accuracy (%)")
	plt.show()