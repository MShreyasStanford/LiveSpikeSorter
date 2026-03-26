from pathlib import Path

def read_eventfile(filename):
	with open(filename, 'r') as f:
		return [ int(line) for line in f ]
	return None

def write_eventfile(filename, events, labels):
	with open(filename, 'w') as f:
		for pred in zip(events, labels):
			f.write(f"{pred[0]} {pred[1]}\n")

path = Path('C://', 'SGL_DATA', 'joplin_20240208', 'decoder_input')

# Read eventfile contents into array
events = read_eventfile(path / 'eventfile.txt')

# Add a shifted copy of the events to act as pre-stimulus events
events_shifted = [ event - 100 * 30 for event in events ]
events.extend(events_shifted)
events = sorted(events)

labels = [ i % 2 for i in range(len(events)) ]

write_eventfile(path / 'eventfile_modified.txt', events, labels)
