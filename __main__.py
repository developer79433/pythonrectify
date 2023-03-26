import sys
from .sudokuimageprocessor.src.processsudoku import  processSudoku as pr

def main(args=None):
	if args is None:
		args = sys.argv[1:]
	pr.main()
	
if __name__ == "__main__":
	sys.exit(main())
