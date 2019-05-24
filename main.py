import re
import math

from utils import *

if __name__ == "__main__":
    for file in os.listdir('.'):
        c = re.match(r'([a-zA-Z_]+)(0\.0+1)', file)
        if c:
            c_val = float(c.group(2))
            os.rename(file, f'{c.group(1)}{abs(int(math.log10(c_val)))}')
