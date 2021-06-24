# =============================================================================
# python time module
# Author : Mayank Rasu

# Please report bug/issues in the Q&A section
# =============================================================================

import time
import numpy as np

def fibonacci(n):
   """Recursive function to print nth Fibonacci number"""
   if n <= 1:
       return n
   else:
       return(fibonacci(n-1) + fibonacci(n-2))

def main():
    num = np.random.randint(1,25)
    print("%dth fibonacci number is : %d"%(num,fibonacci(num)))

# Continuous execution        
starttime=time.time()
timeout = time.time() + 60*2  # 60 seconds times 2 meaning the script will run for 2 minutes
while time.time() <= timeout:
    try:
        main()
        time.sleep(5 - ((time.time() - starttime) % 5.0)) # 5 second interval between each new iteration
    except KeyboardInterrupt:
        print('\n\nKeyboard exception received. Exiting.')
        exit()
        
