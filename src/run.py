import sys
import argparse

# Zadani:
# There is file in your repository src/run.py
# The first argument is always output .json file
# The second argument might be -v. It will start visual mode with debug images.
# The src/run.py accepts image filenames as arguments

# https://naucse.python.cz/2020/pyladies-ostrava-podzim-pokrocili/intro/argparse/
# https://docs.python.org/3/library/argparse.html

if __name__ == "__main__":
    for index, argument in enumerate(sys.argv):
        print(f"Argument číslo {index} je: {argument}")