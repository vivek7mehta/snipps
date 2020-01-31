import argparse
from tests import test_train_test_split

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--tests', help="run test cases", action='store_true')
    args = parser.parse_args()

    if args.tests == False:
        parser.print_help()
        exit()
    else:
        test_train_test_split.run_test()

