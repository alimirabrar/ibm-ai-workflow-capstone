import os
import unittest

# Run all unit tests from the unittests directory
if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Load all test modules
    test_dir = os.path.join(os.path.dirname(__file__), 'unittests')
    suite = loader.discover(test_dir, pattern='*Tests.py')

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print('\nAll tests passed!')
    else:
        print('\nSome tests failed.')
