import unittest
import os
import importlib
# Ensure that the config module can be found if tests are run from the root directory
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Store original environment variable
original_use_testnet_env = os.environ.get('USE_TESTNET')

class TestConfig(unittest.TestCase):

    def setUp(self):
        # Reload config before each test to pick up env changes
        # Need to ensure 'config' is in sys.modules to reload
        if 'config' in sys.modules:
            del sys.modules['config']
        global config
        import config

    def tearDown(self):
        # Restore original environment variable after each test
        if original_use_testnet_env is not None:
            os.environ['USE_TESTNET'] = original_use_testnet_env
        elif 'USE_TESTNET' in os.environ:
            del os.environ['USE_TESTNET']
        
        # Reload config after restoring env to avoid affecting other tests or subsequent runs
        if 'config' in sys.modules:
            del sys.modules['config']
        import config


    def test_use_testnet_env_not_set(self):
        if 'USE_TESTNET' in os.environ:
            del os.environ['USE_TESTNET']
        importlib.reload(config)
        self.assertTrue(config.USE_TESTNET, "Should be True if USE_TESTNET env is not set")

    def test_use_testnet_env_false_lowercase(self):
        os.environ['USE_TESTNET'] = 'false'
        importlib.reload(config)
        self.assertFalse(config.USE_TESTNET, "Should be False if USE_TESTNET env is 'false'")

    def test_use_testnet_env_false_uppercase(self):
        os.environ['USE_TESTNET'] = 'FALSE'
        importlib.reload(config)
        self.assertFalse(config.USE_TESTNET, "Should be False if USE_TESTNET env is 'FALSE'")
        
    def test_use_testnet_env_false_mixed_case(self):
        os.environ['USE_TESTNET'] = 'False'
        importlib.reload(config)
        self.assertFalse(config.USE_TESTNET, "Should be False if USE_TESTNET env is 'False'")

    def test_use_testnet_env_true_lowercase(self):
        os.environ['USE_TESTNET'] = 'true'
        importlib.reload(config)
        self.assertTrue(config.USE_TESTNET, "Should be True if USE_TESTNET env is 'true'")

    def test_use_testnet_env_true_uppercase(self):
        os.environ['USE_TESTNET'] = 'TRUE'
        importlib.reload(config)
        self.assertTrue(config.USE_TESTNET, "Should be True if USE_TESTNET env is 'TRUE'")

    def test_use_testnet_env_empty_string(self):
        os.environ['USE_TESTNET'] = ''
        importlib.reload(config)
        self.assertTrue(config.USE_TESTNET, "Should be True if USE_TESTNET env is an empty string")
        
    def test_use_testnet_env_other_string(self):
        os.environ['USE_TESTNET'] = 'yes' # Any string other than 'false' (case-insensitive)
        importlib.reload(config)
        self.assertTrue(config.USE_TESTNET, "Should be True if USE_TESTNET env is any other string like 'yes'")

if __name__ == '__main__':
    unittest.main()
