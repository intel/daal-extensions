from pydaalcontrib.helpers import merge_kwargs
import unittest

class TestHelpers(unittest.TestCase):
	def test_merge_kwargs_no_overwrite(self):
		op_kwargs = ['key1', 'key2']
		inner_kwargs = {'key1': 1, 'key2': 2}
		outer_kwargs = {'key3': 3, 'key4': 4}

		kwargs = merge_kwargs(outer_kwargs, inner_kwargs, op_kwargs)

		self.assertEqual(kwargs, inner_kwargs)

	def test_merge_kwargs_no_overwrite2(self):
		op_kwargs = ['key1', 'key2']
		inner_kwargs = {'key1': 1, 'key2': 2}
		outer_kwargs = {'key1': 3, 'key2': 4}

		kwargs = merge_kwargs(outer_kwargs, inner_kwargs, op_kwargs)

		self.assertEqual(kwargs, inner_kwargs)

	def test_merge_kwargs_with_overwrite(self):
		op_kwargs = ['key1', 'key2']
		inner_kwargs = {'key0': 1, 'key9': 2}
		outer_kwargs = {'key1': 3, 'key2': 4}

		kwargs = merge_kwargs(outer_kwargs, inner_kwargs, op_kwargs)

		self.assertEqual(kwargs['key1'], 3)
		self.assertEqual(kwargs['key2'], 4)