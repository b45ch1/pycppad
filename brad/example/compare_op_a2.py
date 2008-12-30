# Comparision operators with type a2double

def test_compare_op_a2():
	x = ad(ad(2.))
	y = ad(ad(3.))
	z = ad(ad(2.))
	
	# assert comparisions that should be true
	assert x == x
	assert x == z
	assert x != y
	assert x <= x
	assert x <= z
	assert x <= y
	assert x <  y
	
	# assert omparisions that should be false
	assert not x == y
	assert not x != z
	assert not x != x
	assert not x >= y
	assert not x >  y
