from cosmofit.utils import OrderedSet, NamespaceDict

def test_misc():
    OrderedSet()
    assert len(OrderedSet(i for i in range(4))) == 4
    assert OrderedSet(1, 2) == OrderedSet({1, 2})
    assert len(OrderedSet({1, 2})) == 2
    d = NamespaceDict(a=2)
    dict(d)


if __name__ == '__main__':

    test_misc()
