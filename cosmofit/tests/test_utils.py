from cosmofit.utils import NamespaceDict

def test_misc():
    d = NamespaceDict(a=2)
    dict(d)


if __name__ == '__main__':

    test_misc()
