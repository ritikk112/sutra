from sutra.core.extractor.base import FunctionSymbol, ClassSymbol, Location, Visibility


def _loc():
    return Location(line_start=1, line_end=2, byte_start=0, byte_end=10)


def test_symbols_default_to_non_local():
    fn = FunctionSymbol(
        id="sutra python r f.py outer().", name="outer", qualified_name="f.outer",
        file_path="f.py", location=_loc(), body_hash="sha256:x", language="python",
        visibility=Visibility.PUBLIC, is_exported=True,
    )
    assert fn.is_local is False
    assert fn.enclosing_moniker is None


def test_local_symbol_carries_enclosing_moniker():
    cls = ClassSymbol(
        id="sutra python r f.py outer().NoCause#", name="NoCause",
        qualified_name="f.outer.NoCause", file_path="f.py", location=_loc(),
        body_hash="sha256:x", language="python", visibility=Visibility.PUBLIC,
        is_exported=True, is_local=True,
        enclosing_moniker="sutra python r f.py outer().",
    )
    assert cls.is_local is True
    assert cls.enclosing_moniker == "sutra python r f.py outer()."
