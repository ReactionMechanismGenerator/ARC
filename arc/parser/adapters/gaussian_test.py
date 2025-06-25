from pathlib import Path
import pytest

import numpy as np

from arc.parser.adapters.gaussian import GaussianParser

TEST_ROOT = Path(__file__).resolve().parents[3] / "testing"


@pytest.fixture
def gaussian_parser(request: pytest.FixtureRequest):
    """Return a GaussianParser for the file name passed via param."""
    return GaussianParser(TEST_ROOT / request.param)

# ───────────────────── 1.  logfile_contains_errors  ─────────────────────────
@pytest.mark.parametrize(
    "parser, expected",
    [
        ("trsh/gaussian/l9999.out", "Unconverged"),
        ("trsh/gaussian/l301.out", "No data on chk file."),
        ("trsh/gaussian/401.out", "Basis set data is not on the checkpoint file."),
        ("trsh/gaussian/l913.out", "Maximum optimization cycles reached"),
    ],
    indirect=["gaussian_parser"]
)

def test_logfile_contains_errors(gaussian_parser: GaussianParser, expected: str):
    assert gaussian_parser.logfile_contains_errors() == expected
    
# ───────────────────── 2.  parse_geometry  ──────────────────────────────────
@pytest.mark.parametrize(
    "gaussian_parser, expected_geom",
    [
        (
            "TS_confs/TS0_conf_0.out",
{
    "symbols": (
        "C", "C", "O", "O", "H", "H", "H", "H", "H",
        "C", "C", "H", "H", "H", "H", "H", "H"
    ),
    "isotopes": (
        12, 12, 16, 16, 1, 1, 1, 1, 1,
        12, 12, 1, 1, 1, 1, 1, 1
    ),
    "coords": (
        ( 1.503320,  1.312485, -0.238732),
        ( 1.861483, -0.055676,  0.309921),
        ( 0.739110, -0.883543,  0.555460),
        ( 0.152714, -1.235797, -0.640951),
        ( 0.816304,  1.842723,  0.437972),
        ( 1.027722,  1.220824, -1.225032),
        ( 2.412096,  1.921317, -0.354198),
        ( 2.339884,  0.025319,  1.297790),
        ( 2.556133, -0.582152, -0.366526),
        (-2.245250,  0.315565,  0.749505),
        (-1.862309,  0.279849, -0.699372),
        (-3.053586,  1.047616,  0.927615),
        (-2.598670, -0.665538,  1.097658),
        (-1.390650,  0.606240,  1.379516),
        (-0.764202, -0.549743, -0.738134),
        (-1.472702,  1.215127, -1.122232),
        (-2.550384, -0.240357, -1.378425)
    )
},
        ),
        (
            "xyz/Gaussian_large.log",
{
    "symbols": (
        "N", "C", "C", "C", "H", "H", "C", "C", "C",
        "C", "H", "H", "C", "C", "C", "H", "C", "C",
        "N", "H", "H", "C", "H", "C", "C", "C", "H",
        "H", "H", "H", "C", "C", "C", "H", "H", "H",
        "H", "H", "H", "H", "H", "H", "H", "H", "H",
        "O", "O", "O", "H", "H", "O", "H", "H"
    ),
    "isotopes": (
        14, 12, 12, 12, 1, 1, 12, 12, 12,
        12, 1, 1, 12, 12, 12, 1, 12, 12,
        14, 1, 1, 12, 1, 12, 12, 12, 1,
        1, 1, 1, 12, 12, 12, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        16, 16, 16, 1, 1, 16, 1, 1
    ),
    "coords": (
        ( 1.516992,  0.177437,  0.362189),
        ( 0.101445,  0.514651,  0.698592),
        ( 2.340942,  1.306326, -0.008051),
        (-0.386480, -0.399328,  1.835390),
        ( 0.054100,  1.576079,  1.030741),
        (-0.581295,  0.429478, -0.188162),
        ( 1.781531, -1.078028, -0.289812),
        ( 1.847002,  2.385298, -0.758344),
        ( 3.669932,  1.325100,  0.476167),
        (-1.903937, -0.275207,  2.043424),
        (-0.111715, -1.454765,  1.629507),
        ( 0.143518, -0.141467,  2.773200),
        ( 0.713350, -1.778949, -0.894403),
        ( 3.071101, -1.668626, -0.273753),
        ( 2.667791,  3.489611, -1.004626),
        ( 0.837169,  2.365773, -1.161280),
        ( 4.178810,  0.131822,  1.227589),
        ( 4.476106,  2.436698,  0.215068),
        (-2.635221, -0.910249,  0.901402),
        (-2.185366,  0.799717,  2.104553),
        (-2.177883, -0.723369,  3.023812),
        ( 0.898852, -3.040968, -1.451884),
        (-0.283232, -1.329730, -0.939923),
        ( 4.335820, -1.057964,  0.273571),
        ( 3.227696, -2.944035, -0.846303),
        ( 3.976108,  3.519370, -0.516697),
        ( 2.283543,  4.326646, -1.585544),
        ( 5.148117,  0.347077,  1.717562),
        ( 3.475175, -0.127735,  2.049325),
        ( 5.499943,  2.462490,  0.584855),
        (-3.984872, -0.299158,  0.658324),
        (-2.715146, -2.391685,  1.047208),
        ( 2.163239, -3.633048, -1.426274),
        ( 0.059751, -3.558716, -1.913851),
        ( 4.918222, -1.846375,  0.798803),
        ( 4.965848, -0.744565, -0.590277),
        ( 4.212970, -3.412405, -0.838059),
        ( 4.610547,  4.382016, -0.709191),
        (-4.825262, -0.844349,  1.136951),
        (-3.997015,  0.765249,  0.962313),
        (-3.286236, -2.811257,  0.201994),
        (-3.189421, -2.733518,  1.979469),
        (-1.697184, -2.814326,  1.003559),
        ( 2.318957, -4.618201, -1.858432),
        (-4.365403,  1.157031, -1.857381),
        (-4.233304, -0.385465, -0.729879),
        (-4.560403,  1.094303, -0.871587),
        (-6.546075,  0.505430, -0.691368),
        (-5.506168, -0.198147, -0.756314),
        (-6.712846,  0.643500, -1.635837),
        (-5.118540,  2.519666, -0.446020),
        (-4.788328,  2.527652,  0.474571),
        (-6.082056,  1.583154, -0.402379)
    )
},
        ),
    ],
    indirect=["gaussian_parser"],
)

def test_parse_geometry(gaussian_parser: GaussianParser, expected_geom: dict):
    geom = gaussian_parser.parse_geometry()

    # 1.  quick structural sanity
    assert geom is not None
    for key in ("symbols", "isotopes", "coords"):
        assert key in geom, f"missing {key} in parsed geometry"

    # 2.  count checks
    assert len(geom["symbols"])  == len(expected_geom["symbols"])
    assert len(geom["isotopes"]) == len(expected_geom["isotopes"])
    assert len(geom["coords"])   == len(expected_geom["coords"])

    # 3.  light-weight content check
    assert geom["symbols"][:3]   == expected_geom["symbols"][:3]        # first few symbols
    assert geom["isotopes"][:3]  == expected_geom["isotopes"][:3]
    np.testing.assert_allclose(
        np.array(geom["coords"][:3]), np.array(expected_geom["coords"][:3]), rtol=1e-3, atol=1e-4
    )
