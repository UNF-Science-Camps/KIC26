import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utilities'))
try:
    import sesy_vizualisation as _viz
except ImportError:
    _viz = None


def _eq(got, exp, tol=1e-9):
    """Float-tolerant lighed for enkeltværdi eller tuple."""
    if isinstance(exp, tuple):
        return len(got) == len(exp) and all(
            abs(float(g) - float(e)) < tol for g, e in zip(got, exp)
        )
    return abs(float(got) - float(exp)) < tol

def _fail(name, inp, got, exp):
    print(f"{name}{inp}  →  {got}  ≠  {exp}  ✗")

def _ok(name):
    print(f"Alle tests bestået for {name}  ✓")


# ── opg 1 ─────────────────────────────────────────────────────────────────

def testopg1_1(func):
    """(0,0) og (x,y) → linje a·x+b, dvs. a=y/x, b=0"""
    name = "opg1_1"
    cases = [
        ((2,  4),   (2.0,  0.0)),
        ((3,  9),   (3.0,  0.0)),
        ((5,  0),   (0.0,  0.0)),
        ((4, -8),   (-2.0, 0.0)),
        ((2,  1),   (0.5,  0.0)),
        ((1,  7),   (7.0,  0.0)),
    ]
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            x, y = inp
            if _viz: _viz.feedback_kurve([(0, 0), (x, y)], got, exp)
            return
    _ok(name)

def testopg1_2(func):
    """(x1,y1) og (x2,y2) → linje a·x+b"""
    name = "opg1_2"
    cases = [
        ((0,0,1,1),    (1.0,  0.0)),
        ((0,1,1,2),    (1.0,  1.0)),
        ((1,3,3,7),    (2.0,  1.0)),
        ((-1,-1,1,1),  (1.0,  0.0)),
        ((0,5,2,1),    (-2.0, 5.0)),
        ((1,2,4,2),    (0.0,  2.0)),
    ]
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            x1,y1,x2,y2 = inp
            if _viz: _viz.feedback_kurve([(x1,y1),(x2,y2)], got, exp)
            return
    _ok(name)

def testopg1_3(func):
    """(x1,y1),(x2,y2),(x3,y3) → parabel a·x²+b·x+c"""
    name = "opg1_3"
    cases = [
        ((1,1,2,4,3,9),     (1.0, 0.0,  0.0)),
        ((0,1,1,3,2,7),     (1.0, 1.0,  1.0)),
        ((0,3,1,4,2,9),     (2.0,-1.0,  3.0)),
        ((-1,3,0,4,2,0),    (-1.0,0.0,  4.0)),
        ((0,-1,2,5,4,15),   (0.5, 2.0, -1.0)),
        ((1,6,2,11,3,18),   (1.0, 2.0,  3.0)),
    ]
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: sæt 3 ligninger op — én per punkt: y = a·x²+b·x+c — og løs for a, b og c.")
            x1,y1,x2,y2,x3,y3 = inp
            if _viz: _viz.feedback_kurve([(x1,y1),(x2,y2),(x3,y3)], got, exp)
            return
    _ok(name)

# ── opg 2 ─────────────────────────────────────────────────────────────────

def testopg2_1(func):
    """(0,0) og punktet punkter[0] → linje a·x+b (punkt-udgave af opg1_1)"""
    name = "opg2_1"
    cases = [
        ([(2,  4)],   (2.0,  0.0)),
        ([(3,  9)],   (3.0,  0.0)),
        ([(5,  0)],   (0.0,  0.0)),
        ([(4, -8)],   (-2.0, 0.0)),
        ([(2,  1)],   (0.5,  0.0)),
        ([(1,  7)],   (7.0,  0.0)),
    ]
    for inp, exp in cases:
        got = func(inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            x, y = inp[0]
            if _viz: _viz.feedback_kurve([(0, 0), (x, y)], got, exp)
            return
    _ok(name)

def testopg2_2(func):
    """2 punkter → linje (kalder opg1_2 internt)"""
    name = "opg2_2"
    cases = [
        ([(0,0),(1,1)],    (1.0,  0.0)),
        ([(0,2),(2,6)],    (2.0,  2.0)),
        ([(1,5),(3,1)],    (-2.0, 7.0)),
        ([(-2,0),(0,4)],   (2.0,  4.0)),
        ([(0,0),(5,5)],    (1.0,  0.0)),
        ([(2,3),(4,3)],    (0.0,  3.0)),
    ]
    for inp, exp in cases:
        got = func(inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            if _viz: _viz.feedback_kurve(inp, got, exp)
            return
    _ok(name)

def testopg2_3(func):
    """3 punkter → parabel"""
    name = "opg2_3"
    cases = [
        ([(1,1),(2,4),(3,9)],     (1.0, 0.0,  0.0)),
        ([(0,1),(1,3),(2,7)],     (1.0, 1.0,  1.0)),
        ([(0,3),(1,4),(2,9)],     (2.0,-1.0,  3.0)),
        ([(-1,3),(0,4),(2,0)],    (-1.0,0.0,  4.0)),
        ([(0,-1),(2,5),(4,15)],   (0.5, 2.0, -1.0)),
        ([(1,6),(2,11),(3,18)],   (1.0, 2.0,  3.0)),
    ]
    for inp, exp in cases:
        got = func(inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: pak punkterne ud og kald opg1_3 — samme fremgangsmåde.")
            if _viz: _viz.feedback_kurve(inp, got, exp)
            return
    _ok(name)

# ── opg 3 ─────────────────────────────────────────────────────────────────

def testopg3_1(func):
    """SSE for præcis 3 punkter, linje a·x+b"""
    name = "opg3_1"
    cases = [
        ((1, 0, [(0,0),(1,1),(2,2)]),    0.0),
        ((1, 1, [(0,0),(1,1),(2,2)]),    3.0),
        ((2, 0, [(1,1),(2,2),(3,3)]),    14.0),
        ((0, 0, [(1,2),(2,4),(3,6)]),    56.0),
        ((0, 3, [(1,1),(2,2),(3,3)]),    5.0),
        ((1, 1, [(1,4),(2,6),(3,8)]),    29.0),
    ]
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            a, b, punkter = inp
            if _viz: _viz.feedback_loss((a, b), punkter, got, exp)
            return
    _ok(name)

def testopg3_2(func):
    """SSE for n punkter, linje a·x+b"""
    name = "opg3_2"
    cases = [
        ((2, 0, [(1,2),(2,4),(3,6)]),           0.0),
        ((0, 0, [(1,1),(2,2)]),                 5.0),
        ((1, 0, [(0,1),(1,1),(2,1)]),           2.0),
        ((2, 1, [(1,3),(2,5),(3,7),(4,9)]),     0.0),
        ((1, 0, [(0,0),(1,2)]),                 1.0),
        ((0, 5, [(0,5),(5,5),(10,5)]),          0.0),
    ]
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            a, b, punkter = inp
            if _viz: _viz.feedback_loss((a, b), punkter, got, exp)
            return
    _ok(name)

def testopg3_3(func):
    """SSE for n punkter, parabel a·x²+b·x+c"""
    name = "opg3_3"
    cases = [
        ((1, 0, 0, [(1,1),(2,4),(3,9)]),     0.0),
        ((0, 0, 0, [(1,1)]),                 1.0),
        ((1, 1, 1, [(0,1),(1,3),(2,7)]),     0.0),
        ((2,-1, 3, [(0,3),(1,4),(2,9)]),     0.0),
        ((1, 0, 0, [(0,1),(1,1),(2,4)]),     1.0),
        ((0, 2, 0, [(1,2),(2,4),(3,6)]),     0.0),
    ]
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            a, b, c, punkter = inp
            if _viz: _viz.feedback_loss((a, b, c), punkter, got, exp)
            return
    _ok(name)

def testopg3_4(func):
    """SSE for n punkter, plan a1·x+a2·y+b=z"""
    name = "opg3_4"
    cases = [
        ((1, 1, 0, [(1,1,2),(2,2,4),(0,0,0)]),    0.0),
        ((0, 0, 0, [(1,0,1),(0,1,1)]),             2.0),
        ((1, 2, 3, [(1,1,6),(2,0,5),(0,2,7)]),     0.0),
        ((2, 0, 1, [(1,5,3),(2,7,5)]),             0.0),
        ((1, 1, 1, [(0,0,0),(1,0,0),(0,1,0)]),     9.0),
        ((0, 0, 5, [(0,0,5),(1,2,5),(3,4,5)]),     0.0),
    ]
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: samme mønster som opg3_1-opg3_3 — for hvert punkt (x,y,z), kvadrer (a1·x+a2·y+b−z) og læg sammen.")
            return
    _ok(name)


# ── opg 4 (SSE til MSE) ───────────────────────────────────────────────────────

def testopg4_1(func):
    """MSE for n punkter, linje a·x+b (SSE delt med antal punkter)"""
    name = "opg4_1"
    cases = [
        ((2, 0, [(1,2),(2,4),(3,6)]),           0.0),
        ((0, 0, [(1,1),(2,2)]),                 5.0 / 2),
        ((1, 0, [(0,1),(1,1),(2,1)]),           2.0 / 3),
        ((2, 1, [(1,3),(2,5),(3,7),(4,9)]),     0.0),
        ((1, 0, [(0,0),(1,2)]),                 1.0 / 2),
        ((0, 5, [(0,5),(5,5),(10,5)]),          0.0),
    ]
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: brug jeres opg3_2 (SSE) og divider med len(punkter).")
            return
    _ok(name)

def testopg4_2(func):
    """MSE for n punkter, parabel a·x²+b·x+c (SSE delt med antal punkter)"""
    name = "opg4_2"
    cases = [
        ((1, 0, 0, [(1,1),(2,4),(3,9)]),     0.0),
        ((0, 0, 0, [(1,1)]),                 1.0 / 1),
        ((1, 1, 1, [(0,1),(1,3),(2,7)]),     0.0),
        ((2,-1, 3, [(0,3),(1,4),(2,9)]),     0.0),
        ((1, 0, 0, [(0,1),(1,1),(2,4)]),     1.0 / 3),
        ((0, 2, 0, [(1,2),(2,4),(3,6)]),     0.0),
    ]
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: brug jeres opg3_3 (SSE) og divider med len(punkter).")
            return
    _ok(name)

def testopg4_3(func):
    """MSE for n punkter, plan a1·x+a2·y+b=z (SSE delt med antal punkter)"""
    name = "opg4_3"
    cases = [
        ((1, 1, 0, [(1,1,2),(2,2,4),(0,0,0)]),    0.0),
        ((0, 0, 0, [(1,0,1),(0,1,1)]),             2.0 / 2),
        ((1, 2, 3, [(1,1,6),(2,0,5),(0,2,7)]),     0.0),
        ((2, 0, 1, [(1,5,3),(2,7,5)]),             0.0),
        ((1, 1, 1, [(0,0,0),(1,0,0),(0,1,0)]),     9.0 / 3),
        ((0, 0, 5, [(0,0,5),(1,2,5),(3,4,5)]),     0.0),
    ]
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: brug jeres opg3_4 (SSE) og divider med len(punkter).")
            return
    _ok(name)


# ── opg 5 (differentiering af x: potens-, sum- og kædedregel) ────────────────

def testopg5_1(func):
    """f(x) = 5x²  →  find f'(x)"""
    name = "opg5_1"
    f = lambda x: 5 * x**2
    cases = [
        (1,     10.0),
        (2,     20.0),
        (-1,   -10.0),
        (0,      0.0),
        (3,     30.0),
        (0.5,    5.0),
    ]
    for inp, exp in cases:
        got = func(inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: potensreglen — d/dx(c·xⁿ) = c·n·xⁿ⁻¹.")
            if _viz: _viz.feedback_haeldning(f, inp, got, exp)
            return
    _ok(name)

def testopg5_2(func):
    """f(x) = 5x⁶  →  find f'(x)"""
    name = "opg5_2"
    f = lambda x: 5 * x**6
    cases = [
        (1,       30.0),
        (2,      960.0),
        (-1,     -30.0),
        (0,        0.0),
        (0.5,      0.9375),
        (-2,    -960.0),
    ]
    for inp, exp in cases:
        got = func(inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: potensreglen — d/dx(c·xⁿ) = c·n·xⁿ⁻¹.")
            if _viz: _viz.feedback_haeldning(f, inp, got, exp)
            return
    _ok(name)

def testopg5_3(func):
    """f(x) = 3x  →  find f'(x)"""
    name = "opg5_3"
    f = lambda x: 3 * x
    cases = [
        (0,    3.0),
        (1,    3.0),
        (-1,   3.0),
        (5,    3.0),
        (-5,   3.0),
        (2.5,  3.0),
    ]
    for inp, exp in cases:
        got = func(inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: potensreglen — x er x¹, så f'(x) er bare tallet foran x.")
            if _viz: _viz.feedback_haeldning(f, inp, got, exp)
            return
    _ok(name)

def testopg5_4(func):
    """f(x) = 17  →  find f'(x)"""
    name = "opg5_4"
    f = lambda x: 17
    cases = [
        (0,      0.0),
        (1,      0.0),
        (-1,     0.0),
        (100,    0.0),
        (-100,   0.0),
        (0.5,    0.0),
    ]
    for inp, exp in cases:
        got = func(inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: en konstant funktion ændrer sig aldrig — hældningen er altid 0.")
            if _viz: _viz.feedback_haeldning(f, inp, got, exp)
            return
    _ok(name)

def testopg5_5(func):
    """f(x) = x² + 4x  →  find f'(x)"""
    name = "opg5_5"
    f = lambda x: x**2 + 4*x
    cases = [
        (0,     4.0),
        (1,     6.0),
        (-2,    0.0),
        (3,    10.0),
        (-4,   -4.0),
        (2,     8.0),
    ]
    for inp, exp in cases:
        got = func(inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: differentiér hvert led for sig (sumreglen), og læg sammen.")
            if _viz: _viz.feedback_haeldning(f, inp, got, exp)
            return
    _ok(name)

def testopg5_6(func):
    """f(x) = 2x² + 4x + 7  →  find f'(x)"""
    name = "opg5_6"
    f = lambda x: 2*x**2 + 4*x + 7
    cases = [
        (0,     4.0),
        (1,     8.0),
        (-1,    0.0),
        (2,    12.0),
        (-2,   -4.0),
        (0.5,   6.0),
    ]
    for inp, exp in cases:
        got = func(inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: differentiér hvert led for sig (sumreglen) — den konstante 7 forsvinder.")
            if _viz: _viz.feedback_haeldning(f, inp, got, exp)
            return
    _ok(name)

def testopg5_7(func):
    """f(x) = (2x+1)²  →  find f'(x)"""
    name = "opg5_7"
    f = lambda x: (2*x + 1)**2
    cases = [
        (0,      4.0),
        (1,     12.0),
        (-1,    -4.0),
        (2,     20.0),
        (0.5,    8.0),
        (-2,   -12.0),
    ]
    for inp, exp in cases:
        got = func(inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: kædedreglen — differentiér det ydre kvadrat (2·u), og gang med den afledte af det indre (u'=2).")
            if _viz: _viz.feedback_haeldning(f, inp, got, exp)
            return
    _ok(name)

def testopg5_8(func):
    """f(x) = (4x² + 3x)² + 7x + 4  →  find f'(x)"""
    name = "opg5_8"
    f = lambda x: (4*x**2 + 3*x)**2 + 7*x + 4
    cases = [
        (0,      7.0),
        (1,    161.0),
        (2,    843.0),
        (-1,    -3.0),
        (0.5,   42.0),
        (-0.5,   8.0),
    ]
    for inp, exp in cases:
        got = func(inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: kædedreglen på det ydre kvadrat (indre u = 4x²+3x, u' = 8x+3), plus sumreglen for +7x+4.")
            if _viz: _viz.feedback_haeldning(f, inp, got, exp)
            return
    _ok(name)


# ── opg 6 (partielt afledte og gradienter) ───────────────────────────────────

def testopg6_1(func):
    """f(x,y) = x² + y²  →  find gradienten (∂f/∂x, ∂f/∂y)"""
    name = "opg6_1"
    f = lambda x, y: x**2 + y**2
    cases = [
        ((1, 1),    (2.0,  2.0)),
        ((2, 1),    (4.0,  2.0)),
        ((0, 2),    (0.0,  4.0)),
        ((3, 0),    (6.0,  0.0)),
        ((-1, 2),   (-2.0, 4.0)),
        ((2, -1),   (4.0, -2.0)),
    ]
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: for ∂f/∂x, behandl y som en konstant. For ∂f/∂y, behandl x som en konstant.")
            if _viz: _viz.feedback_gradient_3d(f, inp[0], inp[1], got, exp)
            return
    _ok(name)

def testopg6_2(func):
    """f(x,y) = x²·y + y²  →  find gradienten (∂f/∂x, ∂f/∂y)"""
    name = "opg6_2"
    f = lambda x, y: x**2 * y + y**2
    cases = [
        ((1, 1),    (2.0,  3.0)),
        ((2, 1),    (4.0,  6.0)),
        ((0, 2),    (0.0,  4.0)),
        ((3, 0),    (0.0,  9.0)),
        ((-1, 2),   (-4.0, 5.0)),
        ((2, -1),   (-4.0, 2.0)),
    ]
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: for ∂f/∂x, behandl y som en konstant. For ∂f/∂y, behandl x som en konstant.")
            if _viz: _viz.feedback_gradient_3d(f, inp[0], inp[1], got, exp)
            return
    _ok(name)

def testopg6_3(func):
    """f(x,y) = x·y² + y  →  find gradienten (∂f/∂x, ∂f/∂y)"""
    name = "opg6_3"
    f = lambda x, y: x * y**2 + y
    cases = [
        ((1, 1),    (1.0,  3.0)),
        ((2, 1),    (1.0,  5.0)),
        ((0, 2),    (4.0,  1.0)),
        ((3, 0),    (0.0,  1.0)),
        ((-1, 2),   (4.0, -3.0)),
        ((2, -1),   (1.0, -3.0)),
    ]
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: for ∂f/∂x, behandl y som en konstant. For ∂f/∂y, behandl x som en konstant.")
            if _viz: _viz.feedback_gradient_3d(f, inp[0], inp[1], got, exp)
            return
    _ok(name)

def testopg6_4(func):
    """f(x,y) = (2x+y)²  →  find gradienten (∂f/∂x, ∂f/∂y)"""
    name = "opg6_4"
    f = lambda x, y: (2*x + y)**2
    cases = [
        ((0, 0),     (0.0,   0.0)),
        ((1, 0),     (8.0,   4.0)),
        ((0, 1),     (4.0,   2.0)),
        ((2, 1),    (20.0,  10.0)),
        ((3, -2),   (16.0,   8.0)),
        ((-2, 1),  (-12.0,  -6.0)),
    ]
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: kædedreglen — det indre er u=2x+y, med ∂u/∂x=2 og ∂u/∂y=1.")
            if _viz: _viz.feedback_gradient_3d(f, inp[0], inp[1], got, exp)
            return
    _ok(name)

def testopg6_5(func):
    """f(x,y) = (x²+y)² + 3x  →  find gradienten (∂f/∂x, ∂f/∂y)"""
    name = "opg6_5"
    f = lambda x, y: (x**2 + y)**2 + 3*x
    cases = [
        ((0, 0),     (3.0,   0.0)),
        ((1, 0),     (7.0,   2.0)),
        ((0, 1),     (3.0,   2.0)),
        ((1, 1),    (11.0,   4.0)),
        ((2, 0),    (35.0,   8.0)),
        ((-1, 2),   (-9.0,   6.0)),
    ]
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: kædedreglen på (x²+y)² (indre u=x²+y, ∂u/∂x=2x, ∂u/∂y=1), plus sumreglen for +3x.")
            if _viz: _viz.feedback_gradient_3d(f, inp[0], inp[1], got, exp)
            return
    _ok(name)

def testopg6_6(func):
    """f(x,y,z) = x·y + y·z + z²  →  find gradienten (∂f/∂x, ∂f/∂y, ∂f/∂z)"""
    name = "opg6_6"
    cases = [
        ((1, 1, 1),    (1.0, 2.0, 3.0)),
        ((2, 1, 0),    (1.0, 2.0, 1.0)),
        ((0, 2, 1),    (2.0, 1.0, 4.0)),
        ((3, 0, 2),    (0.0, 5.0, 4.0)),
        ((-1, 2, 1),   (2.0, 0.0, 4.0)),
        ((1, -1, -1),  (-1.0, 0.0, -3.0)),
    ]
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: for hver partielt afledte, behandl de to andre variable som konstanter.")
            return
    _ok(name)


# ── opg 7 (gradient af loss) ──────────────────────────────────────────────────

def testopg7_1(func):
    """Gradient af SSE mht. (a,b) for linjen y=a·x+b, ét enkelt punkt"""
    name = "opg7_1"
    cases = [
        ((0, 0, 1, 2),    (-4.0, -4.0)),
        ((1, 0, 1, 1),    (0.0,   0.0)),
        ((2, 1, 1, 3),    (0.0,   0.0)),
        ((1, 0, 2, 4),    (-8.0, -4.0)),
        ((0, 5, 0, 5),    (0.0,   0.0)),
        ((1, 1, 3, 2),    (12.0,  4.0)),
    ]
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: err = a·x+b−y. gradienten er (2·err·x, 2·err).")
            return
    _ok(name)

def testopg7_2(func):
    """Gradient af SSE mht. (a,b) for linjen y=a·x+b, præcis 2 punkter"""
    name = "opg7_2"
    cases = [
        ((0, 0, [(1,2),(2,1)]),      (-8.0,  -6.0)),
        ((1, 0, [(1,1),(2,4)]),      (-8.0,  -4.0)),
        ((2, 1, [(1,3),(0,1)]),      (0.0,    0.0)),
        ((0, 0, [(1,1),(1,1)]),      (-4.0,  -4.0)),
        ((1, 1, [(0,0),(2,2)]),      (4.0,    4.0)),
        ((-1, 2, [(3,0),(1,4)]),     (-12.0, -8.0)),
    ]
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: brug jeres opg7_1 på hvert af de 2 punkter, og læg bidragene sammen.")
            return
    _ok(name)

def testopg7_3(func):
    """Gradient af SSE mht. (a,b) for linjen y=a·x+b, n punkter"""
    name = "opg7_3"
    cases = [
        ((0, 0, [(1,2)]),                (-4.0, -4.0)),
        ((1, 0, [(1,1)]),                (0.0,   0.0)),
        ((0, 0, [(1,1),(2,2),(3,3)]),    (-28.0,-12.0)),
        ((1, 0, [(0,0),(1,2),(2,4)]),    (-10.0, -6.0)),
        ((2, 1, [(1,3)]),                (0.0,   0.0)),
        ((0, 5, [(0,5),(1,5),(2,5)]),    (0.0,   0.0)),
    ]
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: for hvert punkt, beregn err = a·x+b−y og summer 2·err·x (for a) og 2·err (for b).")
            return
    _ok(name)

def testopg7_4(func):
    """Gradient af SSE mht. (a,b,c) for parablen y=a·x²+b·x+c, ét enkelt punkt"""
    name = "opg7_4"
    cases = [
        ((0, 0, 0, 1, 2),    (-4.0,  -4.0,  -4.0)),
        ((0, 0, 0, 2, 1),    (-8.0,  -4.0,  -2.0)),
        ((1, 0, 0, 2, 1),    (24.0,   12.0,  6.0)),
        ((0, 1, 0, 2, 5),    (-24.0, -12.0, -6.0)),
        ((1, 1, 1, 1, 4),    (-2.0,   -2.0, -2.0)),
        ((1, 1, 1, 2, 9),    (-16.0,  -8.0, -4.0)),
    ]
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: err = a·x²+b·x+c−y. gradienten er (2·err·x², 2·err·x, 2·err).")
            return
    _ok(name)

def testopg7_5(func):
    """Gradient af SSE mht. (a,b,c) for parablen y=a·x²+b·x+c, præcis 2 punkter"""
    name = "opg7_5"
    cases = [
        ((0, 0, 0, [(1,2),(2,1)]),     (-12.0,  -8.0,  -6.0)),
        ((1, 0, 0, [(2,1),(1,1)]),     (24.0,   12.0,   6.0)),
        ((0, 1, 0, [(2,5),(1,1)]),     (-24.0, -12.0,  -6.0)),
        ((1, 1, 1, [(1,4),(2,9)]),     (-18.0, -10.0,  -6.0)),
        ((0, 0, 0, [(1,1),(1,1)]),     (-4.0,   -4.0,  -4.0)),
        ((2, -1, 3, [(0,3),(1,4)]),    (0.0,     0.0,   0.0)),
    ]
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: brug jeres opg7_4 på hvert af de 2 punkter, og læg bidragene sammen.")
            return
    _ok(name)

def testopg7_6(func):
    """Gradient af SSE mht. (a,b,c) for parablen y=a·x²+b·x+c, n punkter"""
    name = "opg7_6"
    cases = [
        ((0, 0, 0, [(1,2)]),               (-4.0,  -4.0,  -4.0)),
        ((0, 0, 0, [(2,1)]),               (-8.0,  -4.0,  -2.0)),
        ((0, 0, 0, [(1,1),(2,3)]),         (-26.0, -14.0, -8.0)),
        ((1, 0, 0, [(2,1)]),               (24.0,   12.0,  6.0)),
        ((0, 1, 0, [(2,5),(1,1)]),         (-24.0, -12.0, -6.0)),
        ((1, 1, 1, [(1,4),(2,9)]),         (-18.0, -10.0, -6.0)),
    ]
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: for hvert punkt, beregn err = a·x²+b·x+c−y og summer 2·err·x² (for a), 2·err·x (for b) og 2·err (for c).")
            return
    _ok(name)

def testopg7_7(func):
    """Gradient af SSE mht. (a1,a2,b) for planen a1·x+a2·y+b=z"""
    name = "opg7_7"
    cases = [
        ((0, 0, 0, [(1,1,2)]),                     (-4.0, -4.0, -4.0)),
        ((1, 0, 0, [(2,3,2)]),                     (0.0,   0.0,  0.0)),
        ((0, 0, 0, [(1,0,1),(0,1,1)]),              (-2.0, -2.0, -4.0)),
        ((1, 1, 0, [(1,1,2),(2,2,4),(0,0,0)]),      (0.0,   0.0,  0.0)),
        ((2, 1, 1, [(1,1,1)]),                      (6.0,   6.0,  6.0)),
        ((0, 1, -1, [(3,2,0),(1,1,1)]),             (4.0,   2.0,  0.0)),
    ]
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: for hvert punkt, beregn err = a1·x+a2·y+b−z og summer 2·err·x (for a1), 2·err·y (for a2) og 2·err (for b).")
            return
    _ok(name)


# ── opg 8 (gradient af MSE) ───────────────────────────────────────────────────

def testopg8_1(func):
    """Gradient af MSE mht. (a,b) for linjen y=a·x+b, n punkter (SSE-gradient delt med n)"""
    name = "opg8_1"
    cases = [
        ((0, 0, [(1,2)]),                (-4.0,      -4.0)),
        ((1, 0, [(1,1)]),                (0.0,        0.0)),
        ((0, 0, [(1,1),(2,2),(3,3)]),    (-28.0/3,   -4.0)),
        ((1, 0, [(0,0),(1,2),(2,4)]),    (-10.0/3,   -2.0)),
        ((2, 1, [(1,3)]),                (0.0,        0.0)),
        ((0, 5, [(0,5),(1,5),(2,5)]),    (0.0,        0.0)),
    ]
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: brug jeres opg7_3 (gradient af SSE) og divider hvert element med len(punkter).")
            return
    _ok(name)

def testopg8_2(func):
    """Gradient af MSE mht. (a,b,c) for parablen y=a·x²+b·x+c, n punkter (SSE-gradient delt med n)"""
    name = "opg8_2"
    cases = [
        ((0, 0, 0, [(1,2)]),               (-4.0,   -4.0,  -4.0)),
        ((0, 0, 0, [(2,1)]),               (-8.0,   -4.0,  -2.0)),
        ((0, 0, 0, [(1,1),(2,3)]),         (-13.0,  -7.0,  -4.0)),
        ((1, 0, 0, [(2,1)]),               (24.0,   12.0,   6.0)),
        ((0, 1, 0, [(2,5),(1,1)]),         (-12.0,  -6.0,  -3.0)),
        ((1, 1, 1, [(1,4),(2,9)]),         (-9.0,   -5.0,  -3.0)),
    ]
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: brug jeres opg7_6 (gradient af SSE) og divider hvert element med len(punkter).")
            return
    _ok(name)

def testopg8_3(func):
    """Gradient af MSE mht. (a1,a2,b) for planen a1·x+a2·y+b=z (SSE-gradient delt med n)"""
    name = "opg8_3"
    cases = [
        ((0, 0, 0, [(1,1,2)]),                     (-4.0, -4.0, -4.0)),
        ((1, 0, 0, [(2,3,2)]),                     (0.0,   0.0,  0.0)),
        ((0, 0, 0, [(1,0,1),(0,1,1)]),              (-1.0, -1.0, -2.0)),
        ((1, 1, 0, [(1,1,2),(2,2,4),(0,0,0)]),      (0.0,   0.0,  0.0)),
        ((2, 1, 1, [(1,1,1)]),                      (6.0,   6.0,  6.0)),
        ((0, 1, -1, [(3,2,0),(1,1,1)]),             (2.0,   1.0,  0.0)),
    ]
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: brug jeres opg7_7 (gradient af SSE) og divider hvert element med len(punkter).")
            return
    _ok(name)


# ── opg 9 (gradient descent) ─────────────────────────────────────────────────

def testopg9_1(func):
    """Vægtopdatering: a,b,a_delta,b_delta → (a−a_delta, b−b_delta)"""
    name = "opg9_1"
    cases = [
        ((1,   2,   0,   0),    (1.0,  2.0)),
        ((0,   0,   1,   1),    (-1.0,-1.0)),
        ((5,   3,   2,   1),    (3.0,  2.0)),
        ((1.5, 0.5, 0.3, 0.1),  (1.2,  0.4)),
        ((-2,  4,  -3,   2),    (1.0,  2.0)),
        ((10, -5,   5,  -3),    (5.0, -2.0)),
    ]
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: træk deltaerne fra: return a - a_delta, b - b_delta.")
            return
    _ok(name)

def testopg9_2(func):
    """Ét gradient descent-skridt for linjen y=a·x+b: a,b,punkter,lr → (a,b) efter skridtet"""
    name = "opg9_2"
    cases = [
        ((0, 0, [(1,2)], 1.0),                              (4.0, 4.0)),
        ((1, 0, [(1,1),(2,4)], 0.5),                         (3.0, 1.0)),
        ((0, 0, [(1,1),(2,2),(3,3)], 0.1),                   (0.9333333333333335, 0.4)),
        ((2, 1, [(1,3),(0,1)], 0.25),                        (2.0, 1.0)),
        ((0, 5, [(0,5),(1,5),(2,5)], 1.0),                   (0.0, 5.0)),
        ((-1, 2, [(3,0),(1,4)], 0.05),                       (-0.7, 2.2)),
    ]
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: brug jeres opg8_1 til at finde gradienten, gang med lr, og brug opg9_1 til at opdatere.")
            return
    _ok(name)

def testopg9_3(func):
    """Træningsloop for linjen: kør opg9_2 n gange i træk"""
    name = "opg9_3"
    punkter = [(1,2),(3,4),(4,3),(2.5,2.5)]
    cases = [
        ((0.0, 0.0, punkter, 0.05, 1),                       (0.80625, 0.28750000000000003)),
        ((0.0, 0.0, punkter, 0.05, 3),                       (0.890269775390625, 0.3558129882812501)),
        ((0.0, 0.0, punkter, 0.05, 10),                      (0.8515188432754992, 0.47661894571251673)),
        ((1.0, 1.0, [(0,0),(1,2),(2,4)], 0.1, 5),            (1.3804628806584363, 0.844472098765432)),
    ]
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: kald jeres opg9_2 n gange i træk, og brug det opdaterede (a,b) i næste kald.")
            return
    _ok(name)


def testopg9_4(func):
    """Vægtopdatering for parablen: a,b,c,a_delta,b_delta,c_delta → (a−a_delta, b−b_delta, c−c_delta)"""
    name = "opg9_4"
    cases = [
        ((1, 2, 3, 0, 0, 0),          (1, 2, 3)),
        ((0, 0, 0, 1, 1, 1),          (-1, -1, -1)),
        ((5, 3, 2, 2, 1, 0.5),        (3, 2, 1.5)),
        ((1.5, 0.5, 2.5, 0.3, 0.1, 0.2), (1.2, 0.4, 2.3)),
        ((-2, 4, 1, -3, 2, 0.5),      (1, 2, 0.5)),
    ]
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: samme som opg9_1, bare med 3 parametre — træk hver delta fra sin egen vægt.")
            return
    _ok(name)

def testopg9_5(func):
    """Ét gradient descent-skridt for parablen y=a·x²+b·x+c: a,b,c,punkter,lr → (a,b,c) efter skridtet"""
    name = "opg9_5"
    cases = [
        ((0, 0, 0, [(1,2),(2,1)], 0.1),           (0.6000000000000001, 0.4, 0.30000000000000004)),
        ((1, 0, 0, [(2,1),(1,1)], 0.05),          (0.3999999999999999, -0.30000000000000004, -0.15000000000000002)),
        ((0, 1, 0, [(2,5),(1,1)], 0.02),          (0.24, 1.12, 0.06)),
        ((1, 1, 1, [(1,4),(2,9)], 0.05),          (1.45, 1.25, 1.15)),
    ]
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: brug jeres opg8_2 til at finde gradienten, gang med lr, og brug opg9_4 til at opdatere.")
            return
    _ok(name)

def testopg9_6(func):
    """Træningsloop for parablen: kør opg9_5 n gange i træk"""
    name = "opg9_6"
    punkter = [(1,2),(2,1),(3,0)]
    cases = [
        ((0.0, 0.0, 0.0, punkter, 0.02, 5),   (0.00895658919506171, 0.11662914159670781, 0.13686546457283952)),
        ((0.0, 0.0, 0.0, punkter, 0.02, 20),  (-0.12311594684208543, 0.3421750723226392, 0.449429350984422)),
    ]
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, inp, got, exp)
            print("  Hint: kald jeres opg9_5 n gange i træk, og brug det opdaterede (a,b,c) i næste kald.")
            return
    _ok(name)
