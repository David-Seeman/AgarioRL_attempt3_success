"""The "2+3" splitrun: the only clean mirror-periodic mass pattern with <= 4 pieces per player
(found by tools/mass_cycles.py).  Masses as fractions of the total (37k -> 18.5k per player):

    before a turn   actor X: big 1/3, small 1/6          partner Y: 1/4, 1/6, 1/12
    X single-splits: X's big half (1/6) eats Y's 1/12;   Y's 1/4 eats X's small half (1/12)
    after the turn  X: 1/6, 1/4, 1/12                    Y: 1/3, 1/6      == the mirror image

The geometry of a turn is solved by tools/solve_turn23.py and chained by tools/run_pattern23.py.
"""
import math

TOTAL = 37000.0
MASSES_X = (TOTAL / 3, TOTAL / 6)                      # actor
MASSES_Y = (TOTAL / 4, TOTAL / 6, TOTAL / 12)          # partner


def start_rows(layout, age=100):
    """initial pieces from a layout vector (x,y for X-big at origin fixed; others free):
    layout = [xs_x, xs_y, yb_x, yb_y, ym_x, ym_y, ys_x, ys_y]; X = player 0 acts first"""
    sz = lambda m: math.sqrt(m * 100)
    xs_x, xs_y, yb_x, yb_y, ym_x, ym_y, ys_x, ys_y = layout
    # age 100 ticks: past the 13-tick no-collide window, far from the 416+ tick merge timers
    return [[0, 0.0, 0.0, sz(MASSES_X[0]), age, 0, 0, 0],
            [0, xs_x, xs_y, sz(MASSES_X[1]), age, 0, 0, 0],
            [1, yb_x, yb_y, sz(MASSES_Y[0]), age, 0, 0, 0],
            [1, ym_x, ym_y, sz(MASSES_Y[1]), age, 0, 0, 0],
            [1, ys_x, ys_y, sz(MASSES_Y[2]), age, 0, 0, 0]]
