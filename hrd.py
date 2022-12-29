import heapq
import sys

import numpy as np


def read_input(input_file):
    """ Reading input from input file. Also converts the start state from the
     input format to the output format """
    start_state = np.zeros((5, 4))
    with open(input_file) as file:
        level = 0
        for line in file:
            temp = []
            i = 0
            while i < 4:
                to_append = int(line[i])
                if to_append == 7:
                    temp.append(4)
                    i += 1
                elif to_append in [2, 3, 4, 5, 6]:
                    if line[i + 1] == '\n':
                        temp.append(3)
                        i += 1
                    elif int(line[i+1]) == to_append:
                        temp.append(2)
                        temp.append(2)
                        i += 2
                    else:
                        temp.append(3)
                        i += 1
                else:
                    temp.append(to_append)
                    i += 1
            start_state[level] = temp
            level += 1
    # print(start_state)
    return start_state


def get_successors(state):
    zero_indices = np.where(state == 0)
    successors = []
    # print(zero_indices)
    row_1 = zero_indices[0][0]
    row_2 = zero_indices[0][1]
    col_1 = zero_indices[1][0]
    col_2 = zero_indices[1][1]
    zeros = ((row_1, col_1), (row_2, col_2))
    first = min(col_1, col_2)
    second = max(col_1, col_2)
    minrow = min(row_1, row_2)
    maxrow = max(row_1, row_2)
    # Checking if both zeros next to each other (tgt horizontal)
    if row_1 == row_2 and ((col_1 - col_2 == 1) or (col_1 - col_2 == -1)):
        if row_1 == row_2 and first == 0:
            # All
            if state[row_1][second + 1] == 4:  # There is a 1x1 tile right of the two zeros (all)
                move_4_left_zeros_horizontal(state, row_1, second, row_1,
                                             second+1, successors)
            if state[row_1][second + 1] == 2:  # There is a 1x2 tile right of the zeros (all)
                move_2_left_zeros_horizontal(state, row_1, second, second+1,
                                             row_1, row_2, first,
                                             second + 2, successors)
            # Middle and Top only
            if row_1 != 4:
                if state[row_1 + 1][first] == 4:  # There is a 1x1 tile below [0][0] (top, middle)
                    move_4_left_zeros_horizontal(state, row_1, first,
                                                 row_1 + 1, first,
                                                 successors)
                if state[row_1 + 1][second] == 4:  # There is a 1x1 tile below [0][1] (top, middle)
                    move_4_left_zeros_horizontal(state, row_1, second,
                                                 row_1 + 1, second,
                                                 successors)
                if state[row_1 + 1][first] == 2:  # There is a 1x2 tile below the zeros (top, middle)
                    move_2_left_zeros_horizontal(state, row_1, first,
                                                 second, row_1 + 1,
                                                 row_2 + 1, first, second,
                                                 successors)
                if state[row_1 + 1][first] == 1:  # There is a 2x2 tile below the zeros (middle, top)
                    move_1_left_zeros_horizontal(state, row_1, row_2, first,
                                                 second, row_1 + 2,
                                                 row_2 + 2, first, second,
                                                 successors)
                if state[row_1 + 1][second] == 3:  # There is a 1x2(v) tile below the second 0 (middle, top)
                    move_3_left_zeros_horizontal(state, row_1, second,
                                                 row_1 + 2, second,
                                                 successors)
                if state[row_1 + 1][first] == 3:  # There is a 1x2(v) tile below the first 0 (middle, top)
                    move_3_left_zeros_horizontal(state, row_1, first,
                                                 row_1 + 2, first,
                                                 successors)
            # Middle and Bottom only
            if row_1 != 0:
                if state[row_1 - 1][first] == 4:  # There is a 1x1 tile above [0][0] (middle, bottom)
                    move_4_left_zeros_horizontal(state, row_1, first,
                                                 row_1 - 1, first,
                                                 successors)
                if state[row_1 - 1][second] == 4:  # There is a 1x1 tile above [0][1] (middle, bottom)
                    move_4_left_zeros_horizontal(state, row_1, second,
                                                 row_1 - 1, second,
                                                 successors)
                if state[row_1 - 1][first] == 2:  # There is a 1x2 tile above the zeros (middle, bottom)
                    move_2_left_zeros_horizontal(state, row_1, first,
                                                 second, row_1 - 1,
                                                 row_1 - 1, first, second,
                                                 successors)
                if state[row_1 - 1][first] == 1:  # There is a 2x2 tile above the zeros (middle, bottom)
                    move_1_left_zeros_horizontal(state, row_1, row_2, first,
                                                 second, row_1 - 2,
                                                 row_2 - 2, first, second,
                                                 successors)
                if state[row_1 - 1][first] == 3:  # There is a 1x2(v) tile above the first 0 (middle, bottom)
                    move_3_left_zeros_horizontal(state, row_1, first,
                                                 row_1 - 2, first,
                                                 successors)
                if state[row_1 - 1][second] == 3:  # There is a 1x2(v) tile above the second 0 (middle, bottom)
                    move_3_left_zeros_horizontal(state, row_1, second,
                                                 row_1 - 2, second,
                                                 successors)
        # Both zeros together horizontally on the right
        elif row_1 == row_2 and second == 3:
            # ALL
            # There is a 1x1 tile to the left of the two zeros (all)
            if state[row_1][first - 1] == 4:
                move_4_left_zeros_horizontal(state, row_1, first, row_1,
                                             first - 1, successors)
            # There is a 1x2 tile to the left of the zeros (all)
            if state[row_1][first - 1] == 2:
                move_2_left_zeros_horizontal(state, row_1, first,
                                             first-1,
                                             row_1, row_2, second,
                                             first - 2, successors)

            # Top and middle only
            if row_1 != 4:
                # There is a 1x1 tile below [col][2] (top, middle)
                if state[row_1 + 1][first] == 4:
                    move_4_left_zeros_horizontal(state, row_1, first,
                                                 row_1+1, first, successors)
                # There is a 1x1 tile below [col][3] (top, middle)
                if state[row_1 + 1][second] == 4:
                    move_4_left_zeros_horizontal(state, row_1, second,
                                                 row_1 + 1, second, successors)
                # There is a 1x2 tile below the zeros (top, middle)
                if state[row_1 + 1][second] == 2:
                    move_2_left_zeros_horizontal(state, row_1, first,
                                                 second,
                                                 row_1 + 1, row_2 + 1,
                                                 first,
                                                 second, successors)
                # There is a 2x2 tile below the zeros (top, middle)
                if state[row_1 + 1][second] == 1:
                    move_1_left_zeros_horizontal(state, row_1, row_2,
                                                 first,
                                                 second, row_1 + 2,
                                                 row_2 + 2,
                                                 first, second,
                                                 successors)
                # There is a 1x2(v) tile below [col][2] (top, middle)
                if state[row_1 + 1][first] == 3:
                    move_3_left_zeros_horizontal(state, row_1, first,
                                                 row_1 + 2,
                                                 first, successors)
                # There is a 1x2(v) tile below [col][3] (top, middle)
                if state[row_1 + 1][second] == 3:
                    move_3_left_zeros_horizontal(state, row_1, second,
                                                 row_1 + 2, second,
                                                 successors)

            # Middle and bottom only
            if row_1 != 0:
                # There is a 1x1 tile above [col][2] (middle, bottom)
                if state[row_1 - 1][first] == 4:
                    move_4_left_zeros_horizontal(state, row_1, first, row_1 - 1,
                                                 first, successors)
                # There is a 1x1 tile above [col][3] (middle, bottom)
                if state[row_1 - 1][second] == 4:
                    move_4_left_zeros_horizontal(state, row_1, second,
                                                 row_1 - 1, second, successors)
                # There is a 1x2 tile above the zeros (middle, bottom)
                if state[row_1 - 1][second] == 2:
                    move_2_left_zeros_horizontal(state, row_1, first, second,
                                                 row_1 - 1, row_1 - 1, first,
                                                 second, successors)
                # There is a 2x2 tile above the zeros (middle, bottom)
                if state[row_1 - 1][second] == 1:
                    move_1_left_zeros_horizontal(state, row_1, row_2, first,
                                                 second, row_1 - 2, row_2 - 2,
                                                 first, second, successors)
                # There is a 1x2(v) tile above [col][2] (middle, bottom)
                if state[row_1 - 1][first] == 3:
                    move_3_left_zeros_horizontal(state, row_1, first, row_1 - 2,
                                                 first, successors)
                # There is a 1x2(v) tile above [col][3] (middle, bottom)
                if state[row_1 - 1][second] == 3:
                    move_3_left_zeros_horizontal(state, row_1, second,
                                                 row_1 - 2, second, successors)

        # The two zeros are together and in the middle
        elif row_1 == row_2 and first == 1:
            # one 1x1 piece above the first zero (all cases except row 0)
            if row_1 != 0:
                if state[row_1-1][first] == 4:
                    move_4_left_zeros_horizontal(state, row_1, first,
                                                 row_1-1, first, successors)
                # one 1x1 piece above the second zero (all cases except row 0)
                if state[row_1-1][second] == 4:
                    move_4_left_zeros_horizontal(state, row_1, second,
                                                 row_1-1, second, successors)
                # one 1x2(h) piece above the zeros (all cases except row 0)
                if state[row_1-1][first] == 2 and state[row_1-1][second] == 2 and state[row_1-1][0] != 2 and state[row_1-1][3] != 2:
                    move_2_left_zeros_horizontal(state, row_1, first,
                                                 second, row_1 - 1,
                                                 row_1 - 1, first, second,
                                                 successors)
                # one 2x2 piece above the zeros (all cases except row 0)
                if state[row_1 - 1][first] == 1 and state[row_1 - 1][second] == 1:
                    move_1_left_zeros_horizontal(state, row_1, row_1, first,
                                                 second, row_1 - 2,
                                                 row_1 - 2, first, second,
                                                 successors)
                # one 1x2(v) piece above the first zero (all cases except row 0)
                if state[row_1 - 1][first] == 3:
                    move_3_left_zeros_horizontal(state, row_1, first,
                                                 row_1 - 2, first,
                                                 successors)
                # one 1x2(v) piece above the second zero (all cases except row 0)
                if state[row_1 - 1][second] == 3:
                    move_3_left_zeros_horizontal(state, row_1, second,
                                                 row_1 - 2, second,
                                                 successors)

            # one 1x1 piece below the first zero (all cases except row 4)
            if row_1 != 4:
                if state[row_1+1][first] == 4:
                    move_4_left_zeros_horizontal(state, row_1, first,
                                                 row_1+1, first, successors)
                # one 1x1 piece below the second zero (all cases except row 4)
                if state[row_1+1][second] == 4:
                    move_4_left_zeros_horizontal(state, row_1, second,
                                                 row_1+1, second, successors)
                # one 1x2(h) piece below of the zeros (all cases except row 4)
                if state[row_1 + 1][first] == 2 and state[row_1 + 1][second] == 2 and state[row_1+1][0] != 2 and state[row_1+1][3] != 2:
                    move_2_left_zeros_horizontal(state, row_1, first,
                                                 second, row_1 + 1,
                                                 row_1 + 1, first, second,
                                                 successors)
                # one 2x2 piece below the zeros (all cases except row 4)
                if state[row_1 + 1][first] == 1 and state[row_1 + 1][second] == 1:
                    move_1_left_zeros_horizontal(state, row_1, row_1, first,
                                                 second,
                                                 row_1 + 2, row_1 + 2,
                                                 first,
                                                 second, successors)
                # one 1x2(v) piece below the first zero (all cases except row 4)
                if state[row_1 + 1][first] == 3:
                    move_3_left_zeros_horizontal(state, row_1, first,
                                                 row_1 + 2, first,
                                                 successors)
                # one 1x2(v) piece below the second zero (all cases except row 4)
                if state[row_1 + 1][second] == 3:
                    move_3_left_zeros_horizontal(state, row_1, second,
                                                 row_1 + 2, second,
                                                 successors)
            # one 1x1 piece left of the first zero (all cases except col 0)
            if col_1 != 0:
                if state[row_1][first-1] == 4:
                    move_4_left_zeros_horizontal(state, row_1, first,
                                                 row_1, first-1, successors)
            # one 1x1 piece right of the second zero (all cases except col 3)
            if col_1 != 3:
                if state[row_1][second+1] == 4:
                    move_4_left_zeros_horizontal(state, row_1, second,
                                                 row_1, second+1, successors)

    # Checking if the zeros are vertically together
    elif col_1 == col_2 and ((row_1 - row_2 == 1) or (row_1 - row_2 == -1)):

        if minrow != 0:
            # one 1x1 piece above the top zero (all cases except row 0)
            if state[minrow-1][col_1] == 4:
                move_4_left_zeros_horizontal(state, minrow, col_1, minrow-1,
                                             col_1, successors)
            # one 1x2(v) piece above the top zero (all cases except row 0)
            if state[minrow - 1][col_1] == 3:
                move_3_left_zeros_horizontal(state, minrow, col_1,
                                             minrow - 2, col_1, successors)

        if maxrow != 4:
            # one 1x1 piece below the bottom zero (all cases except row 4)
            if state[maxrow + 1][col_1] == 4:
                move_4_left_zeros_horizontal(state, maxrow, col_1, maxrow+1,
                                             col_1, successors)
            # one 1x2(v) piece below the bottom zero (all cases except row 4)
            if state[maxrow + 1][col_1] == 3:
                move_3_left_zeros_horizontal(state, maxrow, col_1,
                                             maxrow + 2, col_1, successors)
        # one 1x1 piece to the right of the top zero (all cases except col 3)
        if col_1 != 3:
            if state[minrow][col_1+1] == 4:
                move_4_left_zeros_horizontal(state, minrow, col_1, minrow,
                                             col_1+1, successors)
            # one 1x1 piece to the right of the bottom zero (all cases except col 3)
            if state[maxrow][col_1+1] == 4:
                move_4_left_zeros_horizontal(state, maxrow, col_1, maxrow,
                                             col_1 + 1, successors)
            # one 1x2(v) piece right of the zeros (all cases except col 3)
            if state[minrow][col_1 + 1] == 3 and state[maxrow][col_1 + 1] == 3:
                if minrow == 0 or maxrow == 4 or (state[minrow-1][col_1+1] != 3 and state[maxrow+1][col_1+1] != 3):
                    vertical_switch(state, minrow, col_1, minrow, col_1 + 1,
                                    successors)
            # one 1x2(h) piece to the right of the top zero (all cases except col 3)
            if state[minrow][col_1 + 1] == 2:
                move_2_zeros_vertical(state, minrow, col_1, minrow,
                                      col_1 + 2, successors)
            # one 1x2(h) piece to the right of the bottom zero (all cases except col 3)
            if state[maxrow][col_1 + 1] == 2:
                move_2_zeros_vertical(state, maxrow, col_1, maxrow,
                                      col_1 + 2, successors)
            # one 2x2 piece to the right of the zeros (all cases except col 3)
            if state[minrow][col_1 + 1] == 1 and state[maxrow][col_1 + 1] == 1:
                move_1_left_zeros_horizontal(state, minrow, maxrow, col_1,
                                             col_1, minrow, maxrow,
                                             col_1 + 2, col_1 + 2,
                                             successors)

        # one 1x1 piece to the left of the top zero (all cases except col 0)
        if col_1 != 0:
            if state[minrow][col_1-1] == 4:
                move_4_left_zeros_horizontal(state, minrow, col_1, minrow,
                                             col_1 - 1, successors)
            # one 1x1 piece to the left of the bottom zero (all cases except col 0)
            if state[maxrow][col_1-1] == 4:
                move_4_left_zeros_horizontal(state, maxrow, col_1, maxrow,
                                             col_1 - 1, successors)
            # one 1x2(v) piece left of the zeros (all cases except col 0)
            if state[minrow][col_1 - 1] == 3 and state[maxrow][col_1 - 1] == 3:
                if minrow == 0 or maxrow == 4 or (state[minrow - 1][col_1 - 1] != 3 and state[maxrow + 1][col_1 - 1] != 3):
                    vertical_switch(state, minrow, col_1, minrow, col_1 - 1,
                                    successors)
            # one 1x2(h) piece to the left of the top zero (all cases except col 0)
            if state[minrow][col_1 - 1] == 2:
                move_2_zeros_vertical(state, minrow, col_1, minrow,
                                      col_1 - 2, successors)
            # one 1x2(h) piece to the left of the bottom zero (all cases except col 0)
            if state[maxrow][col_1 - 1] == 2:
                move_2_zeros_vertical(state, maxrow, col_1, maxrow,
                                      col_1 - 2, successors)
            # one 2x2 piece to the left of the zeros (all cases except col 0)
            if state[minrow][col_1 - 1] == 1 and state[maxrow][col_1 - 1] == 1:
                move_1_left_zeros_horizontal(state, minrow, maxrow, col_1,
                                             col_1, minrow, maxrow,
                                             col_1 - 2, col_1 - 2,
                                             successors)

    # Checking if the zeros are independent (not together)
    else:
        for i in zeros:
            i_row = i[0]
            i_col = i[1]

            if i_row != 0:
                # one 1x1 piece above the zero (all cases except row 0)
                if state[i_row - 1][i_col] == 4:
                    move_4_left_zeros_horizontal(state, i_row, i_col, i_row - 1,
                                                 i_col, successors)
                # one 1x2(v) piece above the top zero (all cases except row 0)
                if state[i_row - 1][i_col] == 3:
                    move_3_left_zeros_horizontal(state, i_row, i_col,
                                                 i_row - 2, i_col,
                                                 successors)

            if i_row != 4:
                # one 1x1 piece below the zero (all cases except row 4)
                if state[i_row + 1][i_col] == 4:
                    move_4_left_zeros_horizontal(state, i_row, i_col, i_row + 1,
                                                 i_col, successors)
                # one 1x2(v) piece below the bottom zero (all cases except row 4)
                if state[i_row + 1][i_col] == 3:
                    move_3_left_zeros_horizontal(state, i_row, i_col,
                                                 i_row + 2, i_col,
                                                 successors)

            if i_col != 3:
                # one 1x1 piece right of the zero (all cases except col 3)
                if state[i_row][i_col + 1] == 4:
                    move_4_left_zeros_horizontal(state, i_row, i_col, i_row,
                                                 i_col + 1, successors)
                # one 1x2(h) piece to the right of the zero (all cases except col 3)
                if state[i_row][i_col + 1] == 2:
                    move_2_into_one_0(state, i_row, i_col,
                                      i_row, i_col + 2, successors)

            if i_col != 0:
                # one 1x1 piece left of the zero (all cases except col 0)
                if state[i_row][i_col - 1] == 4:
                    move_4_left_zeros_horizontal(state, i_row, i_col, i_row,
                                                 i_col - 1, successors)
                # one 1x2(h) piece to the left of the zero (all cases except col 0)
                if state[i_row][i_col - 1] == 2:
                    move_2_into_one_0(state, i_row, i_col,
                                      i_row, i_col - 2, successors)

    # for i in successors:
    #     print(i)
    # print(successors)
    return successors


def vertical_switch(state, row, col, new_row, new_col, successors):
    new_state = np.copy(state)
    new_state[new_row][new_col] = 0
    new_state[new_row+1][new_col] = 0
    new_state[row][col] = 3
    new_state[row+1][col] = 3
    successors.append((new_state, state))


def move_3_left_zeros_horizontal(state, row, col, new_0_row, new_0_col, successors):
    new_state = np.copy(state)
    new_state[new_0_row][new_0_col] = 0
    new_state[row][col] = 3
    successors.append((new_state, state))


def move_4_left_zeros_horizontal(state, row, col, new_0_row, new_0_col, successors):
    """Moves 4 up to one of the horizontal tgt zeros"""
    new_state = np.copy(state)
    new_state[new_0_row][new_0_col] = 0
    new_state[row][col] = 4

    successors.append((new_state, state))


def move_1_left_zeros_horizontal(state, row_1, row_2, col_1, col_2, new_0_row_1, new_0_row_2, new_0_col_1, new_0_col_2, successors):
    new_state = np.copy(state)
    new_state[row_1][col_1] = 1
    new_state[row_2][col_2] = 1
    new_state[new_0_row_1][new_0_col_1] = 0
    new_state[new_0_row_2][new_0_col_2] = 0

    successors.append((new_state, state))


def move_2_left_zeros_horizontal(state, row, col_1, col_2, new_0_row_1, new_0_row_2, new_0_col_1, new_0_col_2, successors):
    new_state = np.copy(state)
    new_state[row][col_1] = 2
    new_state[row][col_2] = 2
    new_state[new_0_row_1][new_0_col_1] = 0
    new_state[new_0_row_2][new_0_col_2] = 0

    successors.append((new_state, state))


def move_2_into_one_0(state, row, col, new_0_row, new_0_col, successors):
    new_state = np.copy(state)
    new_state[row][col] = 2
    new_state[new_0_row][new_0_col] = 0

    successors.append((new_state, state))


def move_2_zeros_vertical(state, row, col, new_0_row, new_0_col, successors):
    new_state = np.copy(state)
    new_state[row][col] = 2
    new_state[new_0_row][new_0_col] = 0

    successors.append((new_state, state))


class State:

    def __init__(self, state_rep, parent_str, g_value, f_value):
        self.state_rep = state_rep
        self.parent_str = parent_str
        self.g_value = g_value
        self.f_value = f_value

    def __lt__(self, other_state):
        return self.f_value < other_state.f_value


def get_astar_successors(state, g_value):
    """Successor function for A* search with Manhattan Heuristic"""
    raw_suc = get_successors(state)
    heur_suc = []
    for i in raw_suc:
        h_value = h_value_finder(i[0])
        f_value = h_value + g_value + 1
        heur_suc.append(State(i[0], np.array_str(i[1]), g_value + 1, f_value))
    return heur_suc


def get_astar_successors_advanced_heuristic(state, g_value):
    """Successor function for A* search with Advanced Heuristic"""
    raw_suc = get_successors(state)
    heur_suc = []
    for i in raw_suc:
        h_value = advanced_h_value_finder(i[0])
        f_value = h_value + g_value + 1
        heur_suc.append(State(i[0], np.array_str(i[1]), g_value + 1, f_value))
    return heur_suc


def h_value_adder(raw_h, mincol, valA, valB, valC):
    if mincol == 0:
        raw_h += valA
    if mincol == 1:
        raw_h += valB
    if mincol == 2:
        raw_h += valC
    return raw_h


def advanced_h_value_finder(state):
    """ In addition to taking the distance of the 2x2 block on [4][1], [4][2]
        into account, this heuristic also counts how many pieces below the
        2x2 block will have to be moved to make way for the 2x2 block to reach
        the last row. The heuristic adds one to the h_value for each such move.
        This heuristic does not count how many pieces will be moved to get the
        2x2 block horizontally aligned with the opening at the bottom.
    """
    raw_h_value = h_value_finder(state)
    print(raw_h_value)
    one_indices = np.where(state == 1)
    row_1 = one_indices[0][0]
    row_2 = one_indices[0][1]
    row_3 = one_indices[0][2]
    row_4 = one_indices[0][3]
    col_1 = one_indices[1][0]
    col_2 = one_indices[1][1]
    col_3 = one_indices[1][2]
    col_4 = one_indices[1][3]
    maxrow = max(row_1, row_2, row_3, row_4)
    maxcol = max(col_1, col_2, col_3, col_4)
    mincol = min(col_1, col_2, col_3, col_4)
    minrow = min(row_1, row_2, row_3, row_4)
    # If 2x2 block already at the bottom row, no new blocks need to be moved
    # to get it to the bottom row
    if maxrow == 4:
        return raw_h_value
    else:
        # 1x2(v) block below left ones (block can be one or two rows down)
        if state[maxrow+1][mincol] == 3:  # block one row down
            # if left sided, block will have to be moved to the right (2 moves)
            # if in the middle, block can be moved to the left (1 move)
            # if right sided, block will have to be moved to the left (1 move)
            raw_h_value = h_value_adder(raw_h_value, mincol, 2, 1, 1)

        if maxrow == 1 and (state[maxrow+2][mincol] == 3 and state[maxrow+3][mincol] == 3):  # block two rows down
            raw_h_value = h_value_adder(raw_h_value, mincol, 2, 1, 1)

        # 1x2(v) block below right ones (block can be one or two rows down)
        if state[maxrow + 1][maxcol] == 3:  # block one row down
            # if left sided, block will have to be moved to the right (1 move)
            # if in the middle, block can be moved to the right(1 move)
            # if right sided, block will have to be moved to the left (2 moves)
            raw_h_value = h_value_adder(raw_h_value, mincol, 1, 1, 2)

        if maxrow == 1 and (state[maxrow + 2][maxcol] == 3 and state[maxrow + 3][maxcol] == 3):  # block two rows down
            raw_h_value = h_value_adder(raw_h_value, mincol, 1, 1, 2)

        # 1x1 block below left one (blocks can be one row, two rows or three rows below)
        # one row below
        if state[maxrow+1][mincol] == 4:
            # if left sided, block will have to be moved to the right (2 moves)
            # if in the middle, block can be moved to the left (1 move)
            # if right sided, block will have to be moved to the left (1 move)
            raw_h_value = h_value_adder(raw_h_value, mincol, 2, 1, 1)

        # two rows below
        if maxrow <= 2 and (state[maxrow+2][mincol] == 4):
            raw_h_value = h_value_adder(raw_h_value, mincol, 2, 1, 1)

        # three rows below
        if maxrow == 1 and state[maxrow+3][mincol] == 4:
            raw_h_value = h_value_adder(raw_h_value, mincol, 2, 1, 1)

        # 1x1 block below right zeros (blocks can be one row, two rows or three rows below)
        # one row below
        if state[maxrow+1][maxcol] == 4:
            # if left sided, block will have to be moved to the right (1 move)
            # if in the middle, block can be moved to the right (1 move)
            # in right sided, block will have to be moved to the left (2 moves)
            raw_h_value = h_value_adder(raw_h_value, mincol, 1, 1, 2)

        # two rows below
        if maxrow <= 2 and (state[maxrow+2][maxcol] == 4):
            raw_h_value = h_value_adder(raw_h_value, mincol, 1, 1, 2)

        # three rows below
        if maxrow == 1 and state[maxrow+3][maxcol] == 4:
            raw_h_value = h_value_adder(raw_h_value, mincol, 1, 1, 2)

        # 1x2(h) block exactly below the two bottom zeros
        # one row down
        if (state[maxrow+1][mincol] == 2) and (state[maxrow+1][maxcol] == 2):
            # if left sided, block will have to be moved right (2 moves)
            # if middle, block can be moved to either side (2 moves)
            # if right sided, block will have to be moved to the left (2 moves)
            raw_h_value = h_value_adder(raw_h_value, mincol, 2, 2, 2)

        # two rows down
        if maxrow <= 2 and ((state[maxrow+2][mincol] == 2) and
                            (state[maxrow+1][maxcol] == 2)):
            raw_h_value = h_value_adder(raw_h_value, mincol, 2, 2, 2)

        # three rows down
        if maxrow == 1 and ((state[maxrow+3][mincol] == 2) and
                            (state[maxrow+1][maxcol] == 2)):
            raw_h_value = h_value_adder(raw_h_value, mincol, 2, 2, 2)

        # 1x2(h) block only below the left ones (only happens if the 2x2 block is in the middle or right-sided)
        # block can be one, two or three rows down
        # one row down
        if mincol != 0 and\
                (state[maxrow + 1][mincol] == 2) and\
                (state[maxrow + 1][maxcol] != 2) and\
                (state[maxrow+1][mincol-1] == 2):
            # if in the middle, the 2x2 block will have to be moved since no movement of the 1x2 block will free up space (1 move)
            # if 2x2 is right-sided, the 1x2(h) block can move to the left (1 move)
            raw_h_value += 1

        # two rows down
        if mincol != 0 and maxrow <= 2\
                and (state[maxrow + 2][mincol] == 2) and\
                (state[maxrow + 2][maxcol] == 2) and\
                (state[maxrow+2][mincol-1] == 2):
            raw_h_value += 1

        # three rows down
        if mincol != 0 and maxrow == 1\
                and (state[maxrow + 3][mincol] == 2) and\
                (state[maxrow + 3][maxcol] == 2) and\
                (state[maxrow+3][mincol-1] == 2):
            raw_h_value += 1

        # 1x2(h) block only below the right ones (only happens if the 2x2 block is in the middle or left-sided)
        # block can be one, two or three rows down
            # if in the middle, 2x2 block will be moved since no movement of the 1x2 block will free up space (1 move)
            # if 2x2 is left-sided, the 1x2(h) block can be moved to the right (1 move)
        # one row down
        if maxcol != 3 and \
                (state[maxrow + 1][maxcol] == 2) and \
                (state[maxrow + 1][mincol] != 2) and \
                (state[maxrow + 1][maxcol + 1] == 2):
            raw_h_value += 1

        # two rows down
        if maxcol != 3 and maxrow <= 2 and \
                (state[maxrow + 2][maxcol] == 2) and \
                (state[maxrow + 2][mincol] != 2) and \
                (state[maxrow + 2][maxcol + 1] == 2):
            raw_h_value += 1

        # three rows down
        if maxcol != 3 and maxrow == 1 and \
                (state[maxrow + 3][maxcol] == 2) and \
                (state[maxrow + 3][mincol] != 2) and \
                (state[maxrow + 3][maxcol + 1] == 2):
            raw_h_value += 1

        print(raw_h_value)
        return raw_h_value


def h_value_finder(state):
    one_indices = np.where(state == 1)
    row_1 = one_indices[0][0]
    row_2 = one_indices[0][1]
    row_3 = one_indices[0][2]
    row_4 = one_indices[0][3]
    col_1 = one_indices[1][0]
    col_2 = one_indices[1][1]
    col_3 = one_indices[1][2]
    col_4 = one_indices[1][3]
    maxrow = max(row_1, row_2, row_3, row_4)
    maxcol = max(col_1, col_2, col_3, col_4)
    if maxcol == 2:
        value = 4 - maxrow
        return 4 - maxrow
    else:
        value = (4 - maxrow) + 1
        return (4 - maxrow) + 1


def astar(input_file):
    """Performs A* search with Manhattan Heuristic"""
    active_state = read_input(input_file)
    active_state = active_state.astype(int)
    goal_found = 0
    initial_fval = h_value_finder(active_state)
    frontier = [State(active_state, "start", 0, initial_fval)]
    heapq.heapify(frontier)
    path_dict = {}
    explored = {}
    i = -1
    while goal_found == 0:
        active_object = heapq.heappop(frontier)
        active_state = active_object.state_rep
        # print(active_state)
        parent = active_object.parent_str
        key = np.array_str(active_state)
        if key in explored:
            pass
            # print("pruned")
        elif key not in explored:
            i += 1
            path_dict[key] = parent
            explored[key] = active_state
            if active_state[4][1] == active_state[4][2] == 1:
                goal_found = 1
                return i, path_dict, key
            for state in get_astar_successors(active_state, active_object.g_value):
                heapq.heappush(frontier, state)
    return i, path_dict


def astar_advanced_heuristic(input_file):
    """Performs A* search with Advanced Heuristic"""
    active_state = read_input(input_file)
    active_state = active_state.astype(int)
    goal_found = 0
    initial_fval = advanced_h_value_finder(active_state)
    frontier = [State(active_state, "start", 0, initial_fval)]
    heapq.heapify(frontier)
    path_dict = {}
    explored = {}
    i = -1
    while goal_found == 0:
        active_object = heapq.heappop(frontier)
        active_state = active_object.state_rep
        parent = active_object.parent_str
        key = np.array_str(active_state)
        if key in explored:
            pass
            # print("pruned")
        elif key not in explored:
            i += 1
            path_dict[key] = parent
            explored[key] = active_state
            if active_state[4][1] == active_state[4][2] == 1:
                goal_found = 1
                return i, path_dict, key
            for state in get_astar_successors_advanced_heuristic(active_state, active_object.g_value):
                heapq.heappush(frontier, state)
    return i, path_dict


def dfs(input_file):
    active_state = read_input(input_file)
    active_state = active_state.astype(int)
    goal_found = 0  # 1 when goal is found

    frontier = [(active_state, "start")]
    path_dict = {}
    explored = {}
    i = -1
    while goal_found == 0:

        active_object = frontier.pop(-1)
        active_state = active_object[0]
        parent = active_object[1]
        key = np.array_str(active_state)
        if key in explored:
            pass
            # print("pruned")
        elif key not in explored:
            i += 1
            if isinstance(parent, str):
                path_dict[key] = parent
            else:
                path_dict[key] = np.array_str(parent)
            explored[key] = active_state
            if active_state[4][1] == active_state[4][2] == 1:
                goal_found = 1
                return i, path_dict, key
            frontier.extend(get_successors(active_state))
    return i, path_dict


def generate_output_file(result: tuple, output_file):
    # cost = result[0]
    final_state = result[2]
    path_dict = result[1]
    parent = path_dict[final_state]
    path = [final_state, parent]
    while parent != "start":
        to_append = path_dict[parent]
        path.append(to_append)
        parent = to_append
    # print(path)

    cost = len(path) - 2
    file = open(output_file, "w")
    first_line = "Cost of the solution:" + ' ' + str(cost) + '\n'
    file.write(first_line)
    i = len(path) - 2
    while i != -1:
        to_write = path[i][1:-1]
        first = to_write[1:8].replace(" ", "")
        second = to_write[12:19].replace(" ", "")
        third = to_write[23:30].replace(" ", "")
        fourth = to_write[34:41].replace(" ", "")
        fifth = to_write[45:52].replace(" ", "")
        file.write(first)
        file.write('\n')
        file.write(second)
        file.write('\n')
        file.write(third)
        file.write('\n')
        file.write(fourth)
        file.write('\n')
        file.write(fifth)
        file.write('\n')
        file.write('\n')
        i = i-1


if __name__ == '__main__':
    # print_hi('PyCharm')
    # ss = read_input("test_input.txt")
    # get_successors(ss)
    # h_value_finder(ss)
    # generate_output_file(dfs("bug_fixing.txt"))
    # generate_output_file(astar("sample_input.txt"), "classic.txt")
    # generate_output_file(astar_advanced_heuristic("sample_input.txt"), "classic.txt")
    # advanced_h_value_finder(ss)

    # print(len(sys.argv))
    test_file = sys.argv[1]
    dfs_output_file = sys.argv[2]
    astar_output_file = sys.argv[3]
    generate_output_file(dfs(test_file), dfs_output_file)
    generate_output_file(astar(test_file), astar_output_file)
