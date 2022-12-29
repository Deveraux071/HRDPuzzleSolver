# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
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
    print(start_state)
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
                                             second,
                                             row_1, row_2, first - 1,
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

    for i in successors:
        print(i)
    # print(successors)
    return successors


def vertical_switch(state, row, col, new_row, new_col, successors):
    new_state = np.copy(state)
    new_state[new_row][new_col] = 0
    new_state[new_row+1][new_col] = 0
    new_state[row][col] = 3
    new_state[row+1][col] = 3
    # successors.append(tuple(map(tuple, new_state)))

    # old implementation
    # successors.append(new_state)
    # new implementation
    successors.append((new_state, state))

    # successors.append(tuple(map(tuple, new_state)))
    # print("printing now")
    # print(new_state)


def move_3_left_zeros_horizontal(state, row, col, new_0_row, new_0_col, successors):
    new_state = np.copy(state)
    new_state[new_0_row][new_0_col] = 0
    new_state[row][col] = 3
    # successors.append(tuple(map(tuple, new_state)))

    # old implementation
    # successors.append(new_state)
    # new implementation
    successors.append((new_state, state))
    # print(new_state)


def move_4_left_zeros_horizontal(state, row, col, new_0_row, new_0_col, successors):
    """Moves 4 up to one of the horizontal tgt zeros"""
    new_state = np.copy(state)
    new_state[new_0_row][new_0_col] = 0
    new_state[row][col] = 4
    # successors.append(tuple(map(tuple, new_state)))

    # old implementation
    # successors.append(new_state)
    # new implementation
    successors.append((new_state, state))
    # print("Hey")
    # print(new_state)
    # print(successors)


def move_1_left_zeros_horizontal(state, row_1, row_2, col_1, col_2, new_0_row_1, new_0_row_2, new_0_col_1, new_0_col_2, successors):
    new_state = np.copy(state)
    new_state[row_1][col_1] = 1
    new_state[row_2][col_2] = 1
    new_state[new_0_row_1][new_0_col_1] = 0
    new_state[new_0_row_2][new_0_col_2] = 0
    # successors.append(tuple(map(tuple, new_state)))

    # old implementation
    # successors.append(new_state)
    # new implementation
    successors.append((new_state, state))
    # print(new_state)


def move_2_left_zeros_horizontal(state, row, col_1, col_2, new_0_row_1, new_0_row_2, new_0_col_1, new_0_col_2, successors):
    new_state = np.copy(state)
    new_state[row][col_1] = 2
    new_state[row][col_2] = 2
    new_state[new_0_row_1][new_0_col_1] = 0
    new_state[new_0_row_2][new_0_col_2] = 0
    # successors.append(tuple(map(tuple, new_state)))

    # old implementation
    # successors.append(new_state)
    # new implementation
    successors.append((new_state, state))
    # print("HEYY")
    # print(new_state)


def move_2_into_one_0(state, row, col, new_0_row, new_0_col, successors):
    new_state = np.copy(state)
    new_state[row][col] = 2
    new_state[new_0_row][new_0_col] = 0
    # successors.append(tuple(map(tuple, new_state)))

    # old implementation
    # successors.append(new_state)
    # new implementation
    successors.append((new_state, state))
    # print(new_state)


def move_2_zeros_vertical(state, row, col, new_0_row, new_0_col, successors):
    new_state = np.copy(state)
    new_state[row][col] = 2
    new_state[new_0_row][new_0_col] = 0
    # successors.append(tuple(map(tuple, new_state)))

    # old implementation
    # successors.append(new_state)
    # new implementation
    successors.append((new_state, state))
    # print("HEYY")
    # print(new_state)


class State:

    def __init__(self, state_rep, parent_str, g_value, f_value):
        self.state_rep = state_rep
        self.parent_str = parent_str
        self.g_value = g_value
        self.f_value = f_value

    def __lt__(self, other_state):
        return self.f_value < other_state.f_value


def get_astar_successors(state, g_value):
    raw_suc = get_successors(state)
    heur_suc = []
    for i in raw_suc:
        h_value = h_value_finder(i[0])
        f_value = h_value + g_value + 1
        heur_suc.append(State(i[0], np.array_str(i[1]), g_value + 1, f_value))
    return heur_suc


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
        print(active_state)
        parent = active_object.parent_str
        key = np.array_str(active_state)
        if key in explored:
            print("pruned")
        elif key not in explored:
            i += 1
            path_dict[key] = parent
            # if parent == "start":
            #     path_dict[key] = parent
            # else:
            #     path_dict[key] = parent
            explored[key] = active_state
            if active_state[4][1] == active_state[4][2] == 1:
                goal_found = 1
                print(i)
                print(path_dict)
                return i, path_dict, key
            # frontier.extend(get_astar_successors(active_state,
            #                                      active_object.g_value))
            for state in get_astar_successors(active_state, active_object.g_value):
                heapq.heappush(frontier, state)
            print(i)
    return i, path_dict


def dfs(input_file):
    active_state = read_input(input_file)
    active_state = active_state.astype(int)
    goal_found = 0  # 1 when goal is found
    # parent = "start"
    frontier = [(active_state, "start")]
    path_dict = {}
    explored = {}
    i = -1
    while goal_found == 0:
        # frontier.extend(get_successors(active_state))
        active_object = frontier.pop(-1)
        active_state = active_object[0]
        parent = active_object[1]
        print(active_state)
        # key = np.array_str(active_state)
        key = np.array_str(active_state)
        # if path_dict == {}:
        #     path_dict[key] = "start"
        # else:
        #     path_dict[key] =
        if key in explored:
            print("pruned")
            # print(key)
        elif key not in explored:
            i += 1
            if parent == "start":
                path_dict[key] = parent
            else:
                path_dict[key] = np.array_str(parent)
            explored[key] = active_state
            if active_state[4][1] == active_state[4][2] == 1:
                goal_found = 1
                print(i)
                # print(path_dict)
                return i, path_dict, key
            frontier.extend(get_successors(active_state))
            # parent = key
            # active_state = curr
            # parent = key
            print(i)
    # print("printing frontier")
    # print(frontier)
    # print(path_dict)
    return i, path_dict

    # Original Implementation
    # active_state = read_input(input_file)
    # active_state = active_state.astype(int)
    # goal_found = 0  # 1 when goal is found
    # parent = "start"
    # frontier = [active_state]
    # path_dict = {}
    # explored = {}
    # i = -1
    # while goal_found == 0:
    #     # frontier.extend(get_successors(active_state))
    #     active_state = frontier.pop(-1)
    #     print(active_state)
    #     # key = np.array_str(active_state)
    #     key = np.array_str(active_state)
    #     # if path_dict == {}:
    #     #     path_dict[key] = "start"
    #     # else:
    #     #     path_dict[key] =
    #     if key in explored:
    #         print("pruned")
    #         # print(key)
    #     elif key not in explored:
    #         i += 1
    #         path_dict[key] = parent
    #         explored[key] = active_state
    #         if active_state[4][1] == active_state[4][2] == 1:
    #             goal_found = 1
    #             print(i)
    #             return i, path_dict, key
    #         frontier.extend(get_successors(active_state))
    #         parent = key
    #         # active_state = curr
    #         # parent = key
    #         print(i)
    # # print("printing frontier")
    # # print(frontier)
    # return i, path_dict


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
    # print(path)

# For 1x2(v) pieces:

    # FOR 2x2 PIECES:
    # If the zeros are at the top left corner
    # if row_1 == row_2 == 0 and min(col_1, col_2) == 0:
    #     if state[row_1+1][first] == 4:  # There is a 1x1 tile below [0][1]
    #         new_state = np.copy(state)
    #         new_state[row_1 + 1][first] = 0
    #         new_state[row_1][first] = 4
    #         successors.append(new_state)
    #         print(new_state)
    #     if state[row_1+1][second] == 4:  # There is a 1x1 tile below [0][0]
    #         new_state = np.copy(state)
    #         new_state[row_1 + 1][second] = 0
    #         new_state[row_1][second] = 4
    #         successors.append(new_state)
    #         print(new_state)
    #     if state[row_1][second + 1] == 4:  # There is a 1x1 tile to the right of the zeros
    #         move_4_left_zeros_horizontal(state, row_1, second, row_1, second+1, successors)
    #     if state[row_1][second+1] == 2:  # There is a 1x2 tile to right
    #         new_state = np.copy(state)
    #         new_state[row_1][1] = 2
    #         new_state[row_1][3] = 0
    #         successors.append(new_state)
    #         print(new_state)
    #     if state[row_1 + 1][first] == 2:  # There is a 1x2(h) tile below
    #         new_state = np.copy(state)
    #         new_state[row_1][first] = 2
    #         new_state[row_1][second] = 2
    #         new_state[row_1+1][first] = 0
    #         new_state[row_1+1][second] = 0
    #         successors.append(new_state)
    #         print(new_state)
    #     if state[row_1 + 1][first] == 1:  # There is a 2x2 tile below
    #         new_state = np.copy(state)
    #         new_state[row_1][first] = 1
    #         new_state[row_1][second] = 1
    #         new_state[row_1 + 2][first] = 0
    #         new_state[row_1 + 2][second] = 0
    #         successors.append(new_state)
    #         print(new_state)
    #     if state[row_1 + 1][first] == 3:  # There is a 1x2(v) tile below the first 0
    #         new_state = np.copy(state)
    #         new_state[row_1 + 2][first] = 0
    #         new_state[row_1][first] = 3
    #         successors.append(new_state)
    #         print(new_state)
    #     if state[row_1 + 1][second] == 3:  # There is a 1x2(v) tile below the second 0
    #         new_state = np.copy(state)
    #         new_state[row_1 + 2][second] = 0
    #         new_state[row_1][second] = 3
    #         successors.append(new_state)
    #         print(new_state)
    # If the zeros are at the left of any row expect the top and last
    # elif row_1 == row_2 != 0 and row_1 == row_2 != 4 and first == 0:
    #     if state[row_1+1][first] == 4:  # There is a 1x1 tile below [0][0] (top, middle)
    #         move_4_left_zeros_horizontal(state, row_1, first, row_1+1, first, successors)
    #     if state[row_1+1][second] == 4:  # There is a 1x1 tile below [0][1] (top, middle)
    #         move_4_left_zeros_horizontal(state, row_1, second, row_1+1, second, successors)
    #     if state[row_1][second + 1] == 4:  # There is a 1x1 tile right of the two zeros (all)
    #         move_4_left_zeros_horizontal(state, row_1, second, row_1, second+1, successors)
    #     if state[row_1-1][first] == 4:  # There is a 1x1 tile above [0][0] (middle, bottom)
    #         move_4_left_zeros_horizontal(state, row_1, first, row_1 - 1, first, successors)
    #     if state[row_1-1][second] == 4:  # There is a 1x1 tile above [0][1] (middle, bottom)
    #         move_4_left_zeros_horizontal(state, row_1, second, row_1 - 1, second, successors)
    #     if state[row_1-1][first] == 2:  # There is a 1x2 tile above the zeros (middle, bottom)
    #         move_2_left_zeros_horizontal(state, row_1, first, second, row_1 - 1, row_1 - 1, first, second, successors)
    #     if state[row_1+1][first] == 2:  # There is a 1x2 tile below the zeros (top, middle)
    #         move_2_left_zeros_horizontal(state, row_1, first, second, row_1+1, row_2+1, first, second, successors)
    #     if state[row_1][second + 1] == 2:  # There is a 1x2 tile right of the zeros (all)
    #         move_2_left_zeros_horizontal(state, row_1, first, second, row_1, row_2, second+1, second+2, successors)
    #     if state[row_1-1][first] == 1:  # There is a 2x2 tile above the zeros (middle, bottom)
    #         move_1_left_zeros_horizontal(state, row_1, row_2, first, second, row_1 - 2, row_2 - 2, first, second, successors)
    #     if state[row_1+1][first] == 1:  # There is a 2x2 tile below the zeros (middle, top)
    #         move_1_left_zeros_horizontal(state, row_1, row_2, first, second, row_1 + 2, row_2 + 2, first, second, successors)
    #     if state[row_1+1][second] == 3:  # There is a 1x2(v) tile below the second 0 (middle, top)
    #         move_3_left_zeros_horizontal(state, row_1, second, row_1 + 2, second, successors)
    #     if state[row_1+1][first] == 3:  # There is a 1x2(v) tile below the first 0 (middle, top)
    #         move_3_left_zeros_horizontal(state, row_1, first, row_1 + 2, first, successors)
    #     if state[row_1-1][first] == 3:  # There is a 1x2(v) tile above the first 0 (middle, bottom)
    #         move_3_left_zeros_horizontal(state, row_1, first, row_1 - 2, first, successors)
    #     if state[row_1-1][second] == 3:  # There is a 1x2(v) tile above the second 0 (middle, bottom)
    #         move_3_left_zeros_horizontal(state, row_1, second, row_1 - 2, second, successors)


# def move_1_down_left_zeros_horizontal(state, row, col_1, col_2, successors):
#     new_state = np.copy(state)
#     new_state[row][col_1] = 2
#     new_state[row][col_2] = 2
#     new_state[row][col_2 + 1] = 0
#     new_state[row][col_2 + 2] = 0
#     successors.append(new_state)
#     print(new_state)


# def move_4_down_left_zeros_horizontal(state, row, col, successors):
#     """Moves 4 up to one of the horizontal tgt zeros"""
#     new_state = np.copy(state)
#     new_state[row - 1][col] = 0
#     new_state[row][col] = 4
#     successors.append(new_state)
#     print(new_state)
#     # print(successors)


# def move_4_left_left_zeros_horizontal(state, row, col, successors):
#     new_state = np.copy(state)
#     new_state[row][col + 1] = 0
#     new_state[row][col] = 4
#     successors.append(new_state)
#     print(new_state)


# def move_4_up_left_zeros_horizontal(state, row, col, successors):
#     """Moves 4 up to one of the horizontal tgt zeros"""
#     new_state = np.copy(state)
#     new_state[row + 1][col] = 0
#     new_state[row][col] = 4
#     successors.append(new_state)
#     print(new_state)
#     # print(successors)


# def move_2_left_left_zeros_horizontal(state, row, col_1, col_2, successors):
#     new_state = np.copy(state)
#     new_state[row][col_1] = 2
#     new_state[row][col_2] = 2
#     new_state[row][col_2+1] = 0
#     new_state[row][col_2+2] = 0
#     successors.append(new_state)
#     print(new_state)


# def move_2_up_left_zeros_horizontal(state, row, col_1, col_2, successors):
#     new_state = np.copy(state)
#     new_state[row+1][col_1] = 0
#     new_state[row+1][col_2] = 0
#     new_state[row][col_1] = 2
#     new_state[row][col_2] = 2
#     successors.append(new_state)
#     print(new_state)


# def move_2_down_left_zeros_horizontal(state, row, col_1, col_2, successors):
#     new_state = np.copy(state)
#     new_state[row-1][col_1] = 0
#     new_state[row-1][col_2] = 0
#     new_state[row][col_1] = 2
#     new_state[row][col_2] = 2
#     successors.append(new_state)
#     print(new_state)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    # ss = read_input("test_input.txt")
    # get_successors(ss)
    # h_value_finder(ss)
    # generate_output_file(dfs("bug_fixing.txt"))
    # generate_output_file(astar("sample_input.txt"), "astar")


    print(len(sys.argv))
    input_file = sys.argv[1]
    dfs_output_file = sys.argv[2]
    astar_output_file = sys.argv[3]
    # generate_output_file(dfs(input_file), dfs_output_file)
    generate_output_file(astar(input_file), astar_output_file)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
