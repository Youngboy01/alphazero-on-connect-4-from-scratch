"""
AlphaZero on Connect-4 from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_empty_board
import numpy as np

def make_empty_board():
    """Return a 6x7 integer numpy array of zeros representing an empty Connect-4 board."""
    arr = np.zeros((6,7), dtype=int)
    return arr
    pass

# Step 2 - column_top_row
def column_top_row(board, column):
    """Return the lowest empty row in `column`, or -1 if the column is full."""
    n = len(board)
    if board[0][column]!=0:
        return -1
    for row in range(n-1, -1, -1):
        if board[row][column]==0:
            return row
    return -1

# Step 3 - drop_piece
def drop_piece(board, column, player):
    lowest_empty_row = column_top_row(board, column)
    if lowest_empty_row==-1:
        raise ValueError
    new_board = board.copy()
    new_board[lowest_empty_row][column] = player
    return new_board
    pass

# Step 4 - column_full
import numpy as np

def column_full(board, column):
    """Return True if `column` has no empty rows left."""
    return column_top_row(board,column)==-1
    pass

# Step 5 - valid_moves
def valid_moves(board):
    arr=[]
    m = len(board[0])
    for i in range(m):
        if not column_full(board, i):
            arr.append(i)
    return arr

    pass

# Step 6 - four_in_a_row_horizontal
def four_in_a_row_horizontal(board):
    n,m = board.shape
    for row in range(n):
        for col in range(m-3):
            if board[row][col] == board[row][col+1] == board[row][col+2]==board[row][col+3] and board[row][col]!=0:
                return board[row][col]
    return 0

# Step 7 - four_in_a_row_vertical
def four_in_a_row_vertical(board):
    m,n = board.shape
    for row in range(m-3):
        for col in range(n):
            if board[row][col]==board[row+1][col]==board[row+2][col]==board[row+3][col] and board[row][col]!=0:
                return board[row][col]
    return 0
    pass

# Step 8 - four_in_a_row_diagonal_down_right
def four_in_a_row_diagonal_down_right(board):
    m,n = board.shape
    for row in range(m-3):
        for col in range(n-3):
            if board[row][col]!=0 and board[row][col]==board[row+1][col+1]==board[row+2][col+2]==board[row+3][col+3]:
                return board[row][col]
    return 0
    pass

# Step 9 - four_in_a_row_diagonal_up_right
def four_in_a_row_diagonal_up_right(board):
    m,n = board.shape
    for row in range(3,m):
        for col in range(n-3):
            if board[row][col]!=0 and board[row][col]==board[row-1][col+1]==board[row-2][col+2]==board[row-3][col+3]:
                return board[row][col]
    return 0


    pass

# Step 10 - check_winner
import numpy as np

def check_winner(board):
    """Return 1 or 2 if that player has four in a row, else 0."""
    for method in (four_in_a_row_diagonal_down_right, four_in_a_row_diagonal_up_right,four_in_a_row_horizontal,four_in_a_row_vertical):
        winner = method(board)
        if winner:
            return int(winner)
    return 0
    pass

# Step 11 - board_is_full
def board_is_full(board):
    return len(valid_moves(board))==0
    pass

# Step 12 - is_terminal
def is_terminal(board):
    winner  =  check_winner(board)
    if winner!=0:
        return (True,winner)
    if board_is_full(board):
        return (True,0)
    return (False,0)

# Step 13 - other_player
def other_player(player):
    return 3-player
    pass

# Step 14 - step_env
def step_env(board, column, player):
    new_board = drop_piece(board,column,player)
    done,winner = is_terminal(new_board)
    next_player = other_player(player)
    return (new_board,done,winner,next_player)

# Step 15 - encode_board
def encode_board(board, current_player):
    """Encode a 6x7 board as a (2, 6, 7) float32 tensor from current_player's view."""
    other = other_player(current_player)

    current_board = (board==current_player).astype(np.float32)

    opponent_board = (board==other).astype(np.float32)

    return np.stack((current_board,opponent_board))

# Step 16 - board_to_torch_tensor
import torch
def board_to_torch_tensor(board, current_player):
    encoded = encode_board(board,current_player)
    tf = torch.from_numpy(encoded)
    return tf.unsqueeze(0)

# Step 17 - init_conv_backbone
import torch.nn as nn
def init_conv_backbone(in_channels=2, hidden_channels=16):
    return nn.Sequential(
        nn.Conv2d(
            in_channels = in_channels,
            out_channels = hidden_channels,
            kernel_size = 3,
            padding = 1
        )
        ,
        nn.ReLU(),
        nn.Conv2d(
            in_channels = hidden_channels,
            out_channels = hidden_channels,
            kernel_size=3,
            padding=1
        ),
        nn.ReLU()
    )

# Step 18 - init_policy_head
import torch
import torch.nn as nn

def init_policy_head(hidden_channels=16, num_columns=7):
    """Return an nn.Module mapping (B, hidden_channels, 6, 7) -> (B, num_columns) logits."""
    return nn.Sequential(
        nn.Conv2d(
            in_channels = hidden_channels,
            out_channels = 4,
            kernel_size = 1
        ),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(4*6*7, num_columns)
    )

# Step 19 - init_value_head
import torch
import torch.nn as nn

def init_value_head(hidden_channels=16):
    """Return an nn.Module mapping (B, hidden_channels, 6, 7) -> (B, 1) in (-1, 1)."""
    return nn.Sequential(
        nn.Conv2d(
            hidden_channels,
            4,
            1
        ),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(168,1),
        nn.Tanh()
    )

# Step 20 - build_policy_value_net (not yet solved)
# TODO: implement

# Step 21 - policy_value_forward (not yet solved)
# TODO: implement

# Step 22 - action_mask (not yet solved)
# TODO: implement

# Step 23 - masked_policy_logits (not yet solved)
# TODO: implement

# Step 24 - masked_log_softmax (not yet solved)
# TODO: implement

# Step 25 - sample_action_from_policy (not yet solved)
# TODO: implement

# Step 26 - greedy_action_from_policy (not yet solved)
# TODO: implement

# Step 27 - make_mcts_node (not yet solved)
# TODO: implement

# Step 28 - node_q_value (not yet solved)
# TODO: implement

# Step 29 - ucb_score (not yet solved)
# TODO: implement

# Step 30 - select_best_child (not yet solved)
# TODO: implement

# Step 31 - select_leaf (not yet solved)
# TODO: implement

# Step 32 - evaluate_with_network (not yet solved)
# TODO: implement

# Step 33 - expand_node (not yet solved)
# TODO: implement

# Step 34 - backup_value (not yet solved)
# TODO: implement

# Step 35 - run_one_simulation (not yet solved)
# TODO: implement

# Step 36 - run_mcts (not yet solved)
# TODO: implement

# Step 37 - visit_count_policy (not yet solved)
# TODO: implement

# Step 38 - mcts_choose_action (not yet solved)
# TODO: implement

# Step 39 - record_self_play_step (not yet solved)
# TODO: implement

# Step 40 - play_self_play_game (not yet solved)
# TODO: implement

# Step 41 - assign_value_targets (not yet solved)
# TODO: implement

# Step 42 - generate_self_play_batch (not yet solved)
# TODO: implement

# Step 43 - value_loss_mse (not yet solved)
# TODO: implement

# Step 44 - policy_loss_cross_entropy (not yet solved)
# TODO: implement

# Step 45 - l2_regularization_loss (not yet solved)
# TODO: implement

# Step 46 - combined_loss (not yet solved)
# TODO: implement

# Step 47 - encode_batch_states (not yet solved)
# TODO: implement

# Step 48 - iterate_minibatches (not yet solved)
# TODO: implement

# Step 49 - training_step (not yet solved)
# TODO: implement

# Step 50 - training_epoch (not yet solved)
# TODO: implement

# Step 51 - self_play_iteration (not yet solved)
# TODO: implement

# Step 52 - train_loop (not yet solved)
# TODO: implement

# Step 53 - random_policy_action (not yet solved)
# TODO: implement

# Step 54 - greedy_agent_action (not yet solved)
# TODO: implement

# Step 55 - play_one_match (not yet solved)
# TODO: implement

# Step 56 - match_win_rate (not yet solved)
# TODO: implement

# Step 57 - evaluate_against_random (not yet solved)
# TODO: implement

