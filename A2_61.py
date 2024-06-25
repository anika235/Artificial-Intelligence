import math
import random
import copy
import time
from tabulate import tabulate
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from random import choice

class MCTS:

    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight

    def choose(self, node):# find which node gonna be choosen next
        if node.is_terminal(): #terminal means the game is ended
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:# not the children-> not expnaded yet-> find the children
            return node.find_random_child()

        def score(n): #calculates the score
            if self.N[n] == 0: #  returns negative infinity to avoid selecting unseen moves
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward
        return max(self.children[node], key=score)

    def do_rollout(self, node):
        path = self._select(node) #select the leaf node
        leaf = path[-1] # expansion
        self._expand(leaf)
        reward = self._simulate(leaf) #reward simulation
        self._backpropagate(path, reward) #backpropagate for all nodes

    def _select(self, node): # selecting the leaf node
        path = [] #initialize the path
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]: # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys() #number of unexplored children nodes
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)  # selecting the best child to explore

    def _expand(self, node):
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    def _simulate(self, node): #updating the rewards
        invert_reward = True
        while True:
            if node.is_terminal():
                reward = node.reward()
                return 1 - reward if invert_reward else reward
            node = node.find_random_child()
            invert_reward = not invert_reward

    def _backpropagate(self, path, reward): # Send the reward back up to the ancestors of the leaf
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

    def _uct_select(self, node): # selecting the best children
        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])
        log_N_vertex = math.log(self.N[node])
        def uct(n):
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )
        return max(self.children[node], key=uct)

class Node(ABC):
    @abstractmethod
    def find_children(self):
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(self, other):
        "Nodes must be comparable"
        return True

_TTTB = namedtuple("TicTacToeBoard", "tup turn winner terminal")

# Inheriting from a namedtuple is convenient because it makes the class
# immutable and predefines
class TicTacToeBoard(_TTTB, Node):
    def find_children(board):
        if board.terminal:  # If the game is finished then no moves can be made
            return set()
        # Otherwise, you can make a move in each of the empty spots
        return {
            board.make_move(i) for i, value in enumerate(board.tup) if value is None
        }

    def find_random_child(board):
        if board.terminal:
            return None  # If the game is finished then no moves can be made
        empty_spots = [i for i, value in enumerate(board.tup) if value is None]
        return board.make_move(choice(empty_spots))

    def reward(board):
        if not board.terminal:
            raise RuntimeError(f"reward called on nonterminal board {board}")
        if board.winner is board.turn:
            # It's your turn and you've already won. Should be impossible.
            raise RuntimeError(f"reward called on unreachable board {board}")
        if board.turn is (not board.winner):
            return 0  # Your opponent has just won. Bad.
        if board.winner is None:
            return 0.5  # Board is a tie
        # The winner is neither True, False, nor None
        raise RuntimeError(f"board has unknown winner type {board.winner}")

    def is_terminal(board):
        return board.terminal

    def make_move(board, index):
        tup = board.tup[:index] + (board.turn,) + board.tup[index + 1 :]
        turn = not board.turn
        winner = _find_winner(tup)
        is_terminal = (winner is not None) or not any(v is None for v in tup)
        return TicTacToeBoard(tup, turn, winner, is_terminal)

    def to_pretty_string(board):
        to_char = lambda v: ("X" if v is True else ("O" if v is False else " "))
        rows = [
            [to_char(board.tup[3 * row + col]) for col in range(3)] for row in range(3)
        ]
        return (
            "\n  1 2 3\n"
            + "\n".join(str(i + 1) + " " + " ".join(row) for i, row in enumerate(rows))
            + "\n"
        )

def _winning_combos():
    for start in range(0, 9, 3):  # three in a row
        yield (start, start + 1, start + 2)
    for start in range(3):  # three in a column
        yield (start, start + 3, start + 6)
    yield (0, 4, 8)  # down-right diagonal
    yield (2, 4, 6)  # down-left diagonal


def _find_winner(tup):
    "Returns None if no winner, True if X wins, False if O wins"
    for i1, i2, i3 in _winning_combos():
        v1, v2, v3 = tup[i1], tup[i2], tup[i3]
        if False is v1 is v2 is v3:
            return False
        if True is v1 is v2 is v3:
            return True
    return None

def new_tic_tac_toe_board():
    return TicTacToeBoard(tup=(None,) * 9, turn=True, winner=None, terminal=False)

class Player:
    def __init__(self, symbol):
        self.symbol = symbol
        self.search_space = 0

    def get_move(self, board):
        pass

class MinimaxPlayer(Player):
    def __init__(self, symbol):
        super().__init__(symbol)

    def get_move(self, board):
        self.search_space = 0
        _, move = self.minimax(board, True)
        return move

    def minimax(self, board, is_maximizing):
        self.search_space += 1
        if self.check_winner(board):
            return (-1 if is_maximizing else 1), None

        if self.check_draw(board):
            return 0, None

        best_score = -math.inf if is_maximizing else math.inf
        best_move = None
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'O' if is_maximizing else 'X'
                    score, _ = self.minimax(board, not is_maximizing)
                    board[i][j] = ' '
                    if is_maximizing:
                        if score > best_score:
                            best_score = score
                            best_move = (i, j)
                    else:
                        if score < best_score:
                            best_score = score
                            best_move = (i, j)
        return best_score, best_move

    def check_winner(self, board):
        for row in board:
            if row[0] == row[1] == row[2] and row[0] != ' ':
                return True
        for col in range(3):
            if board[0][col] == board[1][col] == board[2][col] and board[0][col] != ' ':
                return True
        if board[0][0] == board[1][1] == board[2][2] and board[0][0] != ' ':
            return True
        if board[0][2] == board[1][1] == board[2][0] and board[0][2] != ' ':
            return True
        return False

    def check_draw(self, board):
        for row in board:
            for cell in row:
                if cell == ' ':
                    return False
        return True

class AlphaBetaPlayer(MinimaxPlayer):
    def __init__(self, symbol):
        super().__init__(symbol)

    def get_move(self, board):
        self.search_space = 0
        _, move = self.minimax(board, True, -math.inf, math.inf)  # Provide alpha and beta
        return move

    def minimax(self, board, is_maximizing, alpha, beta):
        self.search_space += 1
        if self.check_winner(board):
            return (-1 if is_maximizing else 1), None

        if self.check_draw(board):
            return 0, None

        best_score = -math.inf if is_maximizing else math.inf
        best_move = None
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'O' if is_maximizing else 'X'
                    score, _ = self.minimax(board, not is_maximizing, alpha, beta)
                    board[i][j] = ' '
                    if is_maximizing:
                        if score > best_score:
                            best_score = score
                            best_move = (i, j)
                            alpha = max(alpha, best_score)
                            if beta <= alpha:
                                break
                    else:
                        if score < best_score:
                            best_score = score
                            best_move = (i, j)
                            beta = min(beta, best_score)
                            if beta <= alpha:
                                break
        return best_score, best_move

class Board:
    def __init__(self, board):
        self.board = board

    def get_empty_cells(self):
        empty_cells = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == ' ':
                    empty_cells.append((i, j))
        return empty_cells

    def is_game_over(self):
        return any(self.check_winner(symbol) for symbol in ['X', 'O']) or self.check_draw()

    def check_winner(self, symbol):
        for row in self.board:
            if all(cell == symbol for cell in row):
                return True
        for col in range(3):
            if all(self.board[row][col] == symbol for row in range(3)):
                return True
        if all(self.board[i][i] == symbol for i in range(3)) or all(self.board[i][2-i] == symbol for i in range(3)):
            return True
        return False

    def check_draw(self):
        return all(cell != ' ' for row in self.board for cell in row)

    def next_symbol(self):
        count_x = sum(row.count('X') for row in self.board)
        count_o = sum(row.count('O') for row in self.board)
        return 'X' if count_x <= count_o else 'O'

    def get_result(self):
        if any(self.check_winner(symbol) for symbol in ['X', 'O']):
            return 1
        elif self.check_draw():
            return 0
        else:
            return None

    def get_move(self, action, player):
        row, col = action
        if self.board[row][col] == ' ':
            self.board[row][col] = player
            return Board(self.board)
        else:
            return None  # Invalid move

def print_board(board):
    print("  1 2 3")
    for i, row in enumerate(board):
        print(i + 1, end=" ")
        print(" ".join(row))

def check_winner(board):
    for row in board:
        if row[0] == row[1] == row[2] and row[0] != ' ':
            return True
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] != ' ':
            return True
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != ' ':
        return True
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != ' ':
        return True
    return False

def check_draw(board):
    for row in board:
        for cell in row:
            if cell == ' ':
                return False
    return True

def player_move(board, symbol, position):
    if board[position[0]][position[1]] == ' ':
        board[position[0]][position[1]] = symbol
        return True
    else:
        print("Invalid move. Try again.")
        return False

def play_game(player1, player2):
    board = [[' ']*3 for _ in range(3)]
    print_board(board)

    players = [player1, player2]
    while not (check_winner(board) or check_draw(board)):
        for player in players:
            move = player.get_move(board)
            player_symbol = player.symbol
            player_move(board, player_symbol, move)
            print_board(board)
            if check_winner(board):
                print(f"{player_symbol} wins!")
                return
            elif check_draw(board):
                print("It's a draw!")
                return

class MCTSPlayer(Player):
    def __init__(self, symbol):
        super().__init__(symbol)
        self.tree = MCTS()
        self.rollout_count = 0

    def get_move(self, board):
        self.rollout_count = 0
        board_state = TicTacToeBoard(
            tup=tuple(cell if cell != ' ' else None for row in board for cell in row),
            turn=(self.symbol == 'X'),
            winner=None,
            terminal=False
        )

        for _ in range(50):  # Run MCTS rollouts
            self.tree.do_rollout(board_state)
            self.rollout_count += 1

        best_move_state = self.tree.choose(board_state)
        best_move = best_move_state.tup
        for i in range(3):
            for j in range(3):
                if best_move[i * 3 + j] != board[i][j]:
                    return (i, j)

# Example usage
if __name__ == "__main__":
    num_games = 3
    algorithms = [MinimaxPlayer, AlphaBetaPlayer, MCTSPlayer]
    algorithm_names = ["Minimax", "Alpha-Beta Pruning", "MCTS"]

    table_data = []

    for i in range(num_games):
        print("Let's play Tic-Tac-Toe with " + algorithm_names[i] + "!")
        player1 = algorithms[i]('X')
        player2 = algorithms[i]('O')

        start_time = time.time()
        play_game(player1, player2)
        end_time = time.time()

        avg_search_space = player1.search_space if algorithm_names[i] != "MCTS" else player1.rollout_count

        table_data.append([algorithm_names[i], end_time - start_time, avg_search_space])

    print(tabulate(table_data, headers=["Algorithm", "Time (seconds)", "Search Space (average nodes explored)"], tablefmt="grid"))
