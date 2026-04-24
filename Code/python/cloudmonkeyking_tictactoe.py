#
# Tic Tac Toe with Reinforcement Learning
#
# References:
#   - https://www.youtube.com/playlist?list=PLLfIBXQeu3aanwI5pYz6QyzYtnBEgcsZ8
#   - https://github.com/maksimKorzh/tictactoe-mtcs/tree/master/src/tictactoe
#   - https://www.youtube.com/watch?v=UXW2yZndl7U
#   - matthewdeakos.me/2018/03/10/monte-carlo-tree-search/
#   - http://matthewdeakos.me/2018/07/03/integrating-monte-carlo-tree-search-and-neural-networks/

from copy import deepcopy
from cloudmonkeyking_mcts import TreeNode, MCTS

# Tic Tac Toe Board class
class Board:
    def __init__(self, board=None):  # Copy constructor
        self.player_to_make_move = 'X'
        self.player_waiting_to_move = 'O'
        self.empty_cell = '.'
        self.position = {}

        self.init_board()

        if board:
            self.__dict__ = deepcopy(board.__dict__)

    def init_board(self):
        for row in range(3):
            for col in range(3):
                self.position[row, col] = self.empty_cell

    def make_move(self, row, col):
        board_next = Board(self)
        board_next.position[row, col] = self.player_to_make_move
        # switch
        (board_next.player_to_make_move, board_next.player_waiting_to_move) = \
            (board_next.player_waiting_to_move, board_next.player_to_make_move)
        return board_next

    def is_draw(self):
        for row in range(3):
            for col in range(3):
                if self.position[row, col] == self.empty_cell:
                    return False
        return True

    def is_won(self):
        # check if horizontal, vertical and diagonal sequences have same letter

        ##################################
        # vertical sequence detection
        ##################################

        # loop over board columns
        for col in range(3):
            # define winning sequence list
            winning_sequence = []

            # loop over board rows
            for row in range(3):
                # if found same next element in the row
                if self.position[row, col] == self.player_waiting_to_move: # will get swapped, so check only one
                    # update winning sequence
                    winning_sequence.append((row, col))

                # if we have 3 elements in the row
                if len(winning_sequence) == 3:
                    # return the game is won state
                    return True

        ##################################
        # horizontal sequence detection
        ##################################

        # loop over board columns
        for row in range(3):
            # define winning sequence list
            winning_sequence = []

            # loop over board rows
            for col in range(3):
                # if found same next element in the row
                if self.position[row, col] == self.player_waiting_to_move:
                    # update winning sequence
                    winning_sequence.append((row, col))

                # if we have 3 elements in the row
                if len(winning_sequence) == 3:
                    # return the game is won state
                    return True

        ##################################
        # 1st diagonal sequence detection
        ##################################

        # define winning sequence list
        winning_sequence = []

        # loop over board rows
        for row in range(3):
            # init column
            col = row

            # if found same next element in the row
            if self.position[row, col] == self.player_waiting_to_move:
                # update winning sequence
                winning_sequence.append((row, col))

            # if we have 3 elements in the row
            if len(winning_sequence) == 3:
                # return the game is won state
                return True

        ##################################
        # 2nd diagonal sequence detection
        ##################################

        # define winning sequence list
        winning_sequence = []

        # loop over board rows
        for row in range(3):
            # init column
            col = 3 - row - 1

            # if found same next element in the row
            if self.position[row, col] == self.player_waiting_to_move:
                # update winning sequence
                winning_sequence.append((row, col))

            # if we have 3 elements in the row
            if len(winning_sequence) == 3:
                # return the game is won state
                return True

        # by default return non-winning state
        return False

    # generate legal moves to play in the current position
    def generate_states(self):
        # define states list (move list - list of available actions to consider)
        actions = []

        # loop over board rows
        for row in range(3):
            # loop over board columns
            for col in range(3):
                # make sure that current square is empty
                if self.position[row, col] == self.empty_cell:
                    # append available action/board state to action list
                    actions.append(self.make_move(row, col)) # updated move gets added

        # return the list of available actions (board class instances)
        return actions

    # main game loop
    def game_loop(self):
        print('\n  Tic Tac Toe by Code Monkey King\n')
        print('  Type "exit" to quit the game')
        print('  Move format [x,y], eg [1,2] where 1 is row and 2 is column')

        # print board
        print(self)

        mcts = MCTS()

        # game loop
        while True:
            # get user input
            user_input = input('> ')

            # escape condition
            if user_input == 'exit': break

            # skip empty input
            if user_input == '': continue

            try:
                # parse user input (move format [row, col]: 1,2)
                row = int(user_input.split(',')[0]) - 1
                col = int(user_input.split(',')[1]) - 1

                # check move legality
                if self.position[row, col] != self.empty_cell:
                    print(' Illegal move!')
                    continue

                # make move on board
                self = self.make_move(row, col)

                # print board
                print(self)

                # search for the best move
                best_move = mcts.search(self)

                # legal moves available
                try:
                    # make AI move here
                    self = best_move.board

                # game over
                except:
                    pass

                # print board
                print(self)

                # check if the game is won
                if self.is_won():
                    print('player "%s" has won the game!\n' % self.player_waiting_to_move)
                    break

                # check if the game is drawn
                elif self.is_draw():
                    print('Game is drawn!\n')
                    break

            except Exception as e:
                print('  Error:', e)
                print('  Illegal command!')
                print('  Move format [x,y]: 1,2 where 1 is row and 2 is col')

    def __str__(self):
        return_str = ""
        for row in range(3):
            for col in range(3):
                return_str += " " + self.position[row, col] + " "
            return_str += "\n"

        if self.player_to_make_move == "X":
            return_str = "\n -------------- \n" + "   X to move    " + "\n -------------- \n" + return_str
        else:
            return_str = "\n -------------- \n" + "   O to move    " + "\n -------------- \n" + return_str

        return return_str


if __name__ == "__main__":
    # create board instance
    board = Board()

    # start game loop
    board.game_loop()