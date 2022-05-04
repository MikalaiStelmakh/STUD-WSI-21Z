import pygame
import sys
import math
from random import choice
import copy


class BoardSizeException(Exception):
    pass


class PlayerColorException(Exception):
    pass


class Player:
    def __init__(self, color) -> None:
        self.color = color


class Field:
    def __init__(self, row: int, col: int, occupied_by: Player = None) -> None:
        self.row = row
        self.col = col
        self.occupied_by = occupied_by


class Board:
    DEFAULT_BOARD_HEIGHT = 4
    DEFAULT_BOARD_WIDTH = 5

    def __init__(self, height: int = DEFAULT_BOARD_HEIGHT,
                 width: int = DEFAULT_BOARD_WIDTH) -> None:
        if height < self.DEFAULT_BOARD_HEIGHT or width < self.DEFAULT_BOARD_WIDTH:
            raise BoardSizeException(
                "Invalid board size. Width should be >= 5 and height >= 4."
                )
        self.width = width
        self.height = height
        self.fields = [
            [Field(row, col) for col in range(width)] for row in range(height)
            ]
        self.fields_possible_to_occupy = [(height-1, x) for x in range(width)]

    def occupy_field(self, row: int, col: int, player: Player) -> None:
        """Set given player as an owner of the field at given coordinates."""
        if (row, col) in self.fields_possible_to_occupy:
            self.fields[row][col].occupied_by = player
            index = self.fields_possible_to_occupy.index((row, col))
            if row-1 >= 0:
                self.fields_possible_to_occupy[index] = (row-1, col)
            else:
                self.fields_possible_to_occupy.pop(index)

    def field_occupied_by(self, row: int, col: int) -> Player | None:
        """Returns a player that occupied field at given coordinates.
        Returns None if the field is empty."""
        return self.fields[row][col].occupied_by

    def fields_occupied_by_player(self, player: Player) -> list[Field]:
        """Returns coordinates of all fields occupied by given player."""
        return list(filter(lambda x: x.occupied_by == player, self.fields))

    def empty_fields(self) -> list[Field]:
        """Returns all empty fields."""
        return list(filter(lambda x: x.occupied_by is None, self.fields))


class ConnectFourGame:
    def __init__(self, player1: Player, player2: Player,
                 board_height: int = 4, board_width: int = 5,
                 active_player: Player = None) -> None:
        self.board_obj = Board(board_height, board_width)
        self.player1 = player1
        self.player2 = player2
        if self.player1.color.lower() == self.player2.color.lower():
            raise PlayerColorException("Players can't have same colors.")
        self.active_player = active_player or player1

    def switch_active_player(self) -> None:
        self.active_player = self.player2 if self.active_player == self.player1 else self.player1

    def perform_move(self, row: int, col: int):
        """Make a move and switch active player."""
        if self.board_obj.field_occupied_by(row, col) is None:
            self.board_obj.occupy_field(row, col, self.active_player)
            self.switch_active_player()
        return self

    def perform_move_on_copy(self, row: int, col: int) -> "Board":
        """Copies current board, performs the move on it,
        and then returns that copy.
        """
        # Create a copy of self
        new = copy.deepcopy(self)
        new.perform_move(row, col)
        return new


    def get_opponent(self, player: Player) -> Player:
        return player1 if player2 == player else player2

    def check_for_win(self) -> bool:
        """Check for win horizontally"""
        for row in range(self.board_obj.height):
            for col in range(self.board_obj.width-3):
                points = [(row, col), (row, col+1), (row, col+2), (row, col+3)]
                owners = [self.board_obj.field_occupied_by(*point) for point in points]
                if (len(set(owners)) == 1 and
                   (owners[0] == self.player1 or owners[0] == self.player2)):
                    return True

        """Check for win vertically"""
        for col in range(self.board_obj.width):
            for row in range(self.board_obj.height-3):
                points = [(row, col), (row+1, col), (row+2, col), (row+3, col)]
                owners = [self.board_obj.field_occupied_by(*point) for point in points]
                if (len(set(owners)) == 1 and
                   (owners[0] == self.player1 or owners[0] == self.player2)):
                    return True

        """Check for win positively sloped diagonal"""
        for col in range(self.board_obj.width-3):
            for row in range(self.board_obj.height-3):
                points = [(row, col), (row+1, col+1), (row+2, col+2), (row+3, col+3)]
                owners = [self.board_obj.field_occupied_by(*point) for point in points]
                if (len(set(owners)) == 1 and
                   (owners[0] == self.player1 or owners[0] == self.player2)):
                    return True

        """Check for win negatively sloped diagonal"""
        for col in range(self.board_obj.width-3):
            for row in range(self.board_obj.height-1, 2, -1):
                points = [(row, col), (row-1, col+1), (row-2, col+2), (row-3, col+3)]
                owners = [self.board_obj.field_occupied_by(*point) for point in points]
                if (len(set(owners)) == 1 and
                   (owners[0] == self.player1 or owners[0] == self.player2)):
                    return True
        return False


class ConnectFourGameInterface:
    pygame.init()
    SQUARESIZE = 100
    RADIUS = int(SQUARESIZE/2 - 5)
    FONT = pygame.font.SysFont("monospace", 75)

    def __init__(self, game_obj: ConnectFourGame) -> None:
        self.game = game_obj
        self.window_width = self.game.board_obj.width * self.SQUARESIZE
        self.window_height = (self.game.board_obj.height + 1) * self.SQUARESIZE
        self.window = pygame.display.set_mode((self.window_width, self.window_height))

    def draw_board(self, color1, color2):
        board_width = self.game.board_obj.width
        board_height = self.game.board_obj.height
        for col in range(board_width-1, -1, -1):
            for row in range(board_height-1, -1, -1):
                pygame.draw.rect(
                    self.window, color1,
                    (col*self.SQUARESIZE, row*self.SQUARESIZE+self.SQUARESIZE,
                     self.SQUARESIZE, self.SQUARESIZE)
                )
                pygame.draw.circle(
                    self.window, color2,
                    (int(col*self.SQUARESIZE+self.SQUARESIZE/2),
                     int(row*self.SQUARESIZE+self.SQUARESIZE+self.SQUARESIZE/2)),
                    self.RADIUS
                )
        for col in range(board_width):
            for row in range(board_height):
                if self.game.board_obj.field_occupied_by(row, col):
                    pygame.draw.circle(
                        self.window,
                        self.game.board_obj.field_occupied_by(row, col).color,
                        (int(col*self.SQUARESIZE+self.SQUARESIZE/2),
                         self.window_height-int((board_height-row-1)*self.SQUARESIZE+self.SQUARESIZE/2)),
                        self.RADIUS)
        pygame.display.update()

    def perform_move(self, posx):
        col = int(math.floor(posx/self.SQUARESIZE))
        coords = [
            elem for elem in self.game.board_obj.fields_possible_to_occupy if elem[1] == col
            ]
        if coords:
            row = coords[0][0]
            self.game.perform_move(row, col)

    def display_mouse_motion(self, top_bar_color, posx):
        pygame.draw.rect(self.window, top_bar_color,
                        (0, 0, self.window_width, self.SQUARESIZE))
        pygame.draw.circle(
            self.window,
            self.game.active_player.color,
            (posx, int(self.SQUARESIZE/2)),
            self.RADIUS)

    def display_winner_label(self, winner):
        label = self.FONT.render(f"{winner.color.upper()} wins!", 1, winner.color)
        self.window.blit(label, (40, 10))

    def display_draw_label(self):
        label = self.FONT.render("Draw!", 1, 'Red')
        self.window.blit(label, (150, 10))

    def play(self):
        """Start a game between user and random bot."""
        game_over = False
        top_bar_color = 'White'
        empty_circles_color = 'White'
        color = 'Blue'
        self.draw_board(color, empty_circles_color)
        while not game_over:
            # User's turn
            if self.game.active_player == player1:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()

                    if event.type == pygame.MOUSEMOTION:
                        self.display_mouse_motion(top_bar_color, event.pos[0])
                    pygame.display.update()

                    if self.game.active_player == self.game.player1:
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            pygame.draw.rect(self.window, 'White',
                                            (0, 0, self.window_width, self.SQUARESIZE))
                            self.perform_move(posx=event.pos[0])
            # Random bot's turn
            else:
                move = choice(self.game.board_obj.fields_possible_to_occupy)
                self.game.perform_move(*move)

            if self.game.check_for_win():
                game_over = True
                winner = self.game.player1 if self.game.active_player == self.game.player2 else self.game.player2
                self.display_winner_label(winner)
            if len(self.game.board_obj.fields_possible_to_occupy) == 0:
                game_over = True
                self.display_draw_label()
            self.draw_board(color, empty_circles_color)

            if game_over:
                pygame.time.wait(5000)


if __name__ == "__main__":
    player1 = Player('Red')
    player2 = Player('Black')
    BOARD_WIDTH = 5
    BOARD_HEIGHT = 5
    game = ConnectFourGame(player1, player2, BOARD_HEIGHT, BOARD_WIDTH)
    game_interface = ConnectFourGameInterface(game)
    game_interface.play()

