from connect_four import Player, ConnectFourGame, ConnectFourGameInterface
import argparse
import math
import pygame
from random import choice


class PlayerAI(Player):
    """Player with difficulty.
    Random if depth == 0."""
    def __init__(self, color: str, depth: int = 0) -> None:
        super().__init__(color)
        self.depth = depth


def evaluation(game: ConnectFourGame) -> int:
    """Calculates number of possible combinations of four pieces for player2
    and substracts it from player1."""
    score1, score2 = 0, 0
    for row in range(game.board_obj.height):
        for col in range(game.board_obj.width-3):
            points = [(row, col), (row, col+1), (row, col+2), (row, col+3)]
            owners = [game.board_obj.field_occupied_by(*point) for point in points]
            if game.player2 not in owners:
                score1 += 1
            if game.player1 not in owners:
                score2 += 1

    for col in range(game.board_obj.width):
        for row in range(game.board_obj.height-3):
            points = [(row, col), (row+1, col), (row+2, col), (row+3, col)]
            owners = [game.board_obj.field_occupied_by(*point) for point in points]
            if game.player2 not in owners:
                score1 += 1
            if game.player1 not in owners:
                score2 += 1

    for col in range(game.board_obj.width-3):
        for row in range(game.board_obj.height-3):
            points = [(row, col), (row+1, col+1), (row+2, col+2), (row+3, col+3)]
            owners = [game.board_obj.field_occupied_by(*point) for point in points]
            if game.player2 not in owners:
                score1 += 1
            if game.player1 not in owners:
                score2 += 1

    for col in range(game.board_obj.width-3):
        for row in range(game.board_obj.height-1, 2, -1):
            points = [(row, col), (row-1, col+1), (row-2, col+2), (row-3, col+3)]
            owners = [game.board_obj.field_occupied_by(*point) for point in points]
            if game.player2 not in owners:
                score1 += 1
            if game.player1 not in owners:
                score2 += 1
    return score1 - score2


def minimax_recur(game: ConnectFourGame, maximizing_player: Player, depth: int, alpha=-math.inf, beta = math.inf, moves = 1):
    # Give winner 1000 points
    if game.check_for_win():
        if game.active_player.color != maximizing_player.color:
            return 1000/moves
        else:
            return -1000/moves
    if not depth or not game.board_obj.fields_possible_to_occupy:
        return evaluation(game)

    if game.active_player.color == maximizing_player.color:
        for move in game.board_obj.fields_possible_to_occupy:
            alpha = max(
                alpha,
                minimax_recur(game.perform_move_on_copy(*move), maximizing_player, depth - 1, alpha, beta, moves+1)
                )
            if alpha >= beta:
                return beta
        return alpha

    else:
        for move in game.board_obj.fields_possible_to_occupy:
            beta = min(
                beta,
                minimax_recur(game.perform_move_on_copy(*move), maximizing_player, depth - 1, alpha, beta, moves+1)
                )
            if alpha >= beta:
                return alpha
    return beta


def minimax(game: ConnectFourGame, maximizing_player: bool, depth) -> tuple[int, int]:
    """Finds the best move using minimax algorithm with alpha-beta pruning.
    If multiple steps lead to the same result, returns random one."""
    func = max if maximizing_player else min
    move_score_list = [
        (move, minimax_recur(game.perform_move_on_copy(*move), game.player1, depth - 1))
        for move in game.board_obj.fields_possible_to_occupy
        ]
    best_score = func(move_score_list, key=lambda x: x[1])[1]
    # if func == min:
    #     print([move_score[1] for move_score in move_score_list])
    return choice([move_score for move_score in move_score_list if move_score[1] == best_score])[0]


def play(game: ConnectFourGame) -> Player | None:
    """Start a game between two AIs."""
    while True:
        if ((game.active_player.color == game.player1.color and not game.player1.depth == 0)
            or (game.active_player.color == game.player2.color and not game.player2.depth == 0)):
            maximizing_player = True if game.active_player == game.player1 else False
            move = minimax(game, maximizing_player, game.active_player.depth)
        else:
            move = choice(game.board_obj.fields_possible_to_occupy)
        game.perform_move(*move)
        if game.check_for_win():
            winner = game.player1 if game.active_player == game.player2 else game.player2
            return winner
        elif len(game.board_obj.fields_possible_to_occupy) == 0:
            return None


def play_gui(game: ConnectFourGame) -> Player | None:
    """Start a game between two AIs with GUI."""
    game_over = False
    interface_obj = ConnectFourGameInterface(game)
    top_bar_color = 'White'
    empty_circles_color = 'White'
    color = 'Blue'
    interface_obj.draw_board(color, empty_circles_color)
    while not game_over:
        pygame.draw.rect(interface_obj.window, top_bar_color,
                         (0, 0, interface_obj.window_width, interface_obj.SQUARESIZE))
        if ((game.active_player.color == game.player1.color and not game.player1.depth == 0)
            or (game.active_player.color == game.player2.color and not game.player2.depth == 0)):
            maximizing_player = True if game.active_player == game.player1 else False
            move = minimax(game, maximizing_player, game.active_player.depth)
        else:
            move = choice(game.board_obj.fields_possible_to_occupy)
        game.perform_move(*move)
        if game.check_for_win():
            winner = game.player1 if game.active_player == game.player2 else game.player2
            interface_obj.display_winner_label(winner)
            game_over = True
        elif len(game.board_obj.fields_possible_to_occupy) == 0:
            game_over = True
            interface_obj.display_draw_label()
            winner = None
        interface_obj.draw_board(color, empty_circles_color)

        if game_over:
            pygame.time.wait(5000)
            return winner
        else:
            pygame.time.wait(500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        """Connect four game with minimax algorithm."""
        )
    parser.add_argument('width', type=int, nargs=1,
                        help='number of columns')
    parser.add_argument('height', type=int, nargs=1,
                        help='number of rows')
    parser.add_argument('--player1', metavar="DEPTH", action="store",
                        type=int, default=3,
                        help='difficulty of player1 (default: 3). If zero than moves randomly.')
    parser.add_argument('--player2', metavar="DEPTH", action="store",
                        type=int, default=3,
                        help='difficulty of player2 (default: 3). If zero than moves randomly.')
    parser.add_argument('--player1Color', metavar="COLOR", action="store",
                        type=str, default="RED",
                        help='color of player1 (default: RED)')
    parser.add_argument('--player2Color', metavar="COLOR", action="store",
                        type=str, default="BLACK",
                        help='color of player1 (default: BLACK)')
    parser.add_argument('--gui', action='store_true',
                        help='show the game')
    args = parser.parse_args()

    BOARD_WIDTH = args.width[0]
    BOARD_HEIGHT = args.height[0]
    player1 = PlayerAI(args.player1Color, args.player1)
    player2 = PlayerAI(args.player2Color, args.player2)
    game = ConnectFourGame(player1, player2, BOARD_HEIGHT, BOARD_WIDTH)
    if args.gui:
        winner = play_gui(game)
    else:
        winner = play(game)

    if winner:
        print(winner.color, "wins!")
    else:
        print('Draw!')
