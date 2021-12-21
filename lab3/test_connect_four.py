from connect_four import Player, Field, Board, ConnectFourGame, BoardSizeException
import pytest


def test_create_player():
    Player('white')


def test_create_field_without_player():
    Field(1, 1)


def test_create_field_with_player():
    player = Player('White')
    assert Field(1, 1, player).occupied_by == player


def test_board_create_with_default_size():
    board = Board()
    assert len(board.fields) == board.height
    assert len(board.fields[0]) == board.width


def test_board_create_with_given_size():
    board = Board(5, 5)
    assert len(board.fields) == board.height
    assert len(board.fields[0]) == board.width
    assert board.fields_possible_to_occupy == [
        (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)
        ]


def test_board_create_with_given_incorrect_size():
    with pytest.raises(BoardSizeException):
        Board(2, 2)


def test_board_field_occupied_by():
    board = Board()
    assert board.field_occupied_by(3, 3) is None


def test_board_occupy_field():
    board = Board()
    player = Player('White')
    row, col = 3, 3
    assert board.field_occupied_by(row, col) is None
    assert board.fields_possible_to_occupy == [
        (3, 0), (3, 1), (3, 2), (3, 3), (3, 4)
    ]
    board.occupy_field(3, 3, player)
    assert board.field_occupied_by(row, col) == player
    assert board.fields_possible_to_occupy == [
        (3, 0), (3, 1), (3, 2), (2, 3), (3, 4)
    ]


def test_ConnectFourGame_create():
    player1 = Player('White')
    player2 = Player('Black')
    game = ConnectFourGame(player1, player2)
    assert game.player1 == player1
    assert game.player2 == player2
    assert game.active_player == player1


def test_ConnectFourGame_switch_active_player():
    player1 = Player('White')
    player2 = Player('Black')
    game = ConnectFourGame(player1, player2)
    assert game.active_player == player1
    game.switch_active_player()
    assert game.active_player == player2


def test_ConnectFourGame_perform_move():
    player1 = Player('White')
    player2 = Player('Black')
    game = ConnectFourGame(player1, player2, 5, 5)
    game.perform_move(4, 3)
    assert game.board_obj.field_occupied_by(4, 3) == player1
    assert game.active_player == player2


def test_ConnectFourGame_check_for_win_horizontally():
    player1 = Player('White')
    player2 = Player('Black')
    game = ConnectFourGame(player1, player2, 5, 5)
    game.perform_move(4, 0)
    assert game.check_for_win() is False
    game.perform_move(3, 0)
    assert game.check_for_win() is False
    game.perform_move(4, 1)
    assert game.check_for_win() is False
    game.perform_move(2, 0)
    assert game.check_for_win() is False
    game.perform_move(4, 2)
    assert game.check_for_win() is False
    game.perform_move(3, 1)
    assert game.check_for_win() is False
    game.perform_move(4, 3)
    """
      Board
    - - - - -
    - - - - -
    o - - - -
    o o - - -
    x x x x -
    """
    assert game.check_for_win() is True


def test_ConnectFourGame_check_for_win_vertically():
    player1 = Player('White')
    player2 = Player('Black')
    game = ConnectFourGame(player1, player2, 5, 5)
    game.perform_move(4, 0)
    assert game.check_for_win() is False
    game.perform_move(3, 0)
    assert game.check_for_win() is False
    game.perform_move(4, 1)
    assert game.check_for_win() is False
    game.perform_move(2, 0)
    assert game.check_for_win() is False
    game.perform_move(4, 2)
    assert game.check_for_win() is False
    game.perform_move(1, 0)
    assert game.check_for_win() is False
    game.perform_move(3, 2)
    assert game.check_for_win() is False
    game.perform_move(0, 0)
    """
      Board
    o - - - -
    o - - - -
    o - - - -
    o - x - -
    x x x - -
    """
    assert game.check_for_win() is True


lst = [
    [1, 2, 3],
    [1, 2, 3],
    [1, 2, 3]
]
print([lst[row][len(lst)//2] for row in range(len(lst))])