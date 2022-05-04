from connect_four import ConnectFourGame, Player
from minimax import PlayerAI, minimax, play
import copy
from itertools import product
import json


PLAYER1COLOR = "RED"
PLAYER2COLOR = "BLACK"
ITERATIONS = 50

DEPTHS = [0, 1, 2, 3]
BOARD_SIZES = [(6, 5)]



def benchmark(game: ConnectFourGame) -> list[Player | None]:
    winners = list(map(
        lambda x: x.color,
        list(filter(lambda x: x, [play(copy.deepcopy(game)) for _ in range(ITERATIONS)]))
        ))
    return (winners.count(game.player1.color), winners.count(game.player2.color))


if __name__ == "__main__":
    depths = list(product(DEPTHS, DEPTHS))
    result = {
        "Benchmarks": [
            {
                "Board height": size[0],
                "Board width": size[1],
                "Depths": [
                    {
                        "Player 1 depth:": depth[0],
                        "Player 2 depth": depth[1],
                        "Score": {}
                    } for depth in depths
                    ]
            } for size in BOARD_SIZES]}

    for board_size_i, (board_height, board_width) in enumerate(BOARD_SIZES):
        for depth_i, (depth1, depth2) in enumerate(depths):
            player1 = PlayerAI(PLAYER1COLOR, depth1)
            player2 = PlayerAI(PLAYER2COLOR, depth2)
            game = ConnectFourGame(player1, player2, board_height, board_width)
            score = benchmark(game)
            print(f'{board_height=}, {board_width=}, {depth1=}, {depth2=},  {score=}')
            result["Benchmarks"][board_size_i]["Depths"][depth_i]["Score"]["Player1"] = score[0]
            result["Benchmarks"][board_size_i]["Depths"][depth_i]["Score"]["Player2"] = score[1]
    with open('lab3/.benchmarks/test2.json', 'w') as fp:
        json.dump(result, fp, indent=2)