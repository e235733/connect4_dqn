"""Display the original 11-move example."""

from connect4 import Connect4


def main():
    game = Connect4(6, 7)
    columns = [3, 4, 3, 2, 2, 3, 4, 5, 2, 3, 5]

    print("Initial board:")
    game.render()

    for turn, column in enumerate(columns, 1):
        player = game.current_player
        _, done, winner = game.step(column)
        print(f"\nMove {turn}: {'first (o)' if player == 1 else 'second (x)'} -> column {column}")
        game.render()
        if done:
            print(f"Game over. Winner: {winner} (1=o, -1=x, 0=draw)")
            break


if __name__ == "__main__":
    main()
