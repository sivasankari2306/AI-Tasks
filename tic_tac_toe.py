import math

def print_board(board):
    for row in board:
        print("|".join(row))
    print("\n")

def is_winner(board, player):
    return any(all(cell == player for cell in row) for row in board) or \
           any(all(row[i] == player for row in board) for i in range(3)) or \
           all(board[i][i] == player for i in range(3)) or \
           all(board[i][2 - i] == player for i in range(3))

def minimax(board, depth, is_maximizing):
    if is_winner(board, "O"): return 10 - depth
    if is_winner(board, "X"): return depth - 10
    if all(cell != " " for row in board for cell in row): return 0

    if is_maximizing:
        best_score = -math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    board[i][j] = "O"
                    score = minimax(board, depth + 1, False)
                    board[i][j] = " "
                    best_score = max(score, best_score)
        return best_score
    else:
        best_score = math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    board[i][j] = "X"
                    score = minimax(board, depth + 1, True)
                    board[i][j] = " "
                    best_score = min(score, best_score)
        return best_score

def best_move(board):
    best_score = -math.inf
    move = None
    for i in range(3):
        for j in range(3):
            if board[i][j] == " ":
                board[i][j] = "O"
                score = minimax(board, 0, False)
                board[i][j] = " "
                if score > best_score:
                    best_score = score
                    move = (i, j)
    return move

def play_game():
    board = [[" " for _ in range(3)] for _ in range(3)]
    while True:
        print_board(board)
        if is_winner(board, "O"):
            print("AI Wins!")
            break
        elif is_winner(board, "X"):
            print("You Win!")
            break
        elif all(cell != " " for row in board for cell in row):
            print("It's a Tie!")
            break

        x, y = map(int, input("Enter your move (row col): ").split())
        if board[x][y] == " ":
            board[x][y] = "X"
            move = best_move(board)
            if move:
                board[move[0]][move[1]] = "O"

play_game()
